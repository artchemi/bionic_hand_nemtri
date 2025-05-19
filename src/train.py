"""
    Description:
        Achieves:
            - Data Preprocessing over Ninapro DataBase5
            - Training finetune-base model (Saving weights along the way)
            - Visualize training logs (model accuracy and loss during training)
            
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""
import tensorflow as tf
import os
import sys
import mlflow
import random
import seaborn as sns
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from dataset import *
from model import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
import config

   

mlflow.set_experiment('Test trials')
mlflow.tensorflow.autolog(disable=True)    # Autologging

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed) 

# os.environ['TF_DETERMINISTIC_OPS'] = '1'    # For GPU training
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def main():
    # NOTE: Check if Utilizing GPU device
    print(tf.config.list_physical_devices('GPU'))
    
    # NOTE: Data Preprocessings
    
    # Get sEMG samples and labels. (shape: [num samples, 8(sensors/channels)])
    emg, label, emg_person, label_person = folder_extract(    # TODO: Затрекать данные в MLFlow
        config.folder_path,
        exercises=config.exercises,
        myo_pref=config.myo_pref,
        test_exception='S2_E2'
    )

    # Apply Standarization to data, save collected MEAN and STANDARD DEVIATION in the dataset to json
    emg = standarization(emg, config.std_mean_path)
    emg_person = standarization(emg_person, save_path=None)  # уже есть stats

    print(f'Размерность общей выборки - {emg.shape}')
    print(f'Размерность выборки пользователя - {emg_person.shape}')

    # Extract sEMG signals for wanted gestures.
    gest = gestures(emg, label, targets=config.targets, relax_shrink=None)
    gest_person = gestures(emg_person, label_person, targets=config.targets, relax_shrink=None)

    # print(np.unique(gest))
    # Perform train test split
    train_gestures, test_gestures = train_test_split(gest, rand_seed=seed)
    person_gestures = gestures2dict(gest_person)
    
    # NOTE: optional visualization that graphs class/gesture distributions
    # plot_distribution(train_gestures)
    # plot_distribution(test_gestures)
    
    # Convert sEMG data to signal images.
    X_train, y_train = apply_window(train_gestures, window=config.window, step=config.step)
    # Convert sEMG data to signal images.
    X_test, y_test = apply_window(test_gestures, window=config.window, step=config.step)
    X_person_test, y_person_test = apply_window(person_gestures, window=config.window, step=config.step)
    
    X_train = X_train.reshape(-1, 8, config.window, 1)
    X_test = X_test.reshape(-1, 8, config.window, 1)
    X_person_test = X_person_test.reshape(-1, 8, config.window, 1)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_person_test = X_person_test.astype(np.float32)

    print(np.unique(y_train, return_counts=True)[1])

    data = {
        'Class': ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
        'Train': np.unique(y_train, return_counts=True)[1],
        'Test': np.unique(y_test, return_counts=True)[1],
        'User Test': np.unique(y_person_test, return_counts=True)[1]
        }
    
    df = pd.DataFrame(data)

    bar_width = 0.35
    indices = np.arange(len(df))  # позиции по оси Y

    # Создание графика
    plt.figure(figsize=(8, 4))

    # Первый столбец (Precision)
    plt.bar(indices, df['Train'], label='Train', color='skyblue')

    # Второй столбец (Recall)
    plt.bar(indices + bar_width*0.5, df['Test'], label='Test', color='lightgreen')

    plt.bar(indices - bar_width*0.5, df['User Test'], label='User Test')

    # Настройка осей и подписей
    plt.xlabel('Count')
    plt.yticks(indices, df['Class'])
    plt.title('Gestures distribution per sets')
    plt.legend()
    # plt.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    print("Shape of Inputs:\n")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print("Data Type of Inputs\n")
    print(X_train.dtype)
    print(X_test.dtype)
    print("Размерность выборки для пользователя")
    print(X_person_test.shape)
    print(X_person_test.dtype)
    
    # Get Tensorflow model
    cnn = get_model(
        num_classes=config.num_classes,
        filters=config.filters,
        neurons=config.neurons,
        dropout=config.dropout,
        kernel_size=config.k_size,
        input_shape=config.in_shape,
        pool_size=config.p_kernel
    )

    params = {'num_classes': config.num_classes, 'n_neurons': config.neurons, 'filter_1': config.filters[0], 'filter_2': config.filters[1], 'dropout': config.dropout}
    mlflow.log_params(params)    # Логируем параметры модели

    print(cnn.summary())
    
    # NOTE: Обучение baseline модели
    history_train = train_model(
        cnn, X_train, y_train, X_test, y_test,
        config.batch_size, save_path=config.save_path, epochs=50,    # epochs=config.epochs 
        patience=config.patience, lr=config.inital_lr
    )

    # print(np.argmax(cnn(X_person_test), axis=1))
    print('Репорт бейзлайн модели на тестовой выборке')
    print(classification_report(y_test, np.argmax(cnn(X_test), axis=1)))

    print('Репорт бейзлайн модели на пользователе')
    print(classification_report(y_person_test, np.argmax(cnn(X_person_test), axis=1)))

    history_train = train_model(
        cnn, X_person_test, y_person_test, X_person_test, y_person_test,
        config.batch_size, save_path=config.save_path, epochs=10,    # epochs=config.epochs 
        patience=config.patience, lr=config.inital_lr
    )

    print('Репорт дообученной модели на пользователе')
    print(classification_report(y_person_test, np.argmax(cnn(X_person_test), axis=1)))


    # Вычисление матрицы ошибок
    cm = confusion_matrix(y_person_test, np.argmax(cnn(X_person_test), axis=1))
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']

    # Визуализация
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.title('Матрица ошибок')
    plt.tight_layout()
    plt.show()
    
    # print(classification_report(y_person_test, np.argmax(cnn(X_person_test), axis=1)))

    # results = cnn(X_person_test)
    # print(f'Результаты baseline модели: {results[1]}')
    # # NOTE: Загрузка предобученной модели
    # model = get_model(
    #     num_classes=config.num_classes,
    #     filters=config.filters,
    #     neurons=config.neurons,
    #     dropout=config.dropout,
    #     kernel_size=config.k_size,    # точно так же
    #     input_shape=config.in_shape,   # точно так же
    #     pool_size=  config.p_kernel
    # )

    # print('Загрузка весов...')
    # model.load_weights(config.save_path)
    
    # print('Комплияция модели...')
    # # NOTE: Optional test for loaded model's performance
    # model.compile(
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    #         loss='sparse_categorical_crossentropy',
    #         metrics=['accuracy']
    #     )
    # See if weights were the same
    # results = model.evaluate(X_person_test, y_person_test)
    # print(f'Результаты baseline модели: {results[1]}')

    # history_fine = train_model(
    #     model, X_person_test, y_person_test, X_person_test, y_person_test,
    #     config.batch_size, save_path=config.save_path, epochs=1,    # epochs=config.epochs 
    #     patience=config.patience, lr=config.inital_lr
    # )
    # print(f'Результаты после дообучения: {model.evaluate(X_person_test, y_person_test)[1]}')
    

    # NOTE: дообучение на emg_test и label_test, имитация передачи протеза пользователю
    
    # print(history.history)
    # for epoch in range(config.epochs):    # range(config.epochs)
    #     history = train_model(
    #         cnn, X_train, y_train, X_test, y_test,
    #         config.batch_size, save_path=config.save_path, epochs=1,    # epochs=config.epochs 
    #         patience=config.patience, lr=config.inital_lr
    #     )

        # mlflow.log_metrics({'train_loss': history.history['loss'][0],
        #                     'train_accuracy': history.history['accuracy'][0],
        #                     'val_loss': history.history['val_loss'][0],
        #                     'val_accuracy': history.history['val_accuracy'][0]},
        #                     step=epoch+1)


    # # Visualize accuarcy and loss logs
    # plot_logs(history, acc=True, save_path=config.acc_log)
    # plot_logs(history, acc=False, save_path=config.loss_log)
    
    # # Load pretrained model
    # model = get_model(
    #     num_classes=config.num_classes,
    #     filters=config.filters,
    #     neurons=config.neurons,
    #     dropout=config.dropout
    # )
    # model.load_weights(config.save_path)
    
    # # NOTE: Optional test for loaded model's performance
    # model.compile(
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    #         loss='sparse_categorical_crossentropy',
    #         metrics=['accuracy'],
    #     )
    # # See if weights were the same
    # model.evaluate(X_test, y_test)
    
    # # Test with finetune model. (last classifier block removed from base model)
    # finetune_model = get_finetune(config.save_path, config.prev_params, num_classes=config.num_classes)
    # print("finetune model loaded!")
    
    # NOTE: You can load finetune model like this too.
    # finetune_model = create_finetune(model, num_classes=4)
    # finetune_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy'],
    # )
    # finetune_model.evaluate(X_test, y_test)


if __name__ == '__main__':
    with mlflow.start_run():    # Контекстный менеджер для запуска трекера
        main()