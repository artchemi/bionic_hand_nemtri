"""
    Description:
        Achieves:
            - Data Preprocessing over Ninapro DataBase5
            - Training finetune-base model (Saving weights along the way)
            - Visualize training logs (model accuracy and loss during training)
            
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""
from dataset import *
from model import *
import config
import tensorflow as tf
import os

import mlflow

mlflow.set_experiment('Test trials')
mlflow.tensorflow.autolog(disable=True)    # Autologging

import random    # Fix seed

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
    emg, label = folder_extract(    # TODO: Затрекать данные в MLFlow
        config.folder_path,
        exercises=config.exercises,
        myo_pref=config.myo_pref
    )

    # Apply Standarization to data, save collected MEAN and STANDARD DEVIATION in the dataset to json
    emg = standarization(emg, config.std_mean_path)
    # Extract sEMG signals for wanted gestures.
    gest = gestures(emg, label, targets=config.targets)
    # Perform train test split
    train_gestures, test_gestures = train_test_split(gest, rand_seed=seed)
    
    # NOTE: optional visualization that graphs class/gesture distributions
    # plot_distribution(train_gestures)
    # plot_distribution(test_gestures)
    
    # Convert sEMG data to signal images.
    X_train, y_train = apply_window(train_gestures, window=config.window, step=config.step)
    # Convert sEMG data to signal images.
    X_test, y_test = apply_window(test_gestures, window=config.window, step=config.step)
    
    X_train = X_train.reshape(-1, 8, config.window, 1)
    X_test = X_test.reshape(-1, 8, config.window, 1)
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    print("Shape of Inputs:\n")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print("Data Type of Inputs\n")
    print(X_train.dtype)
    print(X_test.dtype)
    print("\n")
    
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
    
    # Start training (And saving weights along training)
    history = train_model(
        cnn, X_train, y_train, X_test, y_test,
        config.batch_size, save_path=config.save_path, epochs=config.epochs,    # epochs=config.epochs 
        patience=config.patience, lr=config.inital_lr
    )
    
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