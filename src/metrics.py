import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))    # Импортируем корневую директорию
import config


NUM_CLASSES = config.num_classes

class MacroPrecision(tf.keras.metrics.Metric):
    """Подкласс метрик для расчета Precision. Считает Precision для каждого класса отдельно, а зтем усредняет.
    (Precision_1 + Precision_2 + ... + Precision_N) / N.
    Вес для всех классов одинаковый.

    Args:
        tf (_type_): Родительский класс.
    """
    def __init__(self, num_classes=NUM_CLASSES, name='macro_precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precisions = [
            tf.keras.metrics.Precision(class_id=i)
            for i in range(num_classes)
        ]
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Определяем формат y_true
        if len(y_true.shape) == 2 and y_true.shape[1] == self.num_classes:
            # One-hot encoded формат - преобразуем в классы
            y_true = tf.argmax(y_true, axis=-1)
        elif len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            # Sparse формат - оставляем как есть
            y_true = tf.squeeze(y_true)
        else:
            raise ValueError(f"Неподдерживаемый формат меток. Ожидается one-hot или sparse, получено {y_true.shape}")
        
        # Обновляем метрики для каждого класса
        for p in self.precisions:
            p.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        return tf.reduce_mean([p.result() for p in self.precisions])
    
    def reset_state(self):
        for p in self.precisions:
            p.reset_state()


class MacroRecall(tf.keras.metrics.Metric):
    """Подкласс метрик для расчета Recall. Считает Recall для каждого класса отдельно, а зтем усредняет.
    (Recall_1 + Recall_2 + ... + Recall_N) / N.
    Вес для всех классов одинаковый.

    Args:
        tf (_type_): Родительский класс.
    """
    def __init__(self, num_classes=NUM_CLASSES, name='macro_recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.recalls = [
            tf.keras.metrics.Recall(class_id=i)
            for i in range(num_classes)
        ]
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Определяем формат y_true
        if len(y_true.shape) == 2 and y_true.shape[1] == self.num_classes:
            # One-hot encoded формат - преобразуем в классы
            y_true = tf.argmax(y_true, axis=-1)
        elif len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            # Sparse формат - оставляем как есть
            y_true = tf.squeeze(y_true)
        else:
            raise ValueError(f"Неподдерживаемый формат меток. Ожидается one-hot или sparse, получено {y_true.shape}")
        
        # Обновляем метрики для каждого класса
        for r in self.recalls:
            r.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        return tf.reduce_mean([r.result() for r in self.recalls])
    
    def reset_state(self):
        for r in self.recalls:
            r.reset_state()
    

class MacroF1Score(tf.keras.metrics.Metric):    # TODO: Переделеать
    def __init__(self, num_classes, name='macro_f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precisions = [
            tf.keras.metrics.Precision(class_id=i) 
            for i in range(num_classes)
        ]
        self.recalls = [
            tf.keras.metrics.Recall(class_id=i)
            for i in range(num_classes)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Обновляем precision и recall для каждого класса
        for p, r in zip(self.precisions, self.recalls):
            p.update_state(y_true, y_pred, sample_weight)
            r.update_state(y_true, y_pred, sample_weight)

    def result(self):
        # Вычисляем F1 для каждого класса и усредняем
        f1_scores = []
        for p, r in zip(self.precisions, self.recalls):
            precision = p.result()
            recall = r.result()
            f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            f1_scores.append(f1)
        return tf.reduce_mean(f1_scores)  # Макро-усреднение

    def reset_state(self):
        # Сбрасываем состояния всех метрик
        for p, r in zip(self.precisions, self.recalls):
            p.reset_state()
            r.reset_state()