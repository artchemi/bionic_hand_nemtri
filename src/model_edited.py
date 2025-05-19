import tensorflow as tf

class NeuralNetworkModel:
    def __init__(self, input_shape, num_classes):
        self.model = self.build_model(input_shape, num_classes)
        self.compile_model()

    def build_model(self, input_shape, num_classes):
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model

    def compile_model(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2):
        history = self.model.fit(train_data, train_labels,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=validation_split)
        return history

    def evaluate(self, test_data, test_labels):
        results = self.model.evaluate(test_data, test_labels)
        print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")
        return results

    def fine_tune(self, new_data, new_labels, epochs=5, batch_size=32):
        # Можно изменить learning rate перед дообучением при необходимости:
        tf.keras.backend.set_value(self.model.optimizer.lr, 1e-5)
        history = self.model.fit(new_data, new_labels,
                                 epochs=epochs,
                                 batch_size=batch_size)
        return history

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
