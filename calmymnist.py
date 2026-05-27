import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import utils
from scipy.optimize import minimize


class MnistNN:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(512, input_shape=(784,)))
        self.model.add(Activation('relu'))
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(10))
        self.model.add(Activation('linear'))  # No softmax here for logits
        
    def run(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784).astype('float32') / 255
        X_test = X_test.reshape(10000, 784).astype('float32') / 255
        Y_train = utils.to_categorical(y_train, 10)
        Y_test = utils.to_categorical(y_test, 10)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train, Y_train, batch_size=128, epochs=3, verbose=1)
        score = self.model.evaluate(X_test, Y_test)
        logits = self.model.predict(X_test)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # Save logits and true labels
        self.logits = logits
        self.y_test = y_test

    def temperature_scaling(self, logits, T):
        """Apply temperature scaling to logits"""
        exp_logits = np.exp(logits / T)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def nll_with_temperature(self, T, logits, labels):
        """Negative Log-Likelihood (NLL) with temperature scaling"""
        probs = self.temperature_scaling(logits, T)
        nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
        return nll

    def calibrate(self):
        """Find optimal temperature by minimizing NLL"""
        initial_temperature = 1.0
        res = minimize(self.nll_with_temperature, initial_temperature, args=(self.logits, self.y_test), bounds=[(0.1, 10)])
        optimal_temperature = res.x[0]
        print(f"Optimal temperature: {optimal_temperature}")
        return optimal_temperature

    def evaluate_calibrated_model(self, T):
        """Evaluate model with calibrated temperature"""
        calibrated_probs = self.temperature_scaling(self.logits, T)
        predicted_labels = np.argmax(calibrated_probs, axis=1)
        accuracy = np.mean(predicted_labels == self.y_test)
        print(f"Calibrated accuracy: {accuracy}")
        return calibrated_probs


a = MnistNN()
a.run()

# Perform calibration
optimal_T = a.calibrate()

# Evaluate calibrated model
calibrated_probs = a.evaluate_calibrated_model(optimal_T)
