import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import pandas as pd 

# from tensorflow.keras.datasets import mnist
# Load MNIST data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize


from tensorflow.keras.datasets import cifar10
# # Load CIFAR10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize



def one_run(epoch_t = 1):

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
   


    # Compile the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    FOLDER_NAME = "bin"

    # model = models.Sequential([
    #     layers.Flatten(input_shape=(28, 28)),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(10)
    # ])

    # # Compile and train the model
    # model.compile(optimizer='adam',
    #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #             metrics=['accuracy'])
    
    # FOLDER_NAME = "MNIST_GOOD"

    to_keep = [1] + list(range(10, 101, 10))
    y_test_2 = y_test.flatten().astype(np.int32)
    for _e in range(1, epoch_t+1):
        print(f"--------------------------------{_e}-----------------------")
        # Train the model
        model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.1)
        if(_e in to_keep and False):
            # Create temperature scaling model
            temperature_layer = TemperatureScaling()
            calibration_model = models.Sequential([model, temperature_layer])


            # Optimize temperature with validation data
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
            epochs = 100
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    calibrated_logits = calibration_model(x_test)
                    loss_value = nll_with_temperature(calibrated_logits, y_test_2)
                grads = tape.gradient(loss_value, calibration_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, calibration_model.trainable_variables))
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {loss_value.numpy()}, Temperature: {temperature_layer.temperature.numpy()}')

            # Evaluate calibration on test set
            test_logits = calibration_model(x_test)
            test_probs = tf.nn.softmax(test_logits)
            test_accuracy = np.mean(np.argmax(test_probs, axis=1) == y_test.flatten())
            print(f'Test Accuracy after Calibration: {test_accuracy}')
            test_logits_bef = model(x_test)
            test_probs_bef = tf.nn.softmax(test_logits_bef)
            lab = y_test
    
            df = pd.DataFrame(test_probs_bef, columns=[f'Class_{i}_Probability' for i in range(test_probs_bef.shape[1])])
            df['True Label'] = lab
            csv_path = f'./{FOLDER_NAME}/bef_{_e}.csv'
            df.to_csv(csv_path, index=False)

            df = pd.DataFrame(test_probs, columns=[f'Class_{i}_Probability' for i in range(test_probs.shape[1])])
            df['True Label'] = lab
            csv_path = f'./{FOLDER_NAME}/aft_{_e}.csv'
            df.to_csv(csv_path, index=False)





# Function for temperature scaling
class TemperatureScaling(tf.keras.layers.Layer):
    def __init__(self, init_temp=1.0, **kwargs):
        super(TemperatureScaling, self).__init__(**kwargs)
        self.temperature = tf.Variable(init_temp, trainable=True, dtype=tf.float32)

    def call(self, logits):
        return logits / self.temperature



# Loss for temperature scaling
def nll_with_temperature(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))




def main():
        one_run(100)

main()