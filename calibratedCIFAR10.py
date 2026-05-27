import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
import pandas as pd 

# Load MNIST data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

def one_run(epoch = 1):
# Build a simple neural network model
    model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)  # CIFAR-10 has 10 classes
])

    # Compile and train the model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epoch, batch_size=32, validation_split=0.1)

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc}")

    # Get the logits from the model
    logits = model.predict(x_test)

    
    # Optimize temperature
    labels = y_test
    initial_temperature = 1.0  # Starting with no scaling (T=1)
    res = minimize(nll_with_temperature, initial_temperature, args=(logits, labels), bounds=[(0.1, 10)])
    optimal_temperature = res.x[0]

    print(f"Optimal temperature: {optimal_temperature}")

    # Apply the optimal temperature to logits
    calibrated_probs = softmax_with_temperature(logits, optimal_temperature)

  
    # Get the logits from the model
    logits = model.predict(x_test)


    # Optimize temperature
    labels = y_test
    initial_temperature = 1.0  # Starting with no scaling (T=1)
    res = minimize(nll_with_temperature, initial_temperature, args=(logits, labels), bounds=[(0.1, 10)])
    optimal_temperature = res.x[0]

    print(f"Optimal temperature: {optimal_temperature}")

    # Apply the optimal temperature to logits
    calibrated_probs = softmax_with_temperature(logits, optimal_temperature)


    # Step 1: Get logits and print before calibration
    logits_before_calibration = model(x_test)
    # print("Logits before calibration:")
    # print(logits_before_calibration[:5])

    # Step 2: Apply temperature scaling and print after calibration
    logits_after_calibration = apply_temperature_scaling(logits_before_calibration, optimal_temperature)
    # print("Logits after calibration:")
    # print(logits_after_calibration[:5])

    # Step 3: Softmax to get probabilities and print
    probs_before_calibration = tf.nn.softmax(logits_before_calibration, axis=1)
    probs_after_calibration = tf.nn.softmax(logits_after_calibration, axis=1)

    # print("Probabilities before calibration:")
    # print(probs_before_calibration[:5])

    # print("Probabilities after calibration:")
    # print(probs_after_calibration[:5])

    lab = y_test
    
    df = pd.DataFrame(probs_before_calibration, columns=[f'Class_{i}_Probability' for i in range(probs_before_calibration.shape[1])])
    df['True Label'] = lab
    csv_path = f'./MNIST_PRED/bef_{epoch}.csv'
    df.to_csv(csv_path, index=False)

    df = pd.DataFrame(probs_after_calibration, columns=[f'Class_{i}_Probability' for i in range(probs_after_calibration.shape[1])])
    df['True Label'] = lab
    csv_path = f'./MNIST_PRED/aft_{epoch}.csv'
    df.to_csv(csv_path, index=False)


# Softmax function with temperature scaling
def softmax_with_temperature(logits, T):
    exp_logits = np.exp(logits / T)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Loss function for calibration (negative log-likelihood)
def nll_with_temperature(T, logits, labels):
    probs = softmax_with_temperature(logits, T)
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    return nll



# Softmax function with temperature scaling
def softmax_with_temperature(logits, T):
    exp_logits = np.exp(logits / T)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Loss function for calibration (negative log-likelihood)
def nll_with_temperature(T, logits, labels):
    probs = softmax_with_temperature(logits, T)
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    return nll

def apply_temperature_scaling(logits, T):
    return logits / T


def main():
    for epoch in range(1, 10):
        one_run(epoch)
    for epoch in range (10, 101, 10):
        one_run(epoch)

main()