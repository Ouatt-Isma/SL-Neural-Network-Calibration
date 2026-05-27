import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, brier_score_loss
from scipy.optimize import minimize

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Build a simple neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy before calibration: {test_acc}")

# Get the logits from the model
logits = model.predict(x_test)

# Softmax function with temperature scaling
def softmax_with_temperature(logits, T):
    exp_logits = np.exp(logits / T)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Loss function for calibration (negative log-likelihood)
def nll_with_temperature(T, logits, labels):
    probs = softmax_with_temperature(logits, T)
    nll = -np.mean(np.log(probs[np.arange(len(labels)), labels]))
    return nll

# Optimize temperature
labels = y_test
initial_temperature = 1.0  # Starting with no scaling (T=1)
res = minimize(nll_with_temperature, initial_temperature, args=(logits, labels), bounds=[(0.1, 10)])
optimal_temperature = res.x[0]

print(f"Optimal temperature: {optimal_temperature}")

# Apply the optimal temperature to logits
calibrated_probs = softmax_with_temperature(logits, optimal_temperature)

# Evaluation Function: Calibration Metrics and Accuracy
def evaluate_model_calibration(logits, calibrated_probs, labels):
    """
    Evaluate calibration of the model before and after applying temperature scaling.
    Metrics: Accuracy, Expected Calibration Error (ECE), and Brier Score
    """
    # Predictions before calibration
    probs_before = tf.nn.softmax(logits, axis=1).numpy()
    predicted_labels_before = np.argmax(probs_before, axis=1)

    # Predictions after calibration
    predicted_labels_after = np.argmax(calibrated_probs, axis=1)

    # Accuracy before and after calibration
    acc_before = accuracy_score(labels, predicted_labels_before)
    acc_after = accuracy_score(labels, predicted_labels_after)

    # Brier score (lower is better)
    brier_before = brier_score_loss(labels, probs_before[np.arange(len(labels)), predicted_labels_before])
    brier_after = brier_score_loss(labels, calibrated_probs[np.arange(len(labels)), predicted_labels_after])

    # Expected Calibration Error (ECE) - Simplified
    def expected_calibration_error(probs, labels, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs >= bin_lower) & (probs < bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(labels[in_bin] == np.argmax(probs[in_bin], axis=1))
                avg_confidence_in_bin = np.mean(np.max(probs[in_bin], axis=1))
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    ece_before = expected_calibration_error(probs_before, labels)
    ece_after = expected_calibration_error(calibrated_probs, labels)

    # Print results
    print(f"Accuracy before calibration: {acc_before}")
    print(f"Accuracy after calibration: {acc_after}")
    print(f"Brier score before calibration: {brier_before}")
    print(f"Brier score after calibration: {brier_after}")
    print(f"ECE before calibration: {ece_before}")
    print(f"ECE after calibration: {ece_after}")

# Evaluate the model calibration
evaluate_model_calibration(logits, calibrated_probs, y_test)

# Plot the confidence histograms (before and after calibration)
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.hist(np.max(tf.nn.softmax(logits, axis=1).numpy(), axis=1), bins=50, alpha=0.7, label='Before Calibration')
plt.title('Confidence Before Calibration')
plt.subplot(1, 2, 2)
plt.hist(np.max(calibrated_probs, axis=1), bins=50, alpha=0.7, label='After Calibration')
plt.title('Confidence After Calibration')
plt.show()
