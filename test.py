import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# Normalize data
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

# Create a simple CNN model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

model = create_model()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val))

# Predict logits on validation set
logits = model.predict(x_val)
y_val = y_val.flatten().astype(np.int32)
y_test = y_test.flatten().astype(np.int32)
# Function for temperature scaling
class TemperatureScaling(tf.keras.layers.Layer):
    def __init__(self, init_temp=1.0, **kwargs):
        super(TemperatureScaling, self).__init__(**kwargs)
        self.temperature = tf.Variable(init_temp, trainable=True, dtype=tf.float32)

    def call(self, logits):
        return logits / self.temperature

# Create temperature scaling model
temperature_layer = TemperatureScaling()
calibration_model = models.Sequential([model, temperature_layer])

# Loss for temperature scaling
def nll_with_temperature(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# Optimize temperature with validation data
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
epochs = 100
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        calibrated_logits = calibration_model(x_val)
        loss_value = nll_with_temperature(calibrated_logits, y_val)
    grads = tape.gradient(loss_value, calibration_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, calibration_model.trainable_variables))
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss_value.numpy()}, Temperature: {temperature_layer.temperature.numpy()}')

# Evaluate calibration on test set
test_logits = calibration_model(x_test)
test_probs = tf.nn.softmax(test_logits)
test_accuracy = np.mean(np.argmax(test_probs, axis=1) == y_test.flatten())
print(f'Test Accuracy after Calibration: {test_accuracy}')
