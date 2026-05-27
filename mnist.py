# cancer: we give private information to predict cancer risk
# How can I trust the output regarding to privacy property(of the input, and the training dataset) ? 

# house pricing: 
# How can I trust the output regarding to accuracy property

# Bias dataset: 
# How can I trust the output regarding to bias property

# MNIST Attacks: 
# How can I trust the output regarding to robustness property

# https://colab.research.google.com/github/AviatorMoser/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb#scrollTo=z4dzeoUFca3O
import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from keras.datasets import mnist     # MNIST dataset is included in Keras
from keras.models import Sequential  # Model type to be used

from keras.layers import Dense, Dropout, Activation # Types of layers to be used in our model
from keras import utils                         # NumPy related tools
import pandas as pd 


class MnistNN:
    def __init__(self):
        # The Sequential model is a linear stack of layers and is very common.

        self.model = Sequential()

        # The first hidden layer is a set of 512 nodes (artificial neurons).
        # Each node will receive an element from each input vector and apply some weight and bias to it.

        self.model.add(Dense(512, input_shape=(784,))) #(784,) is not a typo -- that represents a 784 length vector!

        # An "activation" is a non-linear function applied to the output of the layer above.
        # It checks the new value of the node, and decides whether that artifical neuron has fired.
        # The Rectified Linear Unit (ReLU) converts all negative inputs to nodes in the next layer to be zero.
        # Those inputs are then not considered to be fired.
        # Positive values of a node are unchanged.

        self.model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
    
    def run(self):
        # The MNIST data is split between 60,000 28 x 28 pixel training images and 10,000 28 x 28 pixel images
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        print("X_train shape", X_train.shape)
        print("y_train shape", y_train.shape)
        print("X_test shape", X_test.shape)
        print("y_test shape", y_test.shape)

        X_train = X_train.reshape(60000, 784) # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
        X_test = X_test.reshape(10000, 784)   # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.

        X_train = X_train.astype('float32')   # change integers to 32-bit floating point numbers
        X_test = X_test.astype('float32')

        X_train /= 255                        # normalize each value for each pixel for the entire vector for each input
        X_test /= 255

        print("Training matrix shape", X_train.shape)
        print("Testing matrix shape", X_test.shape)


        nb_classes = 10 # number of unique digits

        Y_train = utils.to_categorical(y_train, nb_classes)
        Y_test = utils.to_categorical(y_test, nb_classes)


        # Let's use the Adam optimizer for learning
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train, Y_train,
                batch_size=128, epochs=3,
                verbose=1)
        score = self.model.evaluate(X_test, Y_test)
        pred = self.model.predict(X_test)
        lab = y_test
        df = pd.DataFrame(pred, columns=[f'Class_{i}_Probability' for i in range(pred.shape[1])])
        df['True Label'] = lab
        csv_path = './mnist_pred.csv'
        df.to_csv(csv_path, index=False)

        print('Test score:', score[0])
        print('Test accuracy:', score[1])
    def get_shape(self):


        res = [] #List pf [W,b]

        for layer in self.model.layers:
            if(type(layer) != Activation):
                res.append([np.shape(layer.get_weights()[0]), np.shape(layer.get_weights()[1])])
        return res

a = MnistNN()
a.run()
