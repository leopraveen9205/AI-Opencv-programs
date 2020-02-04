# -*- coding: utf-8 -*-https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
#https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
"""
Created on Sat Nov 17 11:37:41 2018

@author: Admin
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.imshow(X_train[0])
plt.imshow(X_train[1])


import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from keras import backend as K
K.set_image_dim_ordering('th')

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

classifier = Sequential()
classifier.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))

classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=20, batch_size=5)

scores = classifier.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))


