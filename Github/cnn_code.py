# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 21:08:50 2019

@author: Veronica
"""

import cv2
import os
import numpy as np

def load_images_from_folder(folder,index):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        if img is not None:
            images.append([np.array(img), np.array(index)])
    return images


images = []
for index,folder in enumerate(os.listdir("Images/dataset")):
    images.extend(load_images_from_folder("Images/dataset/"+folder,index))

X = np.array([i[0] for i in images]).reshape(-1, 28, 28, 1)
Y = [i[1] for i in images]

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


X_train= X.reshape(X.shape[0],28,28,1).astype('float32')

y_train = np_utils.to_categorical(Y)
num_classes = y_train.shape[1]

from keras import backend as K
K.set_image_dim_ordering('tf')

classifier = Sequential()
classifier.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=13,activation='softmax'))

classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=50, batch_size=50)

scores = classifier.evaluate(X_train, y_train, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))