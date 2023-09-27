#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, PReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

class_num = 5

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='PReLU', input_shape=(224, 224, 3)))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(32, (3, 3), activation='PReLU'))
model.add(Conv2D(64, (3, 3), activation='PReLU'))
model.add(MaxPool2D(2, 2))

model.add(Conv2D(32, (3, 3), activation='PReLU'))
model.add(Conv2D(64, (3, 3), activation='PReLU'))
model.add(MaxPool2D(2, 2))

model.add(Flatten())
model.add(Dense(1024, activation='PReLU'))
model.add(Dropout(0.5))

model.add(Dense(class_num, activation='softmax'))

model.summary()

