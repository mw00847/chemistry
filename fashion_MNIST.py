import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from tensorflow import keras
import os

data=tf.keras.datasets.fashion_mnist

(training_images,training_labels),(test_images,test_labels)=data.load_data()

training_images =training_images/255
test_images=test_images/255

model=tf.keras.Sequential(([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
]))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images,training_labels,epochs=50)