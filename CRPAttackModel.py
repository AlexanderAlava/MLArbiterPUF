from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

print(tf.__version__)

data = pd.read_csv("CRPSets.csv", header=None)
sampled_data = data.sample(frac=0.66666667, random_state=1)
train_data = sampled_data.drop(sampled_data.columns[[64]], axis=1).values
train_responses = sampled_data[sampled_data.columns[[64]]].values

nonsampled_data = data.drop(sampled_data.index)
test_data = nonsampled_data.drop(sampled_data.columns[[64]], axis=1).values
test_responses = nonsampled_data[sampled_data.columns[[64]]].values

print(test_responses)

# This is how the model is built
model = keras.Sequential([
    keras.layers.Dense(32, activation=tf.nn.softmax, input_shape=(64,)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation=tf.nn.sigmoid),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation=tf.nn.softmax),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1)
])

# This is where parameters of the model are described, we might not need any changes here
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Here the data is added to the model and the number of learning rounds is described
model.fit(train_data, train_responses, epochs=110, shuffle=False, batch_size=55)

test_loss, test_acc = model.evaluate(test_data, test_responses)

print('Test accuracy:', test_acc)
