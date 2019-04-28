########################################################################################################
##05/01/2019                                                      Alexander Alava Chonchol - U35432961##
##CRPAttackModelFinal.py                                                    Evelyn Almeyda - U00000000##
##                                                                      Anibal Jose Garcia - U00000000##
##                                                                                                    ##
##                 This file holds a machine learning implementation for a CRP Attack                 ##
########################################################################################################
from __future__ import absolute_import, division, print_function

# Importing TensorFlow and tf.keras #
import tensorflow as tf
from tensorflow import keras

# Importing helper libraries #
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Printing version of tensorflow #
print(tf.__version__)

# Reading data from the provided file and creating a dataset object #
data = pd.read_csv("CRPSets.csv", header=None)

# Randomly selecting the desired percentage of data used for training, example provided for 10,000 samples #
sampled_data = data.sample(frac=0.83333333)
# sampled_data = data.sample(frac=0.5) | Corresponding fraction when using 6,000 samples #
# sampled_data = data.sample(frac=0.16666667) | Corresponding fraction when using 2,000 samples #

# Retrieving the training data from the sampled data #
train_data = sampled_data.drop(sampled_data.columns[[64]], axis=1).values

# Retrieving the training responses from the sampled data #
train_responses = sampled_data[sampled_data.columns[[64]]].values

# Retrieving all the data not sampled for training to be used for testing #
nonsampled_data = data.drop(sampled_data.index)

# Retrieving the testing data from the nonsampled data #
test_data = nonsampled_data.drop(sampled_data.columns[[64]], axis=1).values

# Retrieving the testing responses from the nonsampled data #
test_responses = nonsampled_data[sampled_data.columns[[64]]].values

# Building the model specifying number of layers, types of layers, number of neurons,
# dropout rates and activation functions as required #
model = keras.Sequential([
    keras.layers.Dense(32, activation=tf.nn.softmax, input_shape=(64,)), # Input shape corresponds to number of bits #
    keras.layers.Dropout(0.1), # Dropping 10% of the current data to avoid overfitting #
    keras.layers.Dense(64, activation=tf.nn.sigmoid),
    keras.layers.Dropout(0.2), # Dropping 20% of the current data to avoid overfitting #
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dropout(0.1), # Dropping 10% of the current data to avoid overfitting #
    keras.layers.Dense(1)
])

# Describing model optimizer, loss function and desired metric in order to compile #
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fitting the training data and responses into the model and specifying number of epochs and batch size #
model.fit(train_data, train_responses, epochs=110, shuffle=False, batch_size=55)

# Evaluating the model with the testing data and responses, and retrieving accuracy and loss #
test_loss, test_acc = model.evaluate(test_data, test_responses)

# Printing average accuracy of the test done with the testing data and responses #
print('Test accuracy:', test_acc)
