import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *

training_data_df = pd.read_csv("CRPSets_training.csv")

X = training_data_df.drop(training_data_df.columns[[64]], axis=1).values
Y = training_data_df[training_data_df.columns[64]].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=64, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='linear'))

model.compile(loss="mean_squared_error", optimizer="adam")


# Train the model

model.fit(
    X,
    Y,
    epochs=2000,
    verbose=2,
    shuffle=True

)


# Load the separate test data set
test_data_df = pd.read_csv("CRPSets_test.csv")


X_test = test_data_df.drop(training_data_df.columns[[64]], axis=1).values
Y_test = test_data_df[training_data_df.columns[64]].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))



# Load the data we make to use to make a prediction
X = pd.read_csv("dataToPredict.csv", header=None).values


# Load the data with correct result to compare to prediction
result = pd.read_csv("expectedResult.csv" , header=None).values

#np.array(result).tolist()

counter = 0
predictionResult = []

# Make a prediction with the neural network
prediction = model.predict(X)


# Normalizing Prediction to Binary value
for i in prediction:
    if i > 0.50:
        predictionResult.append(1)
    else:
        predictionResult.append(0)

#np.array(predictionResult).tolist()

# Checking Prediction
#for i in predictionResult:
    #print("Real result:{}  Prediction Result:{} ".format(result[i], predictionResult[i]))

counter = len([i for i, j in zip(predictionResult, result) if i == j])

percent = (counter / len(result)) * 100

print(prediction)
print(predictionResult)



print("counter is equal to {}".format(counter))

print("Accuracy: {}%".format(percent))