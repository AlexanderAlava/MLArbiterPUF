import pandas as pd
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
    epochs=10,
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
X = pd.read_csv("dataToPredict.csv").values

print(X)

# Load the data with correct result to compare to prediction
result = pd.read_csv("expectedResult.csv").values
counter = 0
predictionResult = []
# Make a prediction with the neural network
prediction = model.predict(X)

#print(counter)

# Normalizing Predicition to Binary value
for i in prediction:
    if i > 0.50:
        predictionResult.append(1)
    else:
        predictionResult.append(0)


#for i in predictionResult:
  #  for j in result:
   #     if i == j:
    #        counter += 1

print(prediction)
print(predictionResult)

print(result)

print("counter is equal to {}".format(counter))

