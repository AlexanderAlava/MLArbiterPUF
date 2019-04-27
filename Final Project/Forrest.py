import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.utils.vis_utils import plot_model

training_data_df = pd.read_csv("CRPSets_11k_training.csv", header=None)

X = training_data_df.drop(training_data_df.columns[[64]], axis=1).values;
Y = training_data_df[training_data_df.columns[64]].values;

model = Sequential()
model.add(Dense(10, input_dim=64, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))



#np.random.shuffel
#sigmoid



#adam = optimizers.Adam(lr = 0.01)
model.compile(loss="binary_crossentropy", optimizer="adam")
#mean_squared_error
#binary_corssentropy

# Train the model

model.fit(
    X,
    Y,
    batch_size=100,
    epochs=200,
    verbose=2,
    shuffle=True

)

print(model.summary())

#plot_model(model, to_file='MODEL.png')
# Load the separate test data set
test_data_df = pd.read_csv("CRPSets_11k_testing.csv", header=None)


X_test = test_data_df.drop(training_data_df.columns[[64]], axis=1).values
Y_test = test_data_df[training_data_df.columns[64]].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))



# Load the data we make to use to make a prediction
X = pd.read_csv("dataToPredict.csv", header=None).values


# Load the data with correct result to compare to prediction
result = pd.read_csv("expectedResult.csv", header=None).values

#np.array(result).tolist()

counter = 0
predictionResult = []

# Make a prediction with the neural network
prediction = model.predict(X)




# Normalizing Prediction to Binary value
# predict result = predict.round

for i in prediction:
    if i >= 0.5:
        predictionResult.append(1)
    else:
        predictionResult.append(0)

#np.array(predictionResult).tolist()

# Checking Prediction
for i in predictionResult:
    print("Real result:{}  Prediction Result:{} ".format(result[i], predictionResult[i]))

counter = len([i for i, j in zip(predictionResult, result) if i == j])

percent = (counter / len(result)) * 100

print(prediction)
print(predictionResult)


#res = list(result.T)

#res2 = list(np.reshape(res, (50)))

#print("result start")
#print(res2)

#print(confusion_matrix(res2, predictionResult))

print("counter is equal to {}".format(counter))

print("Accuracy: {}%".format(percent))


