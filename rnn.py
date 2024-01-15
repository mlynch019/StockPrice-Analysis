import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from math import floor



data = pd.read_csv('./data/all.csv')
# data.head()
closing_price = data['Close']
dates = data['Date'] # track dates separately (indexes imply sequential order but keep dates to graph)
# normilization of data with MinMaxScaler object
scaler = MinMaxScaler(feature_range = (0,1))
scaled_close = scaler.fit_transform(closing_price.values.reshape(-1,1))

# for testing time step amounts. Daily stock data so time stamp 10-30 is good
time_steps = 20

# split data into sequences based on time stamp
# append 'sequence' data to list then convert to np array
x_ = [] 
y_ = []
for i in range(len(scaled_close) - time_steps):
    x_.append(scaled_close[i:i+time_steps,0])
    y_.append(scaled_close[i,0])
x = np.array(x_)
y = np.array(y_)
# print(X)
# train test split, latter 25% test data. Split is only for training data, testing data is separate
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=False)
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
# define sequential model, add Long Short Term Memory layers with dropout, then dense layer. Relu activation function
#input_shape=(X_train.shape[1], X_train.shape[2])
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1],1), return_sequences=True))
model.add(Dropout(.25)) # drop nodes to prevent overfitting
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(.25))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(.25))
model.add(Dense(units=1))

# update for adam, adagram, sgd. Keep loss function as mean squared error 'mse'
model.compile(optimizer='sgd', loss='mse')

# train
model.fit(X_train, y_train, epochs=50, batch_size=32)

# create test set and predict
# x_test_ = closing_price[floor(len(closing_price) * .75):]
# x_test = scaler.fit_transform(x_test_.values.reshape(-1,1))
# test = []
# for i in range(len(scaled_close) - time_steps):
#     test.append(x_test[i:i+time_steps,0])
# test_ = np.array(test)
# test_ = np.reshape(test, (test.shape[0], test.shape[1],1))
predicted = model.predict(X_test)

# # to obtain prices on original scale after inital normilization
predicted_normal_scale = scaler.inverse_transform(predicted.reshape(-1,1))
# y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
# predicted_normal_scale  = np.reshape(predicted_normal_scale , (predicted_normal_scale.shape[0], predicted_normal_scale .shape[1], 1))

final_loss = model.evaluate(X_test, y_test)
print(final_loss)

# obtaining last 'len(y_test)' indexes of date indexes
dates_ = dates[-len(y_test):]

plt.figure()
plt.plot(dates_, y_test, label='Actual Closing Price')
plt.plot(dates_, predicted_normal_scale, label='Predicted Closing Price')
plt.xticks(dates_[::25])
plt.title('Google Stock Price Prediction - Adam Optimizer')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()


# bar chart for comparing loss of different optimizers
# optimizers = ['Adam', 'Adagrad', 'SGD']
# accuracy = [0.00185933, 0.00434327, 0.00338674]

# plt.bar(optimizers, accuracy)
# plt.xlabel('Optimizers')
# plt.ylabel('Loss')
# plt.title('Loss (Mean Squared Error) of Various Optimizers for Recurrent Nueral Network')
# plt.show()

