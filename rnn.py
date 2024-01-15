import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

data = pd.read_csv('./data/all.csv')

# main element is closing price
features = data[['Open', 'Close', 'Volume']]
target = data['Close']
dates = data['Date'] # track dates separately (indexes imply sequential order but keep dates to graph)

# normilization of data with MinMaxScaler object
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

# for testing time step amounts. Daily stock data so time stamp 10-30 is good
time_steps = 20

# Separate data into sequences based on the time step (how much input data to be considered given time)
X_, y_ = [], []
for i in range(len(scaled_features) - time_steps):
    X_.append(scaled_features[i:(i + time_steps)])
    y_.append(scaled_target[i + time_steps])
X = np.array(X_) 
y = np.array(y_)

# train test split, latter 25% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# define sequential model, add Long Short Term Memory and dense layers. Relu activation function
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences = True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(.25))
model.add(LSTM(units=50))
model.add(Dropout(.25))
model.add(Dense(units=1))

# update for adam, adagram, sgd. Keep loss function as mean squared error
model.compile(optimizer='sgd', loss='mean_squared_error')

# train
model.fit(X_train, y_train, epochs=50, batch_size=32)

# predict
predicted = model.predict(X_test)

# to obtain prices on original scale after inital normilization
predicted_normal_scale = scaler.inverse_transform(predicted)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

final_loss = model.evaluate(X_test, y_test)
print(final_loss)

# obtaining last 'len(y_test)' indexes of date indexes
dates_ = dates[-len(y_test):]

plt.figure()
plt.plot(dates_, y_actual, label='Actual Closing Price')
plt.plot(dates_, predicted_normal_scale, label='Predicted Closing Price')
plt.xticks(dates_[::25])
plt.title('Google Stock Price Prediction - SGD Optimizer')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()


# # bar chart for comparing loss of different optimizers
# optimizers = ['Adam', 'Adagrad', 'SGD']
# accuracy = [0.00185933, 0.00434327, 0.00338674]

# plt.bar(optimizers, accuracy)
# plt.xlabel('Optimizers')
# plt.ylabel('Loss')
# plt.title('Loss (Mean Squared Error) of Various Optimizers for Recurrent Nueral Network')
# plt.show()
