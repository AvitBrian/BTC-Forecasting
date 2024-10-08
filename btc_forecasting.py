# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_qAuIbDVKYVFUxthDXf-GxjEUCPUYqN2
"""

# prerequisites
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/BTC-USD.csv')

# Convert 'Date' to datetime format
df_dates = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

plt.figure(figsize=(12,6))
plt.plot(df.index,df["Close"], color='red', label='Predicted BTC Prices')
plt.title('BTC Prices over time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

df.describe()

# Prepare the data
df = df.drop(labels={"Open", "High", "Low", "Adj Close", "Volume"}, axis=1)
df = df.rename(columns={'Close': 'closing_price'})
df.head()

# check for missing values
print(df.isnull().sum())

# Scale the data
scaler = MinMaxScaler()
df['closing_price'] = scaler.fit_transform(df[['closing_price']])

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)

print("training set:", train_data.describe(),  "\ntesting set:", test_data.describe())

import numpy as np

# Time series window
def create_dataset(data, window_size, shuffle=False):
    x, y = [], []

    for i in range(window_size, len(data)):
        x.append(data.iloc[i - window_size:i].values)
        y.append(data.iloc[i].values)

    x, y = np.array(x), np.array(y)

    if shuffle:
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x, y = x[indices], y[indices]

    return x, y

# Usage
window_size = 50
x_train, y_train = create_dataset(train_data, window_size, shuffle=True)
x_test, y_test = create_dataset(test_data, window_size, shuffle=False)

# Reshape the data for the LSTM model
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# model checkpoint
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# Build the LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(window_size, 1)),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val), callbacks=[early_stopping, model_checkpoint])
predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(real_prices, predicted_prices)

# Print metrics
print(f'MAE: {mae}, RMSE: {rmse}, R-squared: {r2}')

# Model summary
model.summary()
# Make predictions

predictions = model.predict(x_test)

# Inverse the scaling to get the real prices
predicted_prices = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


plt.figure(figsize=(12,6))
plt.plot(real_prices, color='blue', label='Actual BTC Prices')
plt.plot(predicted_prices, color='red', label='Predicted BTC Prices')
plt.title('BTC Price Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

