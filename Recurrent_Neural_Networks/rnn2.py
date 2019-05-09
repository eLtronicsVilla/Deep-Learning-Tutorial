#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 23:37:17 2019

@author: brgupta
"""

#import the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
# use iloc to extract index of column
training_set = training_set.iloc[:,1:2].values

# feature scaling
# we have two choices : Normalization or Standarization

# Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# getting the input and the output
# Here x_train will be the input and y_train will be the output
# y_pred should be at the time t+1
# we will take all the stock prices except last one.

x_train = training_set[0:1257]
y_train = training_set[1:1258]

# reshape
x_train = np.reshape(x_train,(1257,1,1))

# Building the RNN
# Importing the keras library and package
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
    
# Initializing the RNN
regressor = Sequential()

# Adding the input layer and LSTM layer
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1))) # we have input shape of 1 feature
# 4 is number of memory units , activation function either can be hyperbolic tangent or the sigmoid activation function,input shape argument is number of feature.

# adding the output layer
regressor.add(Dense(units=1)) # chooose right value for the unit argument

# compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error') # rms prop optimiser is asking more memory than Adam optimizer
# mean squre error is the sum of square difference of your predicted stock prices and we get root mean sqare error to get state of the art.
# we will take square of the error and we sum this for ob
# for testing dataset we will get the state of the art by root mean square error
# In deep learning regression model , we will use mean square error
# It's sum of square differences between your predicted stock prices and real stock prices
 

# fitting the RNN to the training set
regressor.fit(x_train,y_train,batch_size=32,epochs=200)

# 3 : making the predictions and visualizing the result

test_set = pd.read_csv('Google_Stock_Price_Test.csv')

real_stock_price = test_set.iloc[:,1:2].values
# gettig the real stock price of 2017
# predicting the value of financial day t+1
inputs = real_stock_price
# we will use the same scale method on our input dataset
inputs = sc.transform(inputs)
inputs = np.reshape(inputs,(20,1,1)) # here 1 time stamp and 1 feature
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualising the result
plt.plot(real_stock_price,color='red',label='real google stock price')
plt.plot(predicted_stock_price,color='blue',label='predicted google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()

# Home work

# getting the real stock price of 2012-2016
real_stock_price_train = pd.read_csv('Google_Stock_Price_Train.csv')

real_stock_price_train = real_stock_price_train.iloc[:,1:2].values

# getting the predicted stock price of 2012-16
predicted_stock_price_train = regressor.predict(x_train) # here is scaled prediction , so unscale the prediction to get scaled predictions.
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# visualising the result
plt.plot(real_stock_price_train, color='red',label='Real google stock price')
plt.plot(predicted_stock_price_train, color='blue',label='Predicted google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()

# evaluate your RNN model
# Root r- means square on test
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))
 