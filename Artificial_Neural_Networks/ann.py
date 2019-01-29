# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np #for all numerical and computational operation
import matplotlib.pyplot as plt  # to plot the graph
import pandas as pd  # library to work on datasheet  

# Importing the dataset ie geography and gender
dataset = pd.read_csv('Churn_Modelling.csv')  #this is to read the csv file using pandas libraary
X = dataset.iloc[:, 3:13].values  # collecting the row value from range 1 to 12 
y = dataset.iloc[:, 13].values  # collecting the row value of 13

# Encoding categorical data of dependent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()  # encoding the variable  
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # label 0,1,2 for France,Spain,Germany
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # label 0 or 1 for male or female
onehotencoder = OneHotEncoder(categorical_features = [1]) # to create dummy variable to the country
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set 80% and Test set 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # feature scaling for training data
X_test = sc.transform(X_test)    # feature scaling for testing data

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential  # import the keras library for sequential model
from keras.layers import Dense   # import the keras library for dense layer

# Initialising the ANN
classifier = Sequential() #this classifier is to build neural network model

# Adding the input layer and the first hidden layer
# output_dim of layer is 6,kernel_initializer means weight initializer that will initialize uniformly approx to zero , activation function 
# is rectifier function and input dimention (independent variable) for this layer is 11.

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# output_dim (dependent variable) of this layer is 1 and activation function is sigmoid because it's binary classification
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# For Stochastic Gradient Descent , 'adam' is the optimizer,loss function is calculated logrithmic type.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# 100 epoch will run to pass all dataset through NN.
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# predict true or false output 
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
