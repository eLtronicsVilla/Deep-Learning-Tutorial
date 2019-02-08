# Artificial Neural Network 

# Evaluating the ANN - with cross validation score

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN

# We introduce Bias variance tradeoff
# you will get variant accuracy of the model when you train several times.
# Before we have taken the data as training set and teat set and we train the model on trian set and test the model on test dataset
# This is not the best one  we got the variance problem when we get the accuracy on test set , when you run the model again and you test on testset.we can get a very different accuracy.Judging the model performace on only one test set is not super releavant to evaluate the model performace . SO this techinique is called K-fold Cross validation because that will fix this variant problem.This will fix it the training set into 10 fold and we will train our model on 9 fold and test on last remaining fold.we can train the model on 10 combination of training  and test set.Take a average of accuracy of 10 evaluation and also compute the standard deviation, you have to look at the variants.Eventually our analysis will be much more relevance,besides we know these 4 catagories.
#1. High Bias Low variance (small accuracy on low variance)
#2.High Bias High Variance ( Low accuracy and high variance)
#3.Low Bias Low Variance (good accuracy small variance)
#4.Low Bias High Variance (Large accuracy and high variance

# we will build this using keras and scikit learn.

# import kerasClassifier wrapper
from keras.wrappers.scikit_learn import KerasClassifier

# impoet cross validation score for K- fold cross validation
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense

# This functionwill return the classifier from your network.Just build the architecture of your ANN
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# This classifier is not build based on whole x_train and y_train dataset.It build on cross validation for 10 fold, by each time it's measuring the model performance for one test fold. 

# create the clasifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# create class validation : estimator is classifier , X is data to fit, Y is target variable and cv is the number of fold of train and test set for applying k-fold cross validation.n_job to apply to use number of cpu to get run faster.
# we will see different training on different train fold run at the same time.
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
# take mean of the all 10 cross validation accuracies
mean = accuracies.mean()
variance = accuracies.std()
