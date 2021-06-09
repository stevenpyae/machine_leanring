# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:55:15 2021
Artifical Neural Network
Bank Problem, Unusual Churn Rates
Which people are more reliable and which people are likely to exit 
@author: Corbi
"""

# Importing the Libraries
import numpy as np
import pandas as pd
import tensorflow as tf 

print(tf.__version__)
'''Part 1 - Data Pre Processing
'''
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# retrieve information from the Indexes
# We don't need the following because they Don't help 
# Row Number
# CustomerID
# Surame

X = dataset.iloc[:, 3:-1].values      #iloc stands for locate indexes[all the rows, columns 3 till the second last column]
y = dataset.iloc[:, -1].values       # collect the indexes of the last column  

# Encoding categorical data
## Label Ecoding the 'Gender Column'
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2]) # Thrid Column of X which has index 2
# Take all the rows and column 2

## One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough') #index [0], keep columns 
X = np.array(ct.fit_transform(X))     #for the output into a numpy array
# One hot encoding, dummy variables are moved to the very first columns. 
# 1, 0, 0 is France
# 0, 0, 1 is Spain
# 0, 1, 0 is Germany

# Splitting the dataset into the Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Fixing the Seed here

# Feature Scaling - its FUNDAMENTAL for Deep Learning
# Features scale all the rows and columns 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''Part 2 - Building the ANN
'''
# Initializing the ANN
# Keras is integrated into Tensorflow, Sequencial class, as a sequence of layers
# ANN is a sequence of Layers, Input Layers, Hidden Layers, Output Layers
# Boltsman Machines - computation Graphs
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# Dense Class in Tensorflow or PyTorch\
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) # add hidden layer or conv 2d layer dense layer or fully connected layer. 
# the fully connected layer that we create will be object of the Dense Class
# Dense takes in Units which are indicating how many Hidden Neurons we want to have
# FAQ - How many Hidden Neurons we want to have? it is based on Trial and Error.
# Activation of Hidden layer must be Rectifier Function

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
#output layer need only one neuron for Binary
# if got Classificatino Output Layer, we need 3 units that are One Hot Encoded. 
# Activation of Output Layer need Sigmoid Activation Function to calculate the Probablity.
# Activation for Categorical Prediction -> Softmax 

'''Part 3 - Training the ANN
'''
# Compiling the ANN with the Optimizer and the Loss Function 
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# 3 parameters
# Optimizer -> Adam Optimizer which can perform stochastic_gradient_descent
# Loss -> way to compute the difference between predicted and expected, for binary, need 'binary_crossentropy'
# Loss -> 'categorical_crossentropy' for Categorical Prediction
# Metrics -> list of metrics

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
# batch_size determine number of perdictions in the batch you want to compare for the LOSS FUNCTION
# Default is 32 
# We are doing Batch Learning
# epochs -> Network needs to be Trained over a certain amount of epochs

'''Part 4 - Making the predictions and evaluating the model
'''

# Predicting the result of a single observation
''' Use our ANN model to predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?'''
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5) # ANY INPUT of the predict method, must be a 2D array
# Geography and Male need to enter the value of Dummy Variable 
# Remember the predicted method should be called on the observation which went through same Feature scaling 
# REMEMBER NOT TO fit again because your model will be affected because the new variable will change the mean and standard deviation
# CHECK If scaling is applied. Do all PREPROCESSING STEPS
# The result 0.03 states that the person is not likely to exit the company. 

# Predicitng the Test set results
y_pred = ann.predict(X_test)
# Predicted Binary Outcome 
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 1527 correct of class 0, 68 incorrect of class 1 
# 205 correct of class 1, 200 incorrect of class 0 
print(accuracy_score(y_test, y_pred))
# Accuracy of 0.8635