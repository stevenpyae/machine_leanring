# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:23:24 2021
Change the Working Directory
@author: Corbi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



#Encoding Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') # [3] need to take note of the Index
X = np.array(ct.fit_transform(X))     #for the output into a numpy array

# Splitting the Dataset into Training and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Do we have to avoid the dummy variable trap? the class will automatically do it for you.
# Do we have to deploy the best features that has highest P-value? NO. the class will automatically identify the best features for you.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test) #Feature of the test sets which is X rather than the profits which is dependent variables
np.set_printoptions(precision=2) # any numerical values to 2 decimal places

# Comparing the Predicted results vs Actual Results
# print(y_pred)
# print(y_test)
# print(y_pred.reshape(len(y_pred), 1))
print("Predicted vs Actual")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1)) 
# 2 vectors of the real profit and predicted profits reshapes the sequencial array into a array of 1 column

# Prediction 
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

#Getting Linear Equation with Values of the Coeff 
print(regressor.coef_)
print(regressor.intercept_)

# # Evaluating the Model Performance Rsquare score 
# from sklearn.metrics import r2_score
# r2_score(y_test, y_pred)