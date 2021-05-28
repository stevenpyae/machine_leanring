# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:00:19 2021
Ensure that you are at the correct directory to load the data
@author: Corbi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values     #Taking everything until the last column of the data
y = dataset.iloc[:, -1].values      #Taking ONLY the last column of the data 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)
print(y_train)

# Training the Simple Linear Regression Model on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # Methods of regression class, fit trains the linear regression model

# Predicting the Test set Results
y_pred = regressor.predict(X_test) # number of experience for the X_test 
print(y_test)
print(y_pred)
print(regressor.coef_)
print(regressor.intercept_)
print(regressor.predict([[0]]))

# Visualisation the Training Set Results
plt.scatter(X_train, y_train, color = 'red')  # put the red points in accordance to the real salaries 
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # predicted salary of the training set to plot the regression line
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test Set Results
plt.scatter(X_test, y_test, color = 'red')  # put the red points in accordance to the real salaries 
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Regression line remain the same. 
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'red')  # put the red points in accordance to the real salaries 
plt.plot(X_test, regressor.predict(X_test), color = 'green') # Regression line remain the same. 
plt.title('Salary vs Experience (X_Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()