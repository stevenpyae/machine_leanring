# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:59:46 2021
For X square graphs 
@author: Corbi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values     #Taking everything until the last column of the data
y = dataset.iloc[:, -1].values      #Taking ONLY the last column of the data 

#Training the Linear Regression Model on the Whole Set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training Polynomial Regression Model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y) # Poly Nomial Regression Model
print(X_poly)

# Visualising the Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()
# red dots are real salaries
# Blue line is the simple linear regression line based on the linear regression model

#Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue') # Transformed matrix of features sucha s polyreg.fit_tramsofrm(X)
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

#Visualising the polynomial regression results( Higher resolution and Smoother Curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') # Transformed matrix of features sucha s polyreg.fit_tramsofrm(X)
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

#Predicting new results with Linear Regression
lin_reg_pred = lin_reg.predict([[6.5]]) #predicting should be in 2Dimensions
print("Linear Regression Predict {}".format(lin_reg_pred))
#Predicting new Results with Polynomial Regression
poly_reg_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) 
print("Polynomial Regression Predict {}".format(poly_reg_pred))
# can't input that single position level, exactly in put the features 6.5, 6.5 square, 6.5 3times, 6.5 4times

# and put in 2D arrary
