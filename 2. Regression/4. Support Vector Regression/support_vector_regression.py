# -*- coding: utf-8 -*-
"""
Created on Fri May 28 13:39:15 2021

@author: Corbi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values     #it takes all the columns but Labels are useless so you just take the column of the 2nd index of index 1
y = dataset.iloc[:, -1].values      #Taking ONLY the last column of the data 

# Transform and Reshape into a 2D array. Horizontal List to Vertical Array
#because fit_transform for Feature Scaling needs 2D array
y = y.reshape(len(y), 1) #(no of row, no of columns)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
#DO we have to apply feature scaling to the dependant variable (Salary)? Yes 
# We don't want the smaller values to be neglected by the SVR Model
# Don't use Feature scaling for the following Situations
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)



sc_y = StandardScaler()
y = sc_y.fit_transform(y)

# Training the SVR model on the whole data set 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')   #Kernels - Linear or Non-Linear Gaussian RBF Kernel, 
regressor.fit(X, y)

# Predicting new result 
# Because X was scaled by sc_x and Y was scaled by sc_y, need to be on the same scale. Apply the scaling to the particular value
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
#the answer will return the value on the scale that was applied to Y. Reverse the scaling of the Y. Need to do Inverse Transform

# Visualising SVR 
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue') # Need to take note of the scaling
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

# Visualising SVR (Smoother)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue') # Transformed matrix of features sucha s polyreg.fit_tramsofrm(X)
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()