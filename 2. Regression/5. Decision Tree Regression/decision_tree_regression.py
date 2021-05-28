# -*- coding: utf-8 -*-
"""
Created on Fri May 28 13:56:53 2021
But not really adapted in this Position Salaries
@author: Corbi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values     #Taking everything until the last column of the data
y = dataset.iloc[:, -1].values   

#Training the Decision Tree Regression model on the whole dataset 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0) #need to fix the seed for the random
regressor.fit(X, y)

#Predict a new result
print(regressor.predict([[6.5]]))

# Visualisation step plot
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') # Transformed matrix of features sucha s polyreg.fit_tramsofrm(X)
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()

#Visualisation (smoother graph)
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue') # Transformed matrix of features sucha s polyreg.fit_tramsofrm(X)
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salaries')
plt.show()