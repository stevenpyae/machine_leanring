# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:34:32 2021

@author: Corbi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
# retrieve information from the Indexes
x = dataset.iloc[:, :-1].values      #iloc stands for locate indexes [all rows, columns without last column]
y = dataset.iloc[:, -1].values       # collect the indexes of the last column  

# replace missing value with the average of the remnaining data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
# replace the missing values
imputer.fit(x[:, 1:3])          # selection [all rows, only 2 columns]
x[:, 1:3] = imputer.transform(x[:, 1:3])    # this will return the modified matrix

"""ENCODING CATEGORIAL DATA"""
# Encoding the Independent Variable 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') #index [0], keep columns 
x = np.array(ct.fit_transform(x))     #for the output into a numpy array

# Encoding the Dependent Variable 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) 

# Splitting Data Set into Training set and Test Set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1) # Fixing the Seed here

# Feature Scaling to avoid dominating features. Standardisation and Normalisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[: , 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
