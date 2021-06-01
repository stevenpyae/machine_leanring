# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:18:16 2021
Simplified version of Apriori Model
@author: Corbi
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
# We won't use scikit-learn. Apriori-Py model
dataset =  pd.read_csv('Market_Basket_Optimisation.csv', header = None) # no column names
transactions = []
# for loop to populate the transaction list
for i in range(0,7501):# loop over the rows
     transactions.append([str(dataset.values[i,j]) for j in range(0, 20)]) # loop over the columns
    #dataset [i,j] STR to ensure that the products are strings
# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
#min_support, Product to appear minimum amount of time 3x7 transaction in a day, 7501 transactions, 21 transaction per week. 3x7/7501
#min_confidence = 0.2 this is 
#min_lift = a good lift is at three onwards
#min_length COMPULSORY - minimum number of elements in your rule
#max_length COMPULSORY - 2
# Visualising the results=

# Displaying the first results coming directly from the output of the Apriori Function
results = list(rules)
print(results)
#Putting the results well organized into Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])
#Displaying the results sorted by descending supports
resultsinDataFrame.nlargest(n = 10, columns = 'Support')
