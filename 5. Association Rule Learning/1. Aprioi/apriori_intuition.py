# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 10:52:35 2021
People who bought this, also bought that. Prediction on Amazon
Most of the Learning Models uses Apriori Models
@author: Corbi
"""

# Importing the libraries
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
# Note the confidence between each item 'chicken', 'light cream' 29% chance of buying  
# Putting the results well organised into a Pandas DataFrame (NOT ESSENTIAL)
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results] 
    #take full list and each of rule, Rule -> access Index 2 -> first element 0, -> First element 0 (itembase=frozenset) -> 0 ("Light cream") 
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    #take full list and each of rule, Rule -> access Index 2 -> first element 0, -> First element 0 (itembase=frozenset) -> 1 ("Chicken")
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
# Displaying the results non sorted
print(resultsinDataFrame)
# Displaying the results sorted by descending lifts 
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
