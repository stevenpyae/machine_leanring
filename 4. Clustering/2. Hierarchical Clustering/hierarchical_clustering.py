# -*- coding: utf-8 -*-
"""
Created on Mon May 31 19:41:56 2021

@author: Corbi
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the DataSet
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values # Matrix of Features [all the data, indexes] 
# Won't have to split the training and test set, means having dependant variable but we don't have one.
# we will be creating and identifying the dependant variables 
# Explanation, we will be not including the CustomerID
# Only need Annual Income and Spending Score to identify our cluster. Others not necessary

# Using Dendogram to find the optimal number of Clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) #Method, minimum variance, minimising variance inside clusters
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Move the longest vertical line in between the horizontal bars. 
# arguments include, linkage function
# Training the Hierarchical Clustering Model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward') # parameters, 
# n_clusters, affinity is type of distance to be computed, and linkage should be ward as minimum variance
y_hc = hc.fit_predict(X) # creation of New variable. 

print(y_hc)
# Visualising the Clusters 

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()