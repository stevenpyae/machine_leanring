# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:24:39 2021

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


# Usig the elbow method to find the optimal number of clusters
# Running the K mean algorithm several times
from sklearn.cluster import KMeans
#10 different numbers of clusters
wcss = [] # empty list first
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)# Initialise with Kmeans++ #42 bring luck in maths
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # K-mean's inertia is WCSS 

plt.plot(range(1,11), wcss)
plt.title("The Elbow method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Training the K-Meas model on the dataset
# We see that the number of cluster is 5 is the most optimal
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42) # Initialise with Kmeans++ #42 bring luck in maths
y_kmeans = kmeans.fit_predict(X) # Trains and Returns the dependant variables, each group contains the similarities
# Customers will be segmented into 5 clusters, we create these dependant variables to group the customers into 5 clusters
print(y_kmeans)

# Visualisig the clusters 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# X is the Different customers X[Index 0] Annual Income, y_kmeans to select the different  rows,
# Y will be Spending score X[Index 1] Spending Score, y_kmeans to select the different rows

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# This is to label the 5 centroids.
# Get them from the kmeans object. kmeans.cluster_centers_[:, 0] X coordinates
# kmeans.cluster_centers-[:,1] Y coordinate
# 2 days array contains the rows with the coordinates

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()