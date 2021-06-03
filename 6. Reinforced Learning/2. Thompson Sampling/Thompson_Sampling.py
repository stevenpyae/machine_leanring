# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:22:09 2021
Probablistic Sampling 
@author: Corbi
"""

#Importing the Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv') # Observation of each user's preferences

# Implementing the Thompson Sampling 
import random
N = 100
d = 10
ads_selected = []
#Step 1
# Create 2 lists 
# N1(n) - the number of times the ad i got rewarded 1 up to round n list of 10 elements
# N0(n) - the number of times the ad i got rewarded 0 up to round n
numbers_of_rewards_1 = [0] * d
numbers_of__rewards_0 = [0] * d

total_reward = 0
for n in range(0,N):
    ad = 0 # index of the ad that will be selected at each round N
    max_random = 0 # variable for Step 3
    for i in range(0,d):
        #Step 2, Take a random draw from the Distribution Below: 
        # Use Beta-variate 
        # APPLY THE FUNCTION NI1 + 1, NI0 + 1
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of__rewards_0[i]+ 1)
        
        # Trick to apply Step 3: Select the ad that has the highest random_beta
        if (random_beta > max_random): # ad is selected
            max_random = random_beta
            ad = i # selecting the index i of the ad
        #No need to to and Else here. because if the condition is not met, then it will continue.
    
    ads_selected.append(ad)# each time the ad is selected then append it into the ad_selected list
    reward = dataset.values[n, ad] #n - usser, ad - column
    if (reward == 1):
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of__rewards_0[ad] = numbers_of__rewards_0[ad] + 1
    
    totoal_reward = total_reward + reward
    
# Visualisation of Thompson Sampling 
plt.hist(ads_selected) # Takes either a single array or a sequence of arrays which are not required to be of the same length
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

        
# The goal is to identify the ad in the minimum number of rounds, 
# how minimum the rounds that the Thompson Sampling can find the Best Ad?
# UCB minimum was 500 rounds, Lets see if the Thompson Sampling require less than 500 rounds 
# You can check by changing the value of N, 10000-> 5000 -> 1000 -> 500
# Minimum rounds is 100
# Thompson Sampling is more powerful 