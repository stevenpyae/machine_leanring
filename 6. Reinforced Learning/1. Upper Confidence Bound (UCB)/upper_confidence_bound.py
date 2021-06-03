# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:30:47 2021
Targeting the Online Advertising
Dataset assumes that each Ad has a fix conversion rate 
Required assumption for both UCB and Thompson Sampling 
@author: Corbi
"""
#Importing the Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv') # Observation of each user's preferences
# In reality, users connect one buy one, Everything happens in real time, not a static process
# The only way to simulate this, is to make 10 real Ads right now. 
# First user, will click on Ad1, Ad5 and Ad9.  
# retrieve information from the Indexes
# Dataset assum

#Implementing UCB
# Create 2 variables - Number of times the ad i was selected up to round n,
# sum of rewards of the ad i up to round n 
import math

N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0]*d
total_reward = 0

#Implementing step 2 and step 3 at each iteration
for n in range(0, N):
    # We want to select an AD according to the steps and have maximum UCB
    ad = 0 # start from the first ad and iterate over all the ads 
    # In order to get the maximum upper confidence bound, we will need to compare the upper confidence bound of each of the ADS
    max_upper_bound = 0 # each of the ad, we will compute the upper_bound so, this variable is needed to compare
    for i in range (0, d): # Iterate the ad from 1 to 10 
        if(numbers_of_selections[i] > 0): #At first, No ad was selected. So, this will check if add has been selected, if not, Line
            # number_of_selections[i] will increase if the ad is selected
            average_reward = sums_of_rewards[i]/numbers_of_selections[i] #Step 2 find Average
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])  # Step 2 apply Formula 
            # because log(0) is infinity, n+1 is used 
            upper_bound = average_reward + delta_i 
            #This upper_bound will be used to compare with the max_upper_bound 
        else:
            #Trick to ensure that the ad is selected 
            upper_bound = 1e400 
            # a super high value to ensure that the upper bound will be selected always 
            # The Ad will then be selected
            
        # Select the ad i that has the maximum UCB 
        if (upper_bound > max_upper_bound): # This will pass the condition for Selecting the Ad 
            max_upper_bound = upper_bound # update the max upper bound value to the new upper_bound
            ad = i # selecting the index i of the ad
    
    # Update the following, Because Ad will be selected
    # ads_selected - Full list of ads selected
    # numbers_of selections - number of times the ad i was selected up to round n
    # sum_of_rewards -  sum of rewards of the ad i up to round n
    # total_reward - total accumulated reward over the round

    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1  #if the ad is selected you increment the selected ad by 1 
    # The reward is on the dataset
    reward = dataset.values[n, ad] #n - usser, ad - column
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward # The reward that we got by selecting this ad, where to we have this reward?
    total_reward = total_reward + reward
    

# Maximize the Clicks that Users will Have.
# Visualising the Results 
# Plotting the histogram 
print(ads_selected) # the sequence of ads that were selected over the rounds, list of 10k elements each element that was selected at a particular round 
plt.hist(ads_selected) # Takes either a single array or a sequence of arrays which are not required to be of the same length
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

# The goal is to identify the ad in the minimum number of rounds, how many rounds the UCB algorithm able to identify the ad with the Highest CTR
# You can check by changing the value of N, 10000-> 5000 -> 1000 -> 500
# 500 rounds is not enough to identify the best Ad with the Highest CTR, 