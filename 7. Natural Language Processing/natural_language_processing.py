# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:17:33 2021
tsv - tab separated value. 
Check the tsv files for the number of columns 
@author: Corbi
"""

#Importing the Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter= '\t', quoting = 3) # Observation of each user's preferences
# To state that delimiter is \t means tab 
# To Ignore the Double Quote, quoting parameter set to 3. So that you are free from processing Errors

# Cleaning the Texts (Must HAVE)
import re # re library simplify the reviews, 
import nltk # classic library, download the stop word. Seive out Non relevant word to help the prediction such as The, A, an,

nltk.download('stopwords') # Download the stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # apply stemming on our reviews
# Stemming taking only the root of the word that indicate enough about what it words means 
# Reducing the final dimension of the sparse matrix from Bag of Words. 
corpus = []
# for loop to iterate all the reviews. All letters, lower case and add it to corpus. 
for i in range(0, 1000): # because one thousand reviews
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # remove all punctuation and keep only the letters
    
    # any elememnt that isn't a letter will be replaced by a space
    # ^ means NOT. it shows not a to z and A to Z
    # 3rd argument contain what dataset to clean. 
    
    #lower all the cases
    review = review.lower()
    # Split the different elemnts into different words for Stemming 
    review = review.split() # become a list of different words 
    
    #Stemming 
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not') # all stopwords include NOT, so we are removing the NOT which can be a good indicator
    
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    # if the word is in the stopwords, we don't include it. 
    # for loop inside the same row and apply stemming to each of them 
    review = " ".join(review) # this is to join the back together after the stemming and update our review
    corpus.append(review) # corpus is needed to create the bag of words model 
    # They removed the NOT. indicate a negative review. 
    
# Creating the Bag of Words Model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #Maximum number of columns take the most frequent words
# tokenisation process. Run the cell first then simplify the bag of words model
X = cv.fit_transform(corpus).toarray() #all the words from all the review, transform method will put all the words into columns
print(len(X[0])) # 1566 remove the words that don't help at all. 
#Matrix of features should be 2d array
# Create the dependent vector Y
y = dataset.iloc[:, -1].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Fixing the Seed here

# Training the Naive Bayes Model on the Training Set
# Naive bayes is the best model in terms of NLP
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test Set results
y_pred = classifier.predict(X_test)
# np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
# Accuracy is 73 percent. 

# BONNUS 
# Predicting if a single review is positive or negative 
# Positive Review
new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)

# Negative Review
new_review = 'I hate this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)