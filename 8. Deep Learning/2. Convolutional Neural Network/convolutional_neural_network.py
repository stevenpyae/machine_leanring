# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:43:38 2021
Convoluted Neural Network
Tensorflow or PyTorch
Dataset: Section 4- - Convolutional Neural Networks (CNN).zip
Size: 200MB
3 Folders : Training Set, Test Set, Single_Prediction
4000 images of Dogs and Cats - Training Set
1000 images of Dogs and Cats - Test Set

@author: Corbi
"""

# Importing the libraries
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
'''Part 1: Data Preprocessing'''

# Preprocessing the Training Set- APPLY TRANSFORMATION on images to avoid Overfitting (High accuracy on Training, low Accuracy on Test Set)
# Simple flips, zoom in zoom out, rotated - ImageAugmentation
# Keras - Image Data Preprocessing, TimeSeries Data Preprocessing, Text Data Preprocessing
train_datagen = ImageDataGenerator(
                rescale = 1./255,     #Feature Scaling because 0 to 255 pixel value (Compulsory)
                shear_range = 0.2,
                zoom_range=0.2,
                horizontal_flip=True) # Transformation to avoid Overfitting

# Import the training folder
training_set = train_datagen.flow_from_directory(
                    "dataset/training_set", #Path leading to your training set
                    target_size = (64,64), #Training is very long
                    batch_size = 32,
                    class_mode = 'binary') #Binary or Categorical

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255) # Feature Scaling

test_set = test_datagen.flow_from_directory(
                    "dataset/test_set", #Path leading to your training set
                    target_size = (64,64), #Training is very long
                    batch_size = 32,
                    class_mode = 'binary') #Binary or Categorical

'''Part 2: Building the CNN'''
# Initialising the CNN
# Sequence of layers
cnn = tf.keras.models.Sequential()

# Step 1: Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32,kernel_size= 3, activation = 'relu', input_shape = [64,64,3]))
# Filters - Number of feature detectors youw want to apply, Classic Architecture of 32
# Kernel_size - 3x3 matrix
# activation must be rectifier
# Specify the Input shape of your inputs 3 dimensions RGB, and cuz we resized (64,64,3)
# MUST ADD THE input_shape

# Step 2: Pooling Layer to our CNN 
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
# Pool_size = 2x2 box hence pool size is 2 
# Stride = how the box will be shifted, we are sliding by 2 pixels. 

# Adding a second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters = 32,kernel_size= 3, activation = 'relu'))
# Don't need the input_shape. 
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))

# Step 3: Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4: Full Connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

# Step 5: Output Layer
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
'''Part 3: Training the CNN'''
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Training the CNN on the Training set and Evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
# Accuracy of 0.8145 percent

'''Part 4: Making a single prediction'''

import numpy as np 
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64))
# Convert the PIL into an array. Predict method needs a 2d Array
test_image = image.img_to_array(test_image)
# Batches of Images, Each Batch contains 32 images, 
test_image = np.expand_dims(test_image, axis = 0) # where you want to add the dimensions
# Dimension that we are going to add is on the first dimension. 
result = cnn.predict(test_image)
# Call the class indices attributes. 1 correspond to Dog, 0 correspond to Cat
training_set.class_indices
if (result[0][0] == 1): # this is the prediction, accessing the batch and accessing the first image in the batch
    prediction = 'Dog'
else:
    prediction = 'Cat'
    
print(prediction)
    
test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64,64))
# Convert the PIL into an array. Predict method needs a 2d Array
test_image = image.img_to_array(test_image)
# Batches of Images, Each Batch contains 32 images, 
test_image = np.expand_dims(test_image, axis = 0) # where you want to add the dimensions
# Dimension that we are going to add is on the first dimension. 
result = cnn.predict(test_image)
# Call the class indices attributes. 1 correspond to Dog, 0 correspond to Cat
training_set.class_indices
if (result[0][0] == 1): # this is the prediction, accessing the batch and accessing the first image in the batch
    prediction = 'Dog'
else:
    prediction = 'Cat'
    
print(prediction)
