
# coding: utf-8

# In[1]:


#In this project we'll try to correctly classify cell nuclei as benign or malign
#For this, we'll use a Feed-forward neural network. Although NN's aren't optimal for limited-sized data sets, this is a good opportunity to try them out
#Information about the data set:
#  Features are computed from a digitized image of a fine needle spirate (FNA) of a breast mass.  They describe characteristics of the cell nuclei present in the image.
#  Number of instances: 569 (357 benign, 212 malignant)


# In[2]:


#Let's start by importing the relevant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# In[3]:


#Import the data into a DF

df = pd.read_csv('http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat', header = None)


# In[4]:


#What is the shape of the dataset?
#Note: (569, 31)

df.shape


# In[5]:


#Let's look at the first 5 instances

df.head()


# In[6]:


#The first column represent the ID number and the second column the label (benign or malign)
#Let's give them more appropriate names

df.rename(columns = {0: 'ID', 1: 'y_label'}, inplace = True)


# In[7]:


#Because we don't need to know the ID number in this exercise, let's delete it

df.pop('ID')


# In[8]:


#Let's look at the data types of the values in the columns
#Note: all values are float64, except the y label which has object values (either M or B)
df.info()


# In[9]:


#Let's look for the presence of missing values
#Note: There a no missing values!
df.count()


# In[10]:


#Let's make the data more ML friendly by presenting it in only numerical values

df['y_label'].replace(['M', 'B'], ['1', '0'], inplace = True)


# In[11]:


#Now we create the features matrix X and target array y, suitable for Tensorflow

X_df = df.iloc[:,1:].values
X_df = X_df.astype('float32')
y_df = df['y_label'].values
y_df = y_df.astype('int32')


# In[12]:


#And make a training and test set. Due to the limited number of instances we cannot make a training, dev and test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_df,y_df, train_size = 0.9)


# In[13]:


#Now let's start creating the network
#First, we'll create a placeholder for X and y in which we can feed the instances

X = tf.placeholder(tf.float32, shape = (None,30))
y = tf.placeholder(tf.int32, shape = (None))


# In[14]:


#Let's now construct the core of our network: the neurons

from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm

#Define the number of neurons in each layer
n_features = 30
n_hidden1 = 10
n_hidden2 = 5
n_outputs = 2


#The next line is to differentiate between training and test data: When is_training is True, the moving_mean and moving_variance need to be updated
is_training = tf.placeholder(tf.bool, shape = ())
bn_params = {'is_training': is_training, 'decay': 0.99, 'updates_collections': None}

#This is to prevent having to write the following parameters for every layer
with tf.contrib.framework.arg_scope([fully_connected], weights_initializer = tf.contrib.layers.variance_scaling_initializer(), normalizer_fn= batch_norm, normalizer_params = bn_params):

    #Creation of the layers
    hidden1 = fully_connected(inputs = X, num_outputs = n_hidden1, activation_fn = tf.nn.relu)
    hidden2 = fully_connected(inputs = hidden1, num_outputs = n_hidden2, activation_fn = tf.nn.relu)
    logits = fully_connected(inputs = hidden2, num_outputs = n_outputs, activation_fn = tf.nn.softmax)


# In[15]:


#Now we have to define the cost function, which the model will try to improve every iteration

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits = logits)
    loss = tf.reduce_mean(xentropy)


# In[16]:


#Now we define the opimizer: this is how the model will 'learn'
#We'll use ADAM, the learning rate is typically set to 0.001

with tf.name_scope('train'):
    optimizer  = tf.train.AdamOptimizer(learning_rate = 0.00001)
    training_op = optimizer.minimize(loss)


# In[17]:


#Define the evaluation metric

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,y,1)
    acc = tf.reduce_mean(tf.cast(correct,tf.float32))


# In[18]:


# Put long code in variable name (will be used for the initialization of variables)

init = tf.global_variables_initializer()


# In[19]:


#Now, let's create the execution phase

#Define how many times to iterate over the entire dataset
n_epochs = 100

with tf.Session() as sess:
    for epoch in range (n_epochs):
        init.run()
        sess.run(training_op, feed_dict = {X: X_train, y: y_train, is_training: True})
        if epoch % 10 == 0:
            acc_train = acc.eval(feed_dict = {X: X_train, y: y_train, is_training: True})
            acc_test = acc.eval(feed_dict = {X: X_test, y: y_test, is_training: False})
            print('training accuracy: ', acc_train)


# In[20]:


#Final note: As expected, the neural network isn't really capable of accurately classifying the instances due to the limited sample size
#Measures were undertaken (such as different number of layers/units, regularization) but ineffective
#Nonetheless, it was a nice data set to try a neural network on

