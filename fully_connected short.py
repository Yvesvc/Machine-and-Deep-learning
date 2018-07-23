
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
plt.style.use('seaborn-white')
import tensorflow as tf


# In[2]:


from sklearn.datasets import load_digits
digits = load_digits()


# In[3]:



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target)


# In[4]:


n_features = 64
n_hidden1 = 100
n_hidden2 = 50
n_hidden3 = 25
n_hidden4 = 10
n_hidden5 = 5
n_output = 10


# In[5]:


X = tf.placeholder(tf.float32, shape = (None, 64), name = 'X')
y = tf.placeholder(tf.int64, shape = (None), name = 'y')


# In[6]:


from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm



with tf.contrib.framework.arg_scope([fully_connected], weights_initializer =tf.contrib.layers.variance_scaling_initializer(), weights_regularizer = tf.contrib.layers.l1_regularizer(scale = 0.01), activation_fn = tf.nn.elu):
    hidden1 = fully_connected(inputs = X, num_outputs = n_hidden1)
    hidden2 = fully_connected(inputs = hidden1, num_outputs = n_hidden2)
    hidden3 = fully_connected(inputs = hidden2, num_outputs = n_hidden3)
    hidden4 = fully_connected(inputs = hidden3, num_outputs = n_hidden4)
    hidden5 = fully_connected(inputs = hidden4, num_outputs = n_hidden5)
    logits = fully_connected(inputs = hidden5, num_outputs = n_output, activation_fn = None) 


# In[7]:


with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    base_loss = tf.reduce_mean(xentropy, name = 'loss')
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_loss, name= 'loss')


# In[8]:


with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    training_op = optimizer.minimize(loss)


# In[9]:


with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits,y,1)
    acc = tf.reduce_mean(tf.cast(correct, tf.float32))


# In[10]:


get_ipython().run_cell_magic('time', '', "\nn_epochs = 100\nbatch_size = 50\n\nwith tf.Session() as sess:\n    tf.global_variables_initializer().run()\n    for epoch in range(n_epochs):\n        for iteration in range(X_train.shape[0] // batch_size):\n            X_batch = X_train[:iteration * batch_size]\n            y_batch = y_train[:iteration * batch_size]\n            sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})\n    acc_test = acc.eval(feed_dict = {X: X_test, y: y_test})\n    print ('The accuracy of the model is', acc_test)\n    ")

