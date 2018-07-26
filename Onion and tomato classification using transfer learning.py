
# coding: utf-8

# In[1]:


#Today we are going to classify onions and tomatoes. There is a personal motive behind this
#In my personal opinion tomatoes are one of the most delicious vegatables in the world and they give us our beloved Ketchup
#Whereas onions are the devil's vegatable yet it is present in many dishes
#That's why it would be interesting to see if a computer could accurately distinguish the two (and in the future warn me when onion is present in a dish!)


# In[2]:


#The method we are going to use is transfer learning
#This means that we are going to take an existing image classification model and fine tune it a bit
#There are numerous advantages:
#-Leverage from the most powerful models in the world
#-Needs fewer training data
#-Saves you computing resources and time


# In[ ]:


#The following code is mostly based on videos by Deep lizard (https://www.youtube.com/channel/UC4UJ26WkceqONNF5S26OiVw)


# In[3]:


#About the model, vgg16
#This model won the 2016 ImageNet competition (in which images had to be classified in 1000 categories!)
#We'll finetune it to make it distinguish between onions and tomatoes

#About Keras
#Keras is an easy-to-use API for deep learning


# In[4]:


#First, let's import all the necessary libraries

import numpy as np

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#Create the variables that contain the path to our images
#Each folder must contain two folders (onion and tomato) containing the images

train_path = 'C:\\Users\Yves Vc\Documents\onitom/onion-and-tomato/train'
valid_path = 'C:\\Users\Yves Vc\Documents\onitom/onion-and-tomato/valid'



# In[6]:


#Now we are creating the batches for our train and test set respectively
#ImageDataGenerator generates batches of Tensor image data and the size must be 224,224 for the model. Batch size is the size we want to iterate on.

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size= (224,224), classes= ['onion', 'tomato'], batch_size = 10 )
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size= (224,224), classes= ['onion', 'tomato'], batch_size = 10 )



# In[7]:


#Import the model from keras and store it in a variable (including the 138,357,544! the parameters)
vgg16_model = keras.applications.vgg16.VGG16()


# In[8]:


#Lets look at a summary of the layers
#It can be seen that the last layer has shape (None, 1000). Meaning it can classify an image in one out of 1000 categories
vgg16_model.summary()


# In[9]:


#Check the type of the model

type(vgg16_model)


# In[10]:


#Let's change the type of the model to a Sequential model
#Because we are not interested in classifying our image in one out of the 1000 classes, we do not include the output layer
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)


# In[11]:


model.summary()


# In[12]:


#Now let's make all the layers 'untrainable', meaning their parameters won't change (which will save us a ton of computer resources and time)
for layer in model.layers:
    layer.trainable = False


# In[13]:


#Now add a Dense layer that classyfies our images in one of the two classes (onions or tomato)
model.add(Dense(2,activation= 'softmax'))


# In[14]:


model.summary()


# In[15]:


#For the learning part, we'll use Adam and categorical_crossentropy for our loss function
#The metric that will be used for evaluation is accuracy (because this is a classification task)
model.compile(Adam(lr=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[16]:


#Execution phase!
#The input is the train_batches, we will loop over the data 3 times and measure the training and validation accuracy simultaneously

model.fit_generator(train_batches, steps_per_epoch =4, validation_data = valid_batches, validation_steps = 4, epochs= 3, verbose = 2)


# In[17]:


#The model already reached a validation accuracy of 100% after the second epoch!
#Despite we only used 40 training instances for each class, this shows the potential of transfer learning
#It should be noted that only unpeeled and uncut images of onions and tomatoes were used as input.
#If we would use more diverse pictures, we would need a larger amount of data or add an additional layer

