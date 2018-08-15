
# coding: utf-8

# In[ ]:


#In this project I will create a dutch lyrics generator that mimics lyrics made by Boudewijn De Groot and Bart Peeters
#(although it can work for any artist given enough lyrics material!)


# In[1]:


#Inspiration for this project was drawn from http://karpathy.github.io/2015/05/21/rnn-effectiveness/, who is currently the AI director at Tesla
#And for the Keras part: https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218


# In[3]:


#How this works? By building a character-based Neural Language Model
#This model predicts the next character in the sequence based on the specific characters that have come before it in the sequence


# In[4]:


#Import necessary libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import Adam


# In[5]:


#First, we have to gather all the lyrics in a simple txt format


# In[6]:


#Let's get the lyrics file and convert it into a string

filename = 'C:\\Users\\Yves Vc\\Desktop\\Lyrics generator\\Boudewijn De Groot.txt'

lyrics = open(filename, 'r').read()


# In[7]:


#Return all characters to lower case
#because we don't want to be able to differentiate between upper and lower case

lyrics = lyrics.lower()


# In[8]:


#Get unique characters

chars = list(set(lyrics))


# In[9]:


#Number of characters
#Note: 88924
len(lyrics)


# In[10]:


#Number of unique characters
#Note: 55
len(chars)


# In[11]:


#Let's have a look at the characters
chars


# In[12]:


#There are some characters that aren't really interesting, let's remove them
import re 
lyrics = re.sub('”|\n|`|-|:|[|]', "", lyrics)


# In[13]:


#Create a dictionnary to be able to convert characters into numbers (more machine learning friendly) 

char_nr = {}
for i in range(len(chars)):
    char_nr[chars[i]] = i


# In[14]:


#And also a dictionnary that can convert numbers back into characters (this will be used in the test/predict phase)
nr_char = {}
for i in range(len(chars)):
    nr_char[i] = chars[i]


# In[15]:


#Let's make input X and output y: 
#Each instance of X consists of n_steps consecutive characters in the text eg 'Het is moo'
#and the next instance is moved one character in the text eg 'et is mooi'
#y corresponds to the next (11th) characher for every instance eg for the sentence 'Het is moo', the y value is 'i'
n_steps = 10
sentences = []
next_chars = []
for i in range(len(lyrics) - n_steps):
    sentences.append(lyrics[i: i + n_steps])
    next_chars.append(lyrics[i + n_steps])


# In[16]:


#Now each instance is converted into a vector using 1-of-k encoding with shape of each instance (len(lyrics), len(chars))
#Thus, for every character in an instance, all values of that row are zero except for a single one at the index of the character in the vocabulary
X = np.zeros((len(sentences), n_steps, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_nr[char]] = 1
    y[i, char_nr[next_chars[i]]] = 1


# In[17]:


#Create the learning part. We'll use a recurrent neural network because it is able to work with data in which the sequence is important
#The type of cell used is LSTM (which is almost always used because of its advantages)
#And a dense that to connect all the units with shape (len(chars)) to represent every character value

n_units1 = 150

model = Sequential()
model.add(LSTM(n_units1, input_shape = (n_steps, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))


# In[18]:


lr  = 0.01
validation_split = 0.05
batch_size = 128
epochs = 5


optimizer = Adam(lr=lr)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

history = model.fit(X,y, validation_split =validation_split, batch_size=batch_size, epochs=epochs, shuffle=True).history


# In[19]:


#plot the training and validation accuracy per epoch
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');


# In[20]:


#Onto the test phase: we have to be able to convert a string of text so that is suitable with our model:
#For simplicity reasons, the provided string needs to be at least n_steps characters long.


#returns string capped to n_steps characters and no upper case
def string_n_steps(string):

    text = string.lower()[:n_steps] 
    return text
        

#1-of-k encodes the string
def string_to_nr(string):
    text = string_n_steps(string)
    x = np.zeros((1,n_steps,len(chars)))
    
    for i in range(n_steps):
         x[:,i,char_nr[text[i]]] = 1
    x =  x.astype(int)
    return x





# In[21]:



def lyrics_generator(string):
    output = ''
    for i in range(10):
        vec_input = string_to_nr(string) #returns the string as vector with shape (1,n_steps,len(chars))
        character = model.predict(vec_input) #returns the prediction probability of every character
        output += nr_char[np.argmax(character)] #returns the character of highest prediction probability and adds it to output
        string = string[i+1:n_steps] + output +  nr_char[np.argmax(character)] #returns the string minus the 'oldest' character plus the output plus the newest predicted character
    return output #returns the predicted sentence


# In[23]:


#test1
lyrics_generator('wat een tof weer vandaag')
#Provides the 10 character sentence: per van de 


# In[24]:


#test2
lyrics_generator('dit is een test')
#Provides the 10 character sentence: verder ge


# In[25]:


#test3
lyrics_generator('er is nog werk aan de winkel')
#Provides the 10 character sentence: een dat ik


# In[26]:


#Conclusion:
#After only 5 epochs of training and minimal training set, the lyrics generator is already able to produce words, however without much coherence

#The following actions can be undertaken to improve this
#1. Gather more training data (minimum recommended is 1.000.000 character (only 89.000 in this example))
#2. Let the model train longer
#3. Increase the number of characters the program uses to learn to predict the next character (eg 40)
#(4. Make the model more complex)

