
# coding: utf-8

# In[1]:


#In this project I will try to detect the presence of heart disease based on 13 different features.
#If a high accuracy is achieved, this will show that we can predict heart disease in people with high certainty.
#This can be very valuable in practice: A lot of these features are already captured in Electronic Health Records, thus these could be used to sooner detect people with heart disease
#The features are: age, gender, chest pain type, resting blood pressure, serum cholestoral in mg/dl, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise induced angina, ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels (0-3) colored by flourosopy, thal 
#The dataset is available on http://archive.ics.uci.edu/ml/datasets/Heart+Disease


# In[2]:


#Importing the libraries we're going to work with

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[3]:


#The data was collected from four different sources. Each database has the same instance format.
#Import the first data file, data collected from Cleveland

Cleveland = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', header = None)


# In[4]:


#Import the second data file, data collected from Budapest

Budapest = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data', header = None)


# In[5]:


#Import the third data file, data collected from Switserland

Switserland = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data', header = None)


# In[8]:


#Import the fourth and last data file, data collected from California

California = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data', header = None)


# In[9]:


#Now lets merge all this data into one DataFrame

heart = pd.concat([Cleveland, Budapest, Switserland, California], axis = 0)


# In[10]:


#Check shape of the df

heart.shape


# In[11]:


#Let's look at the first 5 instanes
#Note: It looks like all are values are numerical
heart.head()


# In[12]:


#Let's rename our columns to make it more clear

heart.rename(columns = {0: 'age', 1: 'sex', 2: 'cp', 3: 'trestbps', 4: 'chol', 5: 'fbs', 6: 'restecg', 7: 'thalach', 8: 'exang', 9: 'oldpeak', 10: 'slope', 11: 'ca', 12: 'thal', 13: 'disease'}, inplace= True)


# In[13]:


#How many missing values are there per feature?
#The missing values are presented as '?', so we need to change this to a NaN value

heart.replace(['?'], [np.nan], inplace=True)


# In[14]:


#Let's check the amount of missing values per column
#Note: we have a lot of missing values for the thal, ca and slope features
heart.count(axis = 0)


# In[15]:


#Let's look at the datatypes
#Note: Features trestbps to thal are objects
heart.info()


# In[16]:


#Convert object features to floats
heart['trestbps'] = heart['trestbps'].astype('float32')
heart['chol'] = heart['chol'].astype('float32')
heart['fbs'] = heart['fbs'].astype('float32')
heart['restecg'] = heart['restecg'].astype('float32')
heart['thalach'] = heart['thalach'].astype('float32')
heart['exang'] = heart['exang'].astype('float32')
heart['oldpeak'] = heart['oldpeak'].astype('float32')
heart['slope'] = heart['slope'].astype('float32')
heart['ca'] = heart['ca'].astype('float32')
heart['thal'] = heart['thal'].astype('float32')


# In[17]:


#Since we have a lot of missing values for the thal, ca and slope features, it's better to delete them

heart.pop('slope')
heart.pop('ca')
heart.pop('thal')


# In[18]:


#We have to impute the missing values for certain features, let's look at their distribution

heart.hist()


# In[19]:


#The features trestbps, chol, restecg, thalach, oldpeak are continuous, so let's impute the missing values using the median

heart['trestbps'].fillna(heart['trestbps'].median(), inplace= True)
heart['chol'].fillna(heart['chol'].median(), inplace= True)
heart['restecg'].fillna(heart['restecg'].median(), inplace= True)
heart['thalach'].fillna(heart['thalach'].median(), inplace= True)
heart['oldpeak'].fillna(heart['oldpeak'].median(), inplace= True)


# In[20]:


#The features fbs, exang are categorical, so let's impute the missing values using the most frequent value

heart['fbs'].fillna(0.0, inplace = True)
heart['exang'].fillna(0.0, inplace = True)


# In[21]:


#Right now the heart disease label consists of 5 categories: 0 (no heart disease) and 1, 2, 3, 4 are severities of heart disease
#Because we are only interested in presence or absence of heart disease, let's combine 1,2,3,4 into one category

heart['disease'].replace([1,2,3,4], [1,1,1,1], inplace = True)


# In[22]:


#Some ML algorithms don't perform well when the input features have very different scales
#Therefore, let's apply StandardScaler to features with a wide range

from sklearn.preprocessing import StandardScaler

std_scal = StandardScaler()
heart[['chol', 'thalach', 'trestbps', 'age']]= std_scal.fit_transform(heart[['chol', 'thalach', 'trestbps', 'age']])



# In[23]:


#Now we will create the X (features) and y (label) array
X = heart.iloc[:,:10]

y = heart['disease']


# In[24]:


#Let's start with the first model, support vector machines
#And use Grid Search to simultaneously test multiple hyperparameters
#Note: the hyperparameters C = 0.1, kernel = 'rbf' and degree = 3 turn out to be the best, achieving an f1 score of 0.82

from sklearn.svm import SVC
svc = SVC()

from sklearn.model_selection import GridSearchCV

grid_svc = [{'C': [0.1,1,5]}, {'kernel': ['poly', 'rbf', 'linear'] }, {'degree': [1,2,3]}]

gridsearch_svc = GridSearchCV(svc, grid_svc, cv = 10, scoring = 'f1')

gridsearch_svc.fit(X, y)

gridsearch_svc.best_params_
gridsearch_svc.best_score_


# In[26]:


#Now we will do the same using Random Forest
#Note: The hyperparameters max_depth = 5, min_samples_leaf = 25 provided the best score, f1 = 0.81

from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier()

grid_ran_for = [{'max_depth': [5,6,7]}, {'min_samples_leaf': [20,25,30]}]

gridsearch_ran_for = GridSearchCV(ran_for, grid_ran_for, cv = 10, scoring = 'f1')

gridsearch_ran_for.fit(X,y)

gridsearch_ran_for.best_params_
gridsearch_ran_for.best_score_


# In[30]:


#Finally, kNN.
#Note: the f1 score is 0.81 with the most optimal hyperparameter being n_neighbors = 5

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

grid_knn = [{'n_neighbors': [3,5]}]

gridsearch_knn = GridSearchCV(knn, grid_knn, cv = 10, scoring  = 'f1')

gridsearch_knn.fit(X,y)

gridsearch_knn.best_params_
gridsearch_knn.best_score_


# In[31]:


#Let's try ensembling these models to get an even higher accuracy, using Hard voting
#Note: The ensembling the models gives us an accuracy of 0.83!

from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators = [('svc', SVC(C = 0.1, kernel = 'rbf', degree =3)), ('for', RandomForestClassifier(max_depth = 5, min_samples_leaf = 25)), ('knn',KNeighborsClassifier())], voting = 'hard')


# Divide into training and test to prevent overfitting to training data

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
voting.fit(X_train,y_train)
y_pred = voting.predict(X_test)

accuracy_score(y_pred, y_test)


# In[32]:


#Final note: This method achieves an accuracy of around 0.83 on the test data. This is similar to the accuracy on the training data (not shown in code)

#The following steps could be undertaken to further improve this:
# Gather more features
# Gather more data
# Do a more extensive Grid Search
# Use a more complex model

