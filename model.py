#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


heartData = pd.read_csv("heart.csv")


# In[3]:


heartData.head()


# In[4]:


heartData.shape


# In[5]:


heartData.info()


# In[6]:


#checking if there are any missing values in the dataset
heartData.isnull().sum()


# In[7]:


heartData.describe()


# In[8]:


#number of people who have heart disease or do not have it
heartData['target'].value_counts()


# In[9]:


# splitting the features and the target(1=defective heart, 0 = healthy heart)
X = heartData.drop(columns='target',axis=1)
Y = heartData['target']


# In[10]:


X


# In[11]:


Y


# In[12]:


#testing and training data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=2)


# In[13]:


X.shape,X_train.shape,X_test.shape


# In[14]:


model = LogisticRegression(max_iter=1000) 


# In[15]:


model.fit(X_train,Y_train)


# In[16]:


#finding accuracy of model
Xprediction = model.predict(X_train)
training_accuracy = accuracy_score(Xprediction, Y_train)


# In[17]:


print("The accuracy is:",training_accuracy*100)


# In[18]:


#accuracy score for the test data
Xtest_prediction = model.predict(X_test)
testing_accuracy = accuracy_score(Xtest_prediction,Y_test)


# In[19]:


print("The accuracy is:",testing_accuracy*100)







