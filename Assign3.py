#!/usr/bin/env python
# coding: utf-8

# Q1) Load the dataset and split it into a training set (70%) and a test set (30%). 

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import time

from sklearn.datasets import load_breast_cancer 
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

#Splitting the data into Training Set and Test Set 
from sklearn.model_selection import train_test_split 
X_trainOrig, X_testOrig, y_trainOrig, y_testOrig = train_test_split(X,y,test_size=0.3,random_state=0)

#Normalizing the features 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X_trainOrig = sc_X.fit_transform(X_trainOrig) 
X_testOrig = sc_X.transform(X_testOrig)


# Q2) Train Logistic Regression on the dataset and time how long it takes. Look up how to compute execution time of Python code.

# In[3]:


start_time = time.time()

#Fitting Logistic Regression to Training Set 
from sklearn.linear_model import LogisticRegression 
classifierObj = LogisticRegression(random_state=0) 
classifierObj.fit(X_trainOrig, y_trainOrig)

print("--- %s seconds ---" % (time.time() - start_time))

#--- 0.01557 seconds ---


# Q3) Evaluate the resulting model on the test set.

# In[4]:


#Making predictions on the Test Set 
y_predOrig = classifierObj.predict(X_testOrig)

#Evaluating the predictions using a Confusion Matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_testOrig, y_predOrig)
cm

#array([[ 60,   3],
#        [  1, 107]], dtype=int64)


# Q4) Next, use PCA to reduce the dataset’s dimensionality, with an explained variance ratio of at least 95%.

# In[9]:


#Applying PCA 
from sklearn.decomposition import PCA 
pcaObj = PCA(n_components=12)
X_trainPCA = pcaObj.fit_transform(X_trainOrig) 
X_testPCA = pcaObj.transform(X_testOrig) 
components_variance = pcaObj.explained_variance_ratio_
components_variance


# Q5) Train a new Logistic Regression classifier on the PCA reduced dataset and see how long it takes. Was training much faster?

# In[11]:


start_time = time.time()

#Fitting Logistic Regression to Training Set 
from sklearn.linear_model import LogisticRegression 
classifierObj = LogisticRegression(random_state=0) 
classifierObj.fit(X_trainPCA, y_trainOrig)

print("--- %s seconds ---" % (time.time() - start_time))

#Making predictions on the Test Set 
y_predPCA = classifierObj.predict(X_testPCA)
#about the same 0.01562 secs vs. 0.01557 secs


# Q6) Next evaluate the classifier on the test set: how does it compare to the previous classifier? 

# In[12]:


#Evaluating the predictions using a Confusion Matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_testOrig, y_predPCA)
cm

#previous classifier confusion matrix  
#array([[ 60,   3],
#       [  1, 107]], dtype=int64)

#This classifier
#array([[ 61,   2],
#       [  5, 103]], dtype=int64)


# Q7) Use LDA to reduce the dataset’s dimensionality down to 2 linear discriminants.

# In[13]:


#Applying LDA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
ldaObj = LDA(n_components=2) 
X_trainLDA = ldaObj.fit_transform(X_trainOrig,y_trainOrig)
X_testLDA = ldaObj.transform(X_testOrig)


# Q8) Train a new Logistic Regression classifier on the LDA reduced dataset and see how long it takes.

# In[15]:


start_time = time.time()

#Fitting Logistic Regression to Training Set 
from sklearn.linear_model import LogisticRegression 
classifierObj = LogisticRegression(random_state=0) 
classifierObj.fit(X_trainLDA, y_trainOrig)

print("--- %s seconds ---" % (time.time() - start_time))

#Making predictions on the Test Set 
y_predLDA = classifierObj.predict(X_testLDA)

#--- 0.01562 seconds ---


# Q9) Evaluate the classifier on the test set.

# In[16]:


#Evaluating the predictions using a Confusion Matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_testOrig, y_predLDA)
cm
#array([[ 59,   4],
#[  2, 106]], dtype=int64)


# Q10) Use Kernel PCA to reduce the dataset’s dimensionality down to 2 features.

# In[17]:


#kernel PCA 
from sklearn.decomposition import KernelPCA 
kernelPCAObj = KernelPCA(n_components=2, kernel='rbf') 
X_trainKernalPCA = kernelPCAObj.fit_transform(X_trainOrig) 
X_testKernalPCA = kernelPCAObj.transform(X_testOrig)


# Q11) Train a new Logistic Regression classifier on the Kernel PCA reduced dataset and see how long it takes.

# In[18]:


start_time = time.time()

#Fitting Logistic Regression to Training Set 
from sklearn.linear_model import LogisticRegression 
classifierObj = LogisticRegression(random_state=0) 
classifierObj.fit(X_trainKernalPCA, y_trainOrig)

print("--- %s seconds ---" % (time.time() - start_time))

#Making predictions on the Test Set 
y_predKernalPCA = classifierObj.predict(X_testKernalPCA)

#--- 0.01562 seconds ---


# Q12) Evaluate the classifier on the test set.

# In[19]:


#Evaluating the predictions using a Confusion Matrix 
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_testOrig, y_predKernalPCA)
cm

#array([[56,  7],
#[10, 98]], dtype=int64)


# In[ ]:




