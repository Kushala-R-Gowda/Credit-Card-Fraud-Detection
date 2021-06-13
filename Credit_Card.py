#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Importing Dependencies
import numpy as np
import pandas as pd #To use dataframes to get structured data for analysis
from sklearn.model_selection import train_test_split #to spli our data into training data and split data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score #to check accuracy of the model
from sklearn.preprocessing import LabelEncoder


# In[3]:


#Loading Dataset to pandas dataframe
Dataset = pd.read_csv("PS_20174392719_1491204439457_log.csv")


# In[16]:


# Converting non numerical data type to numerical data
le = LabelEncoder()
Dataset.type = le.fit_transform(Dataset.type)


# In[17]:


# First 5 rows of the dataset
Dataset.head()


# In[5]:


# FLast 5 rows of the dataset
Dataset.tail()


# In[19]:


# Information about the Dataset
Dataset.info()


# In[7]:


# Check the number of missing values in each column
Dataset.isnull().sum()
# We don't have missing values here, If we had missing values we should do imputation


# In[8]:


# Distribution of 0->Legit Transaction and 1-> Fraudulent Transaction
Dataset['isFraud'].value_counts()
# Here we have 6354407 Legit Transaction nad 8213 Fraudulent Transaction, Hence we can say this is an unbalanced data, so if we design our model by this dataset it might not give output for fraudulent transactions


# In[20]:


# separating the data for analysis
legit = Dataset[Dataset.isFraud == 0] # rows with legit transactions
fraud = Dataset[Dataset.isFraud == 1] # rows with fraud transactions
print(legit.shape) # shape returns number of rows and number of columns
print(fraud.shape)


# In[10]:


# Statistical measures of the data
legit.amount.describe()


# In[11]:


fraud.amount.describe()


# In[13]:


# Compare the values for both transactions
Dataset.groupby('isFraud').mean() # group the values based on isFraud value


# In[21]:


# Taking a sample of Legit data using Under-Sampling to get similar distribution of both legit and fraud transactions
# Number of Fraudulent Transaction = 8213
legit_sample = legit.sample(n=8213)# this will extract 8213 datpoints randomly


# In[22]:


# Concatenate legit_sample and fraud dataframes
new_Dataset = pd.concat([legit_sample,fraud], axis = 0) # axis= 0 -> add datapoints rowwise


# In[23]:


new_Dataset.head()


# In[24]:


new_Dataset.tail()


# In[9]:


# Check again for value count
new_Dataset['isFraud'].value_counts()


# In[25]:


# Splitting the data into features and targets(either 0 or 1)
x = new_Dataset.drop(columns=['isFraud','nameOrig','nameDest'],axis=1) # drops the the column isFraud and adds other columns to x
y = new_Dataset['isFraud']
print(x)


# In[23]:


print(y)


# In[26]:


# Split the data into training data and testing data
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2,stratify = y, random_state=2) # x has features and y has label, 0.2(20%) of the x data is stored in X_test and its correspoding label is stored in Y_test and 80% of the x data is stored in X_train and its crresponding label is stored in Y_train
# stratify is used to maintain similar distribution of data, random_state to split in some random way


# In[27]:


print(x.shape,X_train.shape,X_test.shape,Y_train.shape)


# In[40]:


Y_test


# In[30]:


# Model Training - Logistic Regression
model = LogisticRegression()
model.fit(X_train,Y_train)


# In[33]:


# Model Evaluation based on Accuracy Score
X_train_prediction = model.predict(X_train) # label output for X_train values trained
training_data_accuracy = accuracy_score(X_train_prediction,Y_train) # Comparing predicted values and actual Y_train labels
print('Accuracy on Training Data : ',training_data_accuracy) # 90.4% Accuracy


# In[34]:


# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on Training Data : ',test_data_accuracy) # 89.7%


# In[41]:


SingleLinePredict = X_test.tail(1)
prediction = model.predict(SingleLinePredict)
print(prediction)


# In[43]:


FilePredict = X_test.tail(10)
prediction = model.predict(FilePredict)
print(prediction)

