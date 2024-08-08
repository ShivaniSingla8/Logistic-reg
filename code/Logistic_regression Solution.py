#!/usr/bin/env python
# coding: utf-8

# Data Exploration:

# In[1]:


#Shivani
import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[2]:


train = pd.read_csv("C:\\Users\\ACS\\OneDrive\\Desktop\\New folder\\Titanic_train.csv")
test = pd.read_csv("C:\\Users\\ACS\\OneDrive\\Desktop\\New folder\\Titanic_test.csv")


# In[3]:


train


# In[4]:


train.describe()


# In[5]:


train.describe(include=object)


# In[6]:


train.drop(columns=['Name','SibSp','Parch','Fare','Cabin','Ticket'],inplace=True)


# In[7]:


import matplotlib.pyplot as plt
train.hist()
plt.tight_layout()


# In[8]:


import seaborn as sns
sns.boxplot(train)


# Data Preprocessing:

# In[9]:


train.duplicated().sum()


# In[10]:


train.isna().sum()


# In[11]:


(177/891)*100


# In[12]:


median1 = train['Age'].median()


# In[13]:


train['Age'].fillna(median1 , inplace = True)


# In[14]:


train


# In[15]:


from sklearn.preprocessing import LabelEncoder

Encode=LabelEncoder()

train.iloc[ : ,3]=Encode.fit_transform(train.iloc[:, 3])
train.iloc[ : ,5]=Encode.fit_transform(train.iloc[:, 5])


# In[16]:


train


# In[17]:


X = train.iloc[:, [0,2,3, 4, 5]]
Y = train.iloc[:, 1]


# In[18]:


X


# Model Building

# In[19]:


classifier = LogisticRegression()
classifier.fit(X,Y)


# In[20]:


y_pred = classifier.predict(X)
y_pred


# In[21]:


y_pred_df = pd.DataFrame({'Y':Y,'Yhat':classifier.predict(X)})


# In[22]:


y_pred_df


# Model Evaluation

# In[23]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print(confusion_matrix)


# In[24]:


((463+243)/(463+86+99+243))*100   #accuracy


# In[25]:


from sklearn.metrics import classification_report
print(classification_report(Y, y_pred))


# # **ROC curve**

# In[26]:


from sklearn.metrics import roc_curve #roc-receiver operating characteristic
from sklearn.metrics import roc_auc_score # auc-area under curve

fpr, tpr, thresholds = roc_curve(Y, classifier.predict_proba(X)[:,1])
# we want to predict probability values for x data
# predict_proba returns probability estimates for all classes
# and the results are ordered by the label of classes i.e. 0 and 1.
# [:,1] will get the predicted probabilities of the positive label only
# here we will get false positive rate, true positive rate and threshold values
auc = roc_auc_score(Y, y_pred)# compute roc_auc_score based on y and y predicted

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='Green')
plt.plot([0, 1], [0, 1], 'k--')# x axis range is 0 to 1, y axis range is 0 to 1, k-- is a line type - dotted
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show() # green one is roc curve


# In[27]:


auc


# # **Interpretation**

# we can observe the positive rate of curve through obtained result.  

# The provided code performs data preprocessing, exploratory data analysis, and trains a logistic regression model to predict passenger survival on the Titanic. To discuss the significance of features in predicting the target variable, we can analyze the model coefficients and visualize the relationships between features and survival.
# 

# Deployment

# In[28]:


import pickle


# In[29]:


pickle.dump(classifier,open('classifier.pkl','wb'))

