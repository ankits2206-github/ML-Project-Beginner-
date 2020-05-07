#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[3]:


data = pd.read_csv("train.csv")
data.head()


# In[56]:


data.isnull().sum()


# In[57]:


data['Age'].isnull().sum()
avg=data['Age'].mean().astype(int)

data['Age']=data['Age'].replace(np.NaN,avg)
data['Age']=data['Age'].astype(int)


# In[58]:


data['Cabin']=data['Cabin'].fillna('A')


# In[59]:


data.dropna(inplace=True)
data.isnull().sum()


# In[60]:


data.info()


# In[61]:


train_x=data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']].values
train_y=data['Survived'].values


# In[62]:


from sklearn import preprocessing

sex=data['Sex'].unique()
le_sex=preprocessing.LabelEncoder()
le_sex.fit(sex)

train_x[:,2]=le_sex.transform(train_x[:,2])


# In[65]:


cabin=data['Cabin'].unique()
le_cabin=preprocessing.LabelEncoder()
le_cabin.fit(cabin)

train_x[:,7]=le_cabin.transform(train_x[:,7])

le_embarked=preprocessing.LabelEncoder()
embarked=data['Embarked'].unique()
le_embarked.fit(embarked)
train_x[:,8]=le_embarked.transform(train_x[:,8])


# In[84]:


mod = DecisionTreeClassifier(criterion='entropy')


# In[85]:


mod.fit(train_x,train_y)


# In[86]:


datatest=pd.read_csv('test.csv')

datatest.isnull().sum()


# In[87]:


avgt=datatest['Age'].mean().astype(int)
datatest['Age']=datatest['Age'].fillna(avgt)
datatest['Age']=datatest['Age'].astype(int)
datatest['Cabin']=datatest['Cabin'].fillna('A')


# In[88]:


datatest.dropna(inplace=True)


# In[89]:


test_x=data[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']].values
test_y=data['Survived'].values


# In[90]:


from sklearn import preprocessing

sex=data['Sex'].unique()
le_sex=preprocessing.LabelEncoder()
le_sex.fit(sex)

test_x[:,2]=le_sex.transform(test_x[:,2])
cabin=data['Cabin'].unique()
le_cabin=preprocessing.LabelEncoder()
le_cabin.fit(cabin)

test_x[:,7]=le_cabin.transform(test_x[:,7])

le_embarked=preprocessing.LabelEncoder()
embarked=data['Embarked'].unique()
le_embarked.fit(embarked)
test_x[:,8]=le_embarked.transform(test_x[:,8])


# In[91]:


y_pred = mod.predict(test_x)


# In[92]:


print("Training score ",mod.score(train_x,train_y))
print("Testing score ",mod.score(test_x,test_y))


# In[93]:


from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(test_y, y_pred))

