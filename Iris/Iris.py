#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[101]:


data = pd.read_csv('iris.csv')
data['Class'].unique()


# In[102]:


data.head()


# In[103]:


real_x = data[['Sepal Length','Sepal Width','Petal length','Petal width']].values
real_y = data['Class'].values


# In[104]:


# from sklearn import preprocessing

# cls = preprocessing.LabelEncoder()
# cls.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
# real_y[:,0]=cls.transform(real_y[:,0])


# In[105]:


train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.2)


# In[106]:


print(train_x.shape,train_y.shape)


# In[107]:


k=10
mean_acc=np.zeros((k-1))
std_acc = np.zeros((k-1))

for i in range(1,k):
    reg = KNeighborsClassifier(n_neighbors=i,).fit(train_x,train_y)
    pred_y=reg.predict(test_x)
    mean_acc[i-1]=metrics.accuracy_score(test_y,pred_y)
    std_acc[i-1]=np.std(pred_y==test_y)/np.sqrt(pred_y.shape[0])
    
mean_acc


# In[108]:


pred_y=reg.predict(test_x)


# In[109]:


k=mean_acc.argmax()+1;
reg=KNeighborsClassifier(n_neighbors=k).fit(train_x,train_y)
print("Training score ",reg.score(train_x,train_y))
print("Testing score ",reg.score(test_x,test_y))

print("K value ",k)


# In[99]:


#Accuracy plot form 1-9 values of K

Ks=10
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

