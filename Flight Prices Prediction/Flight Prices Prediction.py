#!/usr/bin/env python
# coding: utf-8

# Training data is combination of both categorical and numerical also we 
# can see some special character also being used because of which we have to do data Transformation on it before applying it to our model
# 
# Appending of the data set is done to work together with both train and test at a same time and don,t have to make changes separately.After we apply the transformation then we can separate them again into test and train
# 
# Now we will clean the data set and make it organised, as we can see that date is in format of DD/MM/YY , Arrival time also contain no. of stops and route contain '->' symbol.
# 
# In the column ‘Arrival_Time’,if we see we have combination of both time and month but we need only the time details out of it so we split the time into ‘Hours’ and ‘Minute’.
# 
# 
# Total Stops column is combination of number and a categorical variable like ‘1 stop’ . So we need only the number details from this column so we split that and take the number details only also we change the ‘non stop’ into ‘0 stop’ and convert the column into integer type
# 
# We will follow the same procedure for Dep_time columns as we have  followed for Arrival_Time column
# 
# 
# 
# 
# The ‘Route’ columns mainly tell us that how many cities they have taken to reach from source to destination .This column is very important because based on the route they took will directly effect the price of the flight So We split the Route column to extract the information .Regarding the ‘Nan’ values we replace those ‘Nan’ values with ‘None’ .
# 
# 
# To convert categorical text data into model-understandable numerical data, we use the Label Encoder class. So all we have to do, to label encode a column is import the LabelEncoder class from the sklearn library, fit and transform the column of the data, and then replace the existing text data with the new encoded data.
# 
# 
# 

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV




train_df = pd.read_excel('Data_Train.xlsx')
test_df=pd.read_excel('Test_set.xlsx')

train_df


# In[52]:


big_df = train_df.append(test_df)



big_df['Date'] = big_df['Date_of_Journey'].str.split('/').str[0]
big_df['Month'] = big_df['Date_of_Journey'].str.split('/').str[1]
big_df['Year'] = big_df['Date_of_Journey'].str.split('/').str[2]



big_df['Date'] = big_df['Date'].astype(int)
big_df['Month'] = big_df['Month'].astype(int)
big_df['Year'] = big_df['Year'].astype(int)




big_df=big_df.drop(['Date_of_Journey'], axis=1)
big_df['Arrival_Time'] = big_df['Arrival_Time'] .str.split(' ').str[0]
big_df['Total_Stops']=big_df['Total_Stops'].fillna('1 stop')
big_df['Total_Stops']=big_df['Total_Stops'].replace('non-stop','0 stop')
big_df['Stop'] = big_df['Total_Stops'].str.split(' ').str[0]
big_df['Stop'] = big_df['Stop'].astype(int)





big_df=big_df.drop(['Total_Stops'], axis=1)



big_df['Arrival_Hour'] = big_df['Arrival_Time'] .str.split(':').str[0]
big_df['Arrival_Minute'] = big_df['Arrival_Time'] .str.split(':').str[1]

big_df['Arrival_Hour'] = big_df['Arrival_Hour'].astype(int)
big_df['Arrival_Minute'] = big_df['Arrival_Minute'].astype(int)
big_df=big_df.drop(['Arrival_Time'], axis=1)




big_df['Dep_Hour'] = big_df['Dep_Time'] .str.split(':').str[0]
big_df['Dep_Minute'] = big_df['Dep_Time'] .str.split(':').str[1]
big_df['Dep_Hour'] = big_df['Dep_Hour'].astype(int)
big_df['Dep_Minute'] = big_df['Dep_Minute'].astype(int)
big_df=big_df.drop(['Dep_Time'], axis=1)





big_df['Route_1'] = big_df['Route'] .str.split('→ ').str[0]
big_df['Route_2'] = big_df['Route'] .str.split('→ ').str[1]
big_df['Route_3'] = big_df['Route'] .str.split('→ ').str[2]
big_df['Route_4'] = big_df['Route'] .str.split('→ ').str[3]
big_df['Route_5'] = big_df['Route'] .str.split('→ ').str[4]




big_df['Price'].fillna((big_df['Price'].mean()), inplace=True)





big_df['Route_1'].fillna("None",inplace = True)
big_df['Route_2'].fillna("None",inplace = True)
big_df['Route_3'].fillna("None",inplace = True)
big_df['Route_4'].fillna("None",inplace = True)
big_df['Route_5'].fillna("None",inplace = True)






big_df=big_df.drop(['Route'], axis=1)
big_df=big_df.drop(['Duration'], axis=1)




from sklearn.preprocessing import LabelEncoder

lb_encode = LabelEncoder()
big_df["Additional_Info"] = lb_encode.fit_transform(big_df["Additional_Info"])
big_df["Airline"] = lb_encode.fit_transform(big_df["Airline"])
big_df["Destination"] = lb_encode.fit_transform(big_df["Destination"])
big_df["Source"] = lb_encode.fit_transform(big_df["Source"])
big_df['Route_1']= lb_encode.fit_transform(big_df["Route_1"])
big_df['Route_2']= lb_encode.fit_transform(big_df["Route_2"])
big_df['Route_3']= lb_encode.fit_transform(big_df["Route_3"])
big_df['Route_4']= lb_encode.fit_transform(big_df["Route_4"])
big_df['Route_5']= lb_encode.fit_transform(big_df["Route_5"])




df_train = big_df[0:10683]
df_test = big_df[10683:]
df_test = df_test.drop(['Price'], axis =1)




X = df_train.drop(['Price'], axis=1)
y = df_train.Price



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# Lasso regression is similar to linear regression .In linear regression we try to minimize the mean square error and find the best fitting line corresponding to the minimum mean square error where as in lasso we minimize the sum of mean square error and the absolute value of slope of the input data multiplied by a tuning parameter lambda is which chosen by cross validation.As lambda increases, shrinkage occurs so that variables that are at zero can be thrown away.It basically increases some biasness by reducing dependence on the input data and decreases the variance and results in some better prediction.

# In[56]:


from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


lasso = make_pipeline(RobustScaler(),
                      LassoCV(max_iter=1e7,
                              random_state = 42, cv=kfolds))


lasso_fit = lasso.fit(X_train, y_train)


df_test_las = test_df
preds_1 = lasso_fit.predict(df_test)
preds_2= lasso_fit.predict(X_test)
df_test_las['Price'] = preds_1
df_test_las.to_csv('flight_price_10.csv')


rmse = np.sqrt(mean_squared_error(y_test, lasso_fit.predict(X_test)))
print("RMSE: %f" % (rmse))



print("Mean absolute error: %.2f" % np.mean(np.absolute(preds_2 - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((preds_2 - y_test) ** 2))

print("Train score",lasso_fit.score(X_train,y_train))
print("Test score",lasso_fit.score(X_test,y_test))


# Ridge regression is also similar to linear regression as well as to lasso regression.In linear regression we try to minimize the mean square error and find the best fitting line corresponding to the minimum mean square error where as in lasso we minimize the sum of mean square error and the absolute value of slope of the input data multiplied by a tuning parameter lambda which is chosen by cross validation but in ridge we minimize the sum of mean square error and the slope value's square of the input data multiplied by a tuning parameter lambda which is also chosen by cross validation.In opposite to lasso,the variables that are at zero are not thrown away,but their contribution is minimized.It also increases some biasness by reducing dependence on the input data and decreases the variance and results in some better prediction.

# In[55]:


from sklearn.linear_model import RidgeCV

ridge = make_pipeline(RobustScaler(), 
                      RidgeCV(cv=kfolds))


ridge_fit = ridge.fit(X_train, y_train)



df_test_rid = test_df
preds_1 = ridge_fit.predict(df_test)
preds_2= ridge_fit.predict(X_test)
df_test_rid['Price'] = preds_1
df_test_rid.to_csv('flight_price_10.csv')

rmse = np.sqrt(mean_squared_error(y_test, ridge_fit.predict(X_test)))
print("RMSE: %f" % (rmse))


print("Mean absolute error: %.2f" % np.mean(np.absolute(preds_2 - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((preds_2 - y_test) ** 2))


print("Train score",ridge_fit.score(X_train,y_train))
print("Test score",ridge_fit.score(X_test,y_test))


# XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework but it uses a unique type of regression tree. It is mostly used for large and complicated datasets. It start by taking some initial prediction and add the residuals to the leaf by calculating the averages of two consecutive samples and splitting the branch in two parts based on that average itself. Now, it calculates the similarity score for the residuals and calculate the gain of each branch of the tree. The largest gain will decide how we have to divide the residuals and do the same thing for some predefined levels. We prune the tree with negative (gain - gamma) value.We can't prune a branch if it's child is not pruned. We have made the first tree and now we will find the predicted value for every residuals. Now we will use some learning rate in order to predict the values which is called eta in XGBoost and will gradually head towards the more accurate prediction.

# In[63]:


import xgboost as xgb
from xgboost import XGBRegressor

xgb3 = XGBRegressor(learning_rate =0.1, n_estimators=200, max_depth=10,
                    min_child_weight=5, subsample=0.7,
                    reg_alpha=0.00006,random_state=0)

xgb_fit = xgb3.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = np.sqrt(mean_squared_error(y_test, xgb_fit.predict(X_test)))
print("RMSE: %f" % (rmse))


df_test_xgb = test_df
preds_1 = xgb_fit.predict(df_test)
preds_2= xgb_fit.predict(X_test)
df_test_xgb['Price'] = preds_1
df_test_xgb.to_csv('flight_price_10.csv')



print("Mean absolute error: %.2f" % np.mean(np.absolute(preds_2 - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((preds_2 - y_test) ** 2))
print("R2-score: %.2f" % r2_score(preds_2 , y_test))


print("Train score",xgb_fit.score(X_train,y_train))
print("Test score",xgb_fit.score(X_test,y_test))


# In[64]:


# For training set
import matplotlib.pyplot as plt
sample = list(range(1,7479))
plt.plot(sample,xgb_fit.predict(X_train),color="Blue",label="Predicted Data")
plt.scatter(sample,y_train,color="Green",label="Actual Data")
plt.xlabel("Sample Numbers")
plt.ylabel("Price")
plt.title("Training Set-XGBoost")
plt.legend()
plt.show()


# In[65]:


# For testing set
sample = list(range(1,3206))
plt.plot(sample,xgb_fit.predict(X_test),color="Red",label="Predicted Data")
plt.scatter(sample,y_test,color="Green",label="Actual Data")
plt.xlabel("Sample Numbers")
plt.ylabel("Price")
plt.title("Testing Set-XGBoost")
plt.legend()
plt.show()


# In[37]:


df_test_xgb

