#Predicting Number of times Cricket Chirps in the given temprature

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#Importing dataset
dataset=pd.read_csv('Cricket_chirps.csv') 
X=dataset.iloc[:,0:1].values
y=dataset.iloc[:,1].values

#Splitting the data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting simple linear regression into dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor1 = DecisionTreeRegressor(random_state = 0)
regressor1.fit(X_train, y_train)

#Fitting Random Forest regression into dataset
from sklearn.ensemble import RandomForestRegressor
regressor2 = RandomForestRegressor(n_estimators=500,random_state=0)
regressor2.fit(X_train,y_train)


#Predicting the test set result
y_pred=regressor.predict(X_test)

#Predicting the test set result
y_pred1=regressor1.predict(X_test)

#Predicting the test set result
y_pred2=regressor2.predict(X_test)

"""#Finding error
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))"""

# save the model to disk
filename = 'model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

# save the model to disk
filename = 'model1.pkl'
pickle.dump(regressor1, open(filename, 'wb'))

# save the model to disk
filename = 'model2.pkl'
pickle.dump(regressor2, open(filename, 'wb'))


