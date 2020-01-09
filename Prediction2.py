#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 00:34:01 2020

@author: dhirain
"""

def custom_prediction_accuracy(y_test,y_pred,thresold):
    right = 0
    l = len(y_pred)
    for i in range(0,l):
        if(abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)

# Importing the dataset
import pandas as pd
dataset = pd.read_csv('data/odi.csv')
X = dataset.iloc[:,[7,8,9,12,13]].values
y = dataset.iloc[:, 14].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting random forest
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100,max_features=None)
reg.fit(X_train,y_train)

#Fitting Linear regressor 
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit(X_train,y_train)

#fitting SVR 
from sklearn.svm import SVR
reg_svc = SVR(C=1.0, epsilon=0.2)
#reg_svc.fit(X_train, y_train)

#fitting Decision tree
from sklearn.tree import DecisionTreeRegressor
reg_dec = DecisionTreeRegressor(max_depth=2)
reg_dec.fit(X_train, y_train)

# Testing the dataset on trained model
y_pred = reg.predict(X_test)
score = reg.score(X_test,y_test)*100
print("R square value:" , score)
print("Custom accuracy:" , custom_prediction_accuracy(y_test,y_pred,20))

# Testing with a custom input
import numpy as np
new_prediction = reg.predict(sc.transform(np.array([[100,0,13,50,50]])))
print("Prediction score:" , new_prediction)

