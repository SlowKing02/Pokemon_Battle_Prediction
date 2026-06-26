#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 10:29:03 2018

@author: slowking
"""

import numpy as np
import pandas as pd
from sklearn import metrics 

from catboost import CatBoostClassifier



# initialize data
X_train = pd.read_csv("Train_Test_Data/X_train.csv")
X_test = pd.read_csv("Train_Test_Data/X_test.csv")
X_predict = pd.read_csv("Train_Test_Data/X_predict.csv")
y_train = pd.read_csv("Train_Test_Data/y_train.csv")
y_test = pd.read_csv("Train_Test_Data/y_test.csv")

X_train_a = X_train.drop(columns=['Number']).values
X_test_a = X_test.drop(columns=['Number']).values
X_predict_a = X_predict.drop(columns=['Number']).values
y_train_a = y_train.drop(columns=['Number']).values.flatten()
y_test_a = y_test.drop(columns=['Number']).values.flatten()

model = CatBoostClassifier(iterations=50, bagging_temperature=2, random_strength=10, boosting_type='Ordered', depth=9, loss_function='Logloss', logging_level='Verbose')
model.fit(X_train_a, y_train_a)

prediction = model.predict(X_test_a)

acc_catboost = round(model.score(X_test_a, y_test_a) * 100, 2)
metrics.accuracy_score(prediction,y_test_a)
