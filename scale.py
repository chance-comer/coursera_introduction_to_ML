# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 11:42:25 2018

@author: kazantseva
"""

import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.linear_model import Perceptron

train_data = pd.read_csv('perceptron-train.csv', header = None)
test_data = pd.read_csv('perceptron-test.csv', header = None)

train_predictors = train_data.iloc[:, 1:]
train_target = train_data.iloc[:, 0]

test_predictors = test_data.iloc[:, 1:]
test_target = test_data.iloc[:, 0]

scaler = preprocessing.StandardScaler()
train_predictors_scaled = scaler.fit_transform(train_predictors)
test_predictors_scaled = scaler.transform(test_predictors)

perceptron = Perceptron(random_state = 241)
model = perceptron.fit(train_predictors, train_target)
predictions = model.predict(test_predictors)
quality = metrics.accuracy_score(test_target, predictions)

model_scaled = perceptron.fit(train_predictors_scaled, train_target)
predictions_scaled = model_scaled.predict(test_predictors_scaled)
quality_scaled = metrics.accuracy_score(test_target, predictions_scaled)