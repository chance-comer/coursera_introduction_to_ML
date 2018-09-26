# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 09:39:56 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import model_selection, metrics
import matplotlib.pyplot as plt

data = pd.read_csv('gbm-data.csv')

data_ar = np.array(data)

X_train, X_test, y_train, y_test = model_selection.train_test_split(data_ar[:, 1:], data_ar[:, 0], random_state = 241, test_size = 0.8)

learning_rate = [1, 0.5, 0.3, 0.2, 0.1]

res = {}

plt.figure(figsize = (6, 30))

for i, lr in enumerate(learning_rate):
  classifier = GradientBoostingClassifier(learning_rate = lr, n_estimators = 250, random_state = 241, verbose = True)
  classifier.fit(X_train, y_train)
  train_staged_decision = classifier.staged_decision_function(X_train)
  test_staged_decision = classifier.staged_decision_function(X_test)
  sigmoid_train = [1 / (1 + np.exp( -y_pred)) for y_pred in train_staged_decision]
  sigmoid_test = [1 / (1 + np.exp( -y_pred)) for y_pred in test_staged_decision]
  predictions_train = classifier.predict_proba(X_train)
  predictions_test = classifier.predict_proba(X_test)
  log_loss_train = [metrics.log_loss(y_train, iteration_pred) for iteration_pred in sigmoid_train]
  log_loss_test = [metrics.log_loss(y_test, iteration_pred) for iteration_pred in sigmoid_test]

  res['learning_rate_' + str(lr)] = { 'train_min_val_arg' : np.argmin(log_loss_train), \
              'train_min_val' : np.min(log_loss_train), \
              'test_min_val_arg' : np.argmin(log_loss_test), \
              'test_min_val' : np.min(log_loss_test)}
 
  plt.subplot(5, 1, i + 1)
  plt.plot(range(250), log_loss_train, label = 'train')
  plt.plot(range(250), log_loss_test, label = 'test')
  plt.legend()
  plt.title('learning_rate = ' + str(lr))

plt.tight_layout()
#predictions = classifier.predict_proba(X_test)
rf = RandomForestClassifier(random_state = 241, n_estimators = 36)
rf.fit(X_train, y_train)
preds = rf.predict_proba(X_test)
loss = metrics.log_loss(y_test, preds)
#for lr in learning_rate:
  