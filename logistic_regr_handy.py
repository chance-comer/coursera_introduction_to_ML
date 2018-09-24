# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:18:05 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
from sklearn import metrics

data = pd.read_csv('data-logistic.csv', header = None)

target = np.array(data.iloc[:, 0], ndmin = 2)
predictors = np.array(data.iloc[:, 1:])

ww = np.array([0, 0], ndmin = 2)
w_reg_free = np.array([0, 0], ndmin = 2)

k = 0.1
eps = 10 ** -5
l2_coef = 10
n_iter = 0

def grad(ww, k):
  return ww + 1 / len(target[0]) * k * np.array(np.sum(target.T * predictors * (1 - 1 / (1 + np.exp(- target * (ww.dot(predictors.T))))).T, axis = 0), ndmin = 2) - k * l2_coef * ww

def grad_reg_free(ww, k):
  return ww + 1 / len(target[0]) * k * np.array(np.sum(target.T * predictors * (1 - 1 / (1 + np.exp(- target * (ww.dot(predictors.T))))).T, axis = 0), ndmin = 2)

while n_iter < 10000:
  n_iter += 1
  new_w = grad(ww, k)
  if (np.linalg.norm(new_w - ww)) < eps:
    ww = new_w
    break
  else:
    ww = new_w

n_iter = 0

while n_iter < 10000:
  n_iter += 1
  new_w = grad_reg_free(w_reg_free, k)
  if (np.linalg.norm(new_w - w_reg_free)) < eps:
    w_reg_free = new_w
    break
  else:
    w_reg_free = new_w

predictions_w = 1 / (1 + np.exp(- ww.dot(predictors.T)))
predictions_w_reg_free = 1 / (1 + np.exp(- w_reg_free.dot(predictors.T)))

target_bin = np.where(target < 0, 0, 1)

roc_auc = metrics.roc_auc_score(target.reshape(205), predictions_w.reshape(205))
roc_auc_reg_free = metrics.roc_auc_score(target.reshape(205), predictions_w_reg_free.reshape(205))

w1 = 0
w2 = 0
opt_w1 = 0
opt_w2 = 0
n_iter_1 = 0
target_1 = np.array(data.iloc[:, 0])

while n_iter_1 < 10000:
  n_iter_1 += 1
  sum_w1 = 0
  sum_w2 = 0
  for i in range(len(target_1)):
    sum_w1 += target_1[i] * predictors[i][0] * (1 - 1 / (1 + np.exp(- target_1[i] * (w1 * predictors[i][0] + w2 * predictors[i][1]))))
    sum_w2 += target_1[i] * predictors[i][1] * (1 - 1 / (1 + np.exp(- target_1[i] * (w1 * predictors[i][0] + w2 * predictors[i][1]))))
  opt_w1 = w1 + k * 1 / len(target_1) * sum_w1 - k * l2_coef * w1
  opt_w2 = w2 + k * 1 / len(target_1) * sum_w2 - k * l2_coef * w2
  w_vec = np.array([w1, w2])
  opt_w_vec = np.array([opt_w1, opt_w2])
  if (np.linalg.norm(w_vec - opt_w_vec)) <= eps:
    w1 = opt_w1
    w2 = opt_w2
    break
  else:
    w1 = opt_w1
    w2 = opt_w2

a = np.array([100])
b = a
a = np.array([200])

w1 = 0
w2 = 0
sum_w1 = 0
sum_w2 = 0

for i in range(len(target_1)):
    sum_w1 += target_1[i] * predictors[i][0] * (1 - 1 / (1 + np.exp(- target_1[i] * (w1 * predictors[i][0] + w2 * predictors[i][1]))))
    sum_w2 += target_1[i] * predictors[i][1] * (1 - 1 / (1 + np.exp(- target_1[i] * (w1 * predictors[i][0] + w2 * predictors[i][1]))))
    #print(sum_w1, sum_w2)
opt_w1 = w1 + k * 1 / len(target_1) * sum_w1 - k * l2_coef * w1
opt_w2 = w2 + k * 1 / len(target_1) * sum_w2 - k * l2_coef * w2
#predictions_w_1 = 1 / (1 + np.exp(- w1 * predictors[:, 0] - w2 * predictors[:, 1]))
#predictions_w_reg_free = 1 / (1 + np.exp(- w1 * predictors[i][0] + w2 * predictors[i][1]))

#roc_auc = metrics.roc_auc_score(target.reshape(205), predictions_w_1.reshape(205))
#roc_auc_reg_free = metrics.roc_auc_score(target_bin.reshape(205), predictions_w_reg_free.reshape(205))