# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 09:28:35 2018

@author: kazantseva
"""

import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('wine.data', header = None, names = ['Class',
                                                        'Alcohol', 
                                                        'Malic acid', 
                                                        'Ash', 
                                                        'Alcalinity of ash', 
                                                        'Magnesium',
                                                        'Total phenols',
                                                        'Flavanoids',
                                                        'Nonflavanoid phenols',
                                                        'Proanthocyanins',
                                                        'Color intensity',
                                                        'Hue',
                                                        'OD280/OD315 of diluted wines',
                                                        'Proline' ])

predictors = data.iloc[:, 1:]
target = data.iloc[:, 0]

cv = model_selection.KFold(random_state = 42, shuffle = True, n_splits = 5)

n_neighbors = range(1, 51)
mse = []

for n in n_neighbors:
  knn = KNeighborsClassifier(n_neighbors = n)
  mse.append(model_selection.cross_val_score(knn, predictors, target, cv = cv).mean())

plt.figure(figsize = (10, 5))
plt.subplot(1,2,1)
plt.plot(n_neighbors, mse)

answer1_2 = max(np.vstack((n_neighbors, mse)).T, key = lambda x : x[1])
#array([ 4.        ,  0.65777778])

scaler = preprocessing.StandardScaler()
predictors_scaled = scaler.fit_transform(predictors)

mse_scaled = []

for n in n_neighbors:
  knn = KNeighborsClassifier(n_neighbors = n)
  mse_scaled.append(model_selection.cross_val_score(knn, predictors_scaled, target, cv = cv).mean())

plt.subplot(1,2,2)
plt.plot(n_neighbors, mse_scaled)
answer3_4 = max(np.vstack((n_neighbors, mse_scaled)).T, key = lambda x : x[1])
#array([ 2.        ,  0.93285714])