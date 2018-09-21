# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 10:58:10 2018

@author: kazantseva
"""

import sklearn.datasets as ds
from sklearn.neighbors import KNeighborsRegressor
from sklearn import model_selection, preprocessing
import numpy as np

boston = ds.load_boston()
predictors = boston.data
target = boston.target

scaler = preprocessing.StandardScaler()
predictors = scaler.fit_transform(predictors)

metrics_p = np.linspace(1, 10, 200)

cv = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 42)
regressor = KNeighborsRegressor(n_neighbors=5, weights='distance')

param_grid = { 'p': metrics_p }
search_res = model_selection.GridSearchCV(regressor, param_grid, scoring = 'neg_mean_squared_error').fit(predictors, target)





