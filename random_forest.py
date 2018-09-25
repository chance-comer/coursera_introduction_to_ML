# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:45:12 2018

@author: kazantseva
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import model_selection

data = pd.read_csv('abalone.csv')

data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'M' else (0 if x == 'I' else -1))

target = data.iloc[:, -1]
predictors = data.iloc[:, :-1]

tree_count = range(1, 50)
n = 5
scores = []

for c in tree_count:
  tree = RandomForestRegressor(n_estimators = c, random_state = 1)
  tree.fit(predictors, target)
  cv = model_selection.KFold(n_splits = n, shuffle = True, random_state=1)
  model_score = model_selection.cross_val_score(tree, predictors, target, scoring = 'r2', cv = cv).mean()
  scores.append(model_score)
  


