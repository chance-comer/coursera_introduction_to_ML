# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:53:15 2018

@author: kazantseva
"""

import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data = pd.read_csv('svm-data.csv', header = None)

predictors = data.iloc[:, 1:]
target = data.iloc[:, 0]

model = SVC(C = 100000, random_state = 241).fit(predictors, target)

plt.scatter(predictors.iloc[:, 0], predictors.iloc[:, 1], c = target)

for i, txt in enumerate(range(10)):
    plt.annotate(txt, (predictors.iloc[i, 0], predictors.iloc[i, 1]))