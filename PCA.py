# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:42:23 2018

@author: kazantseva
"""

import pandas as pd
from sklearn.decomposition import pca
import numpy as np

data = pd.read_csv('close_prices.csv')
predictors = data.iloc[:, 1:]

main_comp = pca.PCA(n_components = 10)
model =  main_comp.fit(predictors)
predictors_main = model.transform(predictors)
first_comp = predictors_main[:, 0]

djia_index = pd.read_csv('djia_index.csv')

corr_coef = np.corrcoef(first_comp, djia_index.iloc[:, 1]) #0.90965222