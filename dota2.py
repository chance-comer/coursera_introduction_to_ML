# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:57:55 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, model_selection
import matplotlib.pyplot as plt

data = pd.read_csv('features.csv')

# входные данные
X = data.drop(['duration', \
       'radiant_win', 'tower_status_radiant', 'tower_status_dire',\
       'barracks_status_radiant', 'barracks_status_dire'], axis = 1)

# целевая переменная
y = data['radiant_win']

#количество заполненных значений в столбцах с неполными данными
notfull_filled_count = X.count()[X.count() < X.shape[0]]
#количество пропусков в каждом столбце
gap_count = map(lambda x : X.shape[0] - x , notfull_filled_count)

'''
имена столбцов с пропусками: ['first_blood_time', 'first_blood_team', 'first_blood_player1',
       'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time',
       'radiant_flying_courier_time', 'radiant_first_ward_time',
       'dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time',
       'dire_first_ward_time']
Признаки, означающие время приобретения к.-л. предмета (например, radiant_bottle_timeб dire_courier_time), 
купленного за первые пять минут, могут быть не заполнены, если 
команда не покупала этих предметов в течение первых 5 минут игры
'''

# заменяем все пропуски на 0
X = X.fillna(0)

#создаем генератор разбиений
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
#cv_results_
GB_params = {
    'n_estimators' : np.linspace(5, 100, num = 20, dtype = int)
}
GB_estimator = ensemble.GradientBoostingClassifier()
GB_search_res = model_selection.GridSearchCV(GB_estimator, GB_params, scoring = 'roc_auc', cv = kf)
GB_search_res.fit(X, y)
axe_x = [p['n_estimators'] for p in GB_search_res.cv_results_['params']]
axe_y =  GB_search_res.cv_results_['mean_test_score']
#score = model_selection.cross_val_score(GB_estimator, X, y, scoring = 'roc_auc', cv = kf)