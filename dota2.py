# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:57:55 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, model_selection, linear_model, preprocessing
import matplotlib.pyplot as plt
import datetime

#1. Считайте таблицу с признаками из файла features.csv с помощью кода, 
#приведенного выше. Удалите признаки, связанные с итогами матча 
#(они помечены в описании данных как отсутствующие в тестовой выборке).

data = pd.read_csv('features.csv', index_col = 0)
test_data = pd.read_csv('features_test.csv', index_col = 0)

# входные данные
X = data.drop(['duration', \
       'radiant_win', 'tower_status_radiant', 'tower_status_dire',\
       'barracks_status_radiant', 'barracks_status_dire'], axis = 1)

# 4. Какой столбец содержит целевую переменную? Запишите его название.
# целевая переменная
y = data['radiant_win']

#2. Проверьте выборку на наличие пропусков с помощью функции count(), 
#которая для каждого столбца показывает число заполненных значений. 
#Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски, 
#и попробуйте для любых двух из них дать обоснование, почему их значения могут быть пропущены.

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
#3. Замените пропуски на нули с помощью функции fillna(). 

# заменяем все пропуски на 0
X = X.fillna(0)
test_data = test_data.fillna(0)

#создаем генератор разбиений
kf = model_selection.KFold(n_splits=5, shuffle=True)

# будем строить деревья в количетсве 5, 10, 15, 20 ... 100
GB_params = {
    'n_estimators' : np.linspace(5, 100, num = 20, dtype = int)
}

GB_estimator = ensemble.GradientBoostingClassifier()
#создаем объект для поиска наилучшего параметра по сетке, заданной переменной GB_params
GB_search_res = model_selection.GridSearchCV(GB_estimator, GB_params, scoring = 'roc_auc', cv = kf)
GB_search_res.fit(X, y)

#список параметров, для которых считалась модель
axe_x = [p['n_estimators'] for p in GB_search_res.cv_results_['params']]
# усредненные значения качества, подсчитанные по кросс-валидации по 5 блокам, для каждого значения параметра n_estimators 
axe_y =  GB_search_res.cv_results_['mean_test_score']
'''
axe_y = [ 0.63466441,  0.66371738,  0.67548929,  0.68172216,  0.68547591,
        0.68861486,  0.6912931 ,  0.69344001,  0.69557539,  0.69702325,
        0.69853186,  0.69991393,  0.70109118,  0.70194074,  0.70296978,
        0.70382924,  0.70453728,  0.70507885,  0.70578016,  0.70646752]

Видно, что качество улучшается с увеличением числа дерьевьев, однако чем больше дерьвьев, тем меньше 
прирост качества. Значит, при 100 дерьевьях мы находимся вблизи оптимума.
'''
'''
качество модели при 30 деревьях равно 0.68861486267337213
время работы алгоритма 58.247599 секунд
'''
GB_estimator = ensemble.GradientBoostingClassifier(n_estimators = 30)
start_time = datetime.datetime.now()
score = model_selection.cross_val_score(GB_estimator, X, y, scoring = 'roc_auc', cv = kf)
duration = datetime.datetime.now() - start_time

'''
#######################################################
#######################################################
###################### Отчет 1 ########################
#######################################################
#######################################################

1. имена столбцов с пропусками: ['first_blood_time', 'first_blood_team', 'first_blood_player1',
       'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time',
       'radiant_flying_courier_time', 'radiant_first_ward_time',
       'dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time',
       'dire_first_ward_time']
Признаки, означающие время приобретения к.-л. предмета (например, radiant_bottle_timeб dire_courier_time), 
купленного за первые пять минут, могут быть не заполнены, если 
команда не покупала этих предметов в течение первых 5 минут игры

2. radiant_win

3. 58.247599 секунд, качество модели при 30 деревьях равно 0.68861486267337213

4. для увеличения скорости можно уменьшить обучающую выборку (выбрать случайным образом из имеющихся 
данных) или уменьшить глубину дерьевьев
'''

'''
#######################################################
#######################################################
################# LOGISTIC REGRESSION #################
#######################################################
#######################################################
'''
scaler = preprocessing.StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X), index = X.index, columns = X.columns)
scaled_test = pd.DataFrame(scaler.transform(test_data), index = test_data.index, columns = test_data.columns)

def LOG_regression(X):
  LOG_search_res = model_selection.GridSearchCV(LOG_estimator, LOG_params, scoring = 'roc_auc', cv = kf)
  LOG_search_res.fit(X, y)
  #список параметров, для которых считалась модель
  axe_x = [p['C'] for p in LOG_search_res.cv_results_['params']]
  #усредненные значения качества, подсчитанные по кросс-валидации по 5 блокам, для каждого значения параметра n_estimators 
  axe_y =  LOG_search_res.cv_results_['mean_test_score']
  #plt.plot(axe_x, axe_y)
  start_time = datetime.datetime.now()
  score = model_selection.cross_val_score(LOG_search_res.best_estimator_, X, y, scoring = 'roc_auc', cv = kf)
  duration = datetime.datetime.now() - start_time
  #predictions = LOG_search_res.best_estimator_.predict(scaled_X)
  return {
      'best_duration': duration,
      'mean_auc': axe_y,
      'estimator': LOG_search_res.best_estimator_,
      'best_params': LOG_search_res.best_estimator_.get_params(),
      'best_score': score
      }
  
LOG_params = {
    'C' : [0.001, 0.01, 0.1, 0.5, 1]
    }

LOG_estimator = linear_model.LogisticRegression(random_state = 0)

'''
лучший парметр регуляризации 0.01
качество модели при этом параметре 0.7165728828933362
время работы алгоритма секунд 12.581055
'''

shortened_X = scaled_X.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r5_hero', \
                      'r3_hero', 'r4_hero', 'd3_hero', 'd4_hero',
                      'd1_hero', 'd2_hero', 'd5_hero'] , axis = 1)


shortened_X_test = scaled_test.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r5_hero', \
                      'r3_hero', 'r4_hero', 'd3_hero', 'd4_hero',
                      'd1_hero', 'd2_hero', 'd5_hero'] , axis = 1)

'''
лучший парметр регуляризации 0.01
качество модели при этом параметре 0.71644213518877553
время работы алгоритма секунд 10.7164
'''

'''
количество униальных значений 108
'''
unique_heroes = np.unique(np.array(X[['r1_hero', 'r2_hero', 'r5_hero', 'r3_hero',\
                      'r4_hero', 'd3_hero', 'd4_hero','d1_hero',\
                      'd2_hero', 'd5_hero']]).flatten(), return_counts = True)

enumerated_heroes = np.vstack((np.arange(0, len(unique_heroes[0])), unique_heroes[0]))
X_pick = np.zeros((data.shape[0], len(unique_heroes[0])))
X_pick_test = np.zeros((test_data.shape[0], len(unique_heroes[0])))

for i, match_id in enumerate(data.index):
    for p in np.arange(1, 6):
        hero_r_id = data.loc[match_id, 'r%d_hero' % p]
        hero_d_id = data.loc[match_id, 'd%d_hero' % p]
        hero_r_id_num = np.where(enumerated_heroes[1] == hero_r_id)[0][0]
        hero_d_id_num = np.where(enumerated_heroes[1] == hero_d_id)[0][0] 
        X_pick[i, hero_r_id_num] = 1
        X_pick[i, hero_d_id_num] = -1

for i, match_id in enumerate(test_data.index):
    for p in np.arange(1, 6):
        hero_r_id = test_data.loc[match_id, 'r%d_hero' % p]
        hero_d_id = test_data.loc[match_id, 'd%d_hero' % p]
        hero_r_id_num = np.where(enumerated_heroes[1] == hero_r_id)[0][0] if len(np.where(enumerated_heroes[1] == 1)[0]) > 0 else -1
        hero_d_id_num = np.where(enumerated_heroes[1] == hero_d_id)[0][0] if len(np.where(enumerated_heroes[1] == 1)[0]) > 0  else -1
        if hero_r_id_num > 0:
          X_pick_test[i, hero_r_id_num] = 1
        if hero_d_id_num > 0:
          X_pick_test[i, hero_d_id_num] = -1


bow_X = np.hstack((np.array(shortened_X), X_pick))
bow_X_test = np.hstack((np.array(shortened_X_test), X_pick_test))

bow_res = LOG_regression(bow_X)
just_scaled_res = LOG_regression(scaled_X)
shortened_X_res = LOG_regression(shortened_X)

#лучшее качество показывает модель, испольщующая "мешок слов". Рассчитаем предсказания данной модели
predictions = bow_res['estimator'].predict_proba(bow_X_test)
predictions = [1 if prediction[1] > 0.6 else 0 for prediction in predictions]
answer = pd.DataFrame()
answer['match_id'] = test_data['match_id']
answer['radiant_win'] = predictions
#answer.to_csv('a.csv', index = False)

'''
#######################################################
#######################################################
###################### Отчет 2 ########################
#######################################################
#######################################################
'''