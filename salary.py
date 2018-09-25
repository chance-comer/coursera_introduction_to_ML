# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:22:20 2018

@author: kazantseva
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import scipy

data_train = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv')

pred_train = data_train.iloc[:, :-1]
target_train = data_train.iloc[:, -1]
pred_test = data_test.iloc[:, :-1]

pred_train['FullDescription'] = pred_train['FullDescription'].apply(lambda x: x.lower())
pred_train['FullDescription'] = pred_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

pred_test['FullDescription'] = pred_test['FullDescription'].apply(lambda x: x.lower())
pred_test['FullDescription'] = pred_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

vectorizer = TfidfVectorizer(min_df = 5)
vectorizer.fit(pred_train['FullDescription'])

p = vectorizer.transform(pred_train['FullDescription'])
v = vectorizer.transform(pred_test['FullDescription'])

pred_train['LocationNormalized'] = pred_train['LocationNormalized'].fillna('nan')
pred_train['ContractTime'] = pred_train['ContractTime'].fillna('nan')

pred_test['LocationNormalized'] = pred_test['LocationNormalized'].fillna('nan')
pred_test['ContractTime'] = pred_test['ContractTime'].fillna('nan')

#dummies = pd.get_dummies(pred_train, prefix = ['loc', 'contract'], columns = ['LocationNormalized', 'ContractTime'])
#dummies = dummies.drop('FullDescription', axis = 1)
#pred_train = pred_train.drop(['LocationNormalized', 'ContractTime'], axis = 1)
#test_dummies = pd.get_dummies(pred_test, prefix = ['loc', 'contract'], columns = ['LocationNormalized', 'ContractTime'])
#test_dummies = test_dummies.drop('FullDescription', axis = 1)
dict_vect = DictVectorizer()
g = dict_vect.fit_transform(pred_train[['LocationNormalized', 'ContractTime']].to_dict('rec'))
g_test = dict_vect.transform(pred_test[['LocationNormalized', 'ContractTime']].to_dict('rec'))

sparsed_train = scipy.sparse.hstack((g, p))

model = linear_model.Ridge(random_state = 241, alpha = 1)
model.fit(sparsed_train, target_train)

sparsed_test = scipy.sparse.hstack((g_test, v))
predictions = model.predict(sparsed_test)