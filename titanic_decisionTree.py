# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:38:01 2018

@author: kazantseva
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

data_na = pd.read_csv('titanic.csv')

data = data_na[['Age', 'Pclass', 'Fare', 'Sex', 'Survived']]
data = data.dropna()
target = data['Survived']
data = data.drop('Survived', axis = 1)

#encoder = LabelEncoder()
#encoder.fit(data['Sex'])
#data['Sex'] = encoder.transform(data['Sex'])
one_hot = pd.get_dummies(data['Sex'], prefix = 'is', drop_first = True)
data = data.drop('Sex', axis = 1)
data = data.join(one_hot)

model = DecisionTreeClassifier(random_state = 241).fit(data, target)
accuracy = metrics.accuracy_score(target, model.predict(data))
fi = model.feature_importances_