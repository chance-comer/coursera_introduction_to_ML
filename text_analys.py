# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 10:11:25 2018

@author: Nataliya
"""

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import model_selection
from sklearn.svm import SVC

data = datasets.fetch_20newsgroups(subset = 'all', categories = ['alt.atheism', 'sci.space'])

vectorizer = TfidfVectorizer()
vectorized_text = vectorizer.fit_transform(data.data)
feature_mapping = vectorizer.get_feature_names()

text = ['i dont wanna know', 'about the things we\'ve come through']
vectorizer_2 = TfidfVectorizer()
vectorized_text_2 = vectorizer_2.fit_transform(text)

C = np.geomspace(10**-5, 10**5, num = 11)#[10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 100, 1000, 10000, 100000]

cv = model_selection.KFold(random_state = 241, n_splits = 5, shuffle = True)

estimator = SVC(kernel = 'linear', random_state = 241)
#search_res = model_selection.GridSearchCV(estimator, { 'C' : C }, cv = cv, scoring = 'accuracy').fit(vectorized_text, data.target)

estimator = SVC(kernel = 'linear', random_state = 241, C = 1)
model = estimator.fit(vectorized_text, data.target)

inds = model.coef_.indices[np.abs(model.coef_.data).argsort()[-10:]] 

print(sorted(feature_mapping[i] for i in inds))