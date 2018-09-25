# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:16:28 2018

@author: kazantseva
"""

import pandas as pd
import numpy as np
from sklearn import metrics

def write_answer(name, *args):
  #print(*args)
  with open(name, 'w') as f:
    for i, a in enumerate(args):
      try:
        f.write(str(round(a, 3)))
      except:
        f.write(a)
      if (i < len(args) - 1):
        f.write(' ')
      

data = pd.read_csv('classification.csv')

target = data['true']
predictions = data['pred']

metric_flag = target + predictions + 2 * ( predictions - target)

TP = len(metric_flag[metric_flag == 2]) #43
FP = len(metric_flag[metric_flag == 3]) #34
TN = len(metric_flag[metric_flag == 0]) #64
FN = len(metric_flag[metric_flag == -1]) #59

accuracy = metrics.accuracy_score(target, predictions) #0.535
precision = metrics.precision_score(target, predictions) #0.558
recall = metrics.recall_score(target, predictions) #0.422
f1 = metrics.f1_score(target, predictions) #0.480

#write_answer('tp_tn_fp_fn.txt', TP, FP, FN, TN)
#write_answer('ac_pr_rec_f1.txt', accuracy, precision, recall, f1)

scores = pd.read_csv('scores.csv')

roc_log = metrics.roc_auc_score(scores['true'], scores['score_logreg'])
roc_svm = metrics.roc_auc_score(scores['true'], scores['score_svm'])
roc_knn = metrics.roc_auc_score(scores['true'], scores['score_knn'])
roc_tree = metrics.roc_auc_score(scores['true'], scores['score_tree'])

#write_answer('max_roc.txt', 'score_logreg')
max_prs = {}
for i in range(1, len(scores.columns)):
  precision, recall, treshholds = metrics.precision_recall_curve(scores['true'], scores[scores.columns[i]])
  pr_rec = list(zip(precision, recall))
  pr_rec_more07 = sorted([item for item in pr_rec if item[1] >= 0.7], key = lambda x: -x[0])
  max_prs[scores.columns[i]] = pr_rec_more07[0][0]

write_answer('max_prec.txt', 'score_tree')
