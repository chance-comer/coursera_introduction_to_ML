# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 22:49:20 2018

@author: Nataliya
"""

import pandas as pd
import numpy as np
import seaborn as sns
import re
import collections as cls

data = pd.read_csv('titanic.csv')

def write_answer(file_name, answer):
    with open(file_name, mode = 'w') as f:
        f.write(str(answer))

#write_answer('female_male_count.txt', [data.groupby('Sex').count()['Name'][1], data.groupby('Sex').count()['Name'][0]])
#write_answer('survived_percent.txt', sum(data.Survived) / len(data))
#write_answer('1class_percent.txt', len(data[data['Pclass'] == 1]) / len(data))
#write_answer('age.txt', [data.Age.mean(), data.Age.median()])
#write_answer('pirson_correl.txt', np.corrcoef(data['SibSp'], data['Parch'])[0][1])

cnt = cls.Counter()

for name in data[data['Sex'] == 'female']['Name']:
  words = re.search('(Mrs. [\w\W]+\(([\w\W]+)\))|(Mrs. \(([\w\W]+)\))|(Mrs.|Miss.|Mme.|Ms.|Lady.|Mlle.|Countess.|Dr. ([\w\W]+))', name)
  groups = []
  for i in [2, 4, 6]:
    if words.group(i):
      groups.append(words.group(i))
  for grp in groups:
    name_parts = grp.split(sep = ' ')
    for name_part in name_parts:
      cnt[name_part.strip('()"')] += 1
  