# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 22:49:20 2018

@author: Nataliya
"""

import pandas as pd

data = pd.read_csv('titanic.csv')

def write_answer(file_name, answer):
    with open(file_name, mode = 'w') as f:
        f.write(str(answer))

#write_answer('female_male_count.txt', [data.groupby('Sex').count()['Name'][1], data.groupby('Sex').count()['Name'][0]])
#write_answer('survived_percent.txt', sum(data.Survived) / len(data))
#write_answer('1class_percent.txt', len(data[data['Pclass'] == 1]) / len(data))
#write_answer('age.txt', [data.Age.mean(), data.Age.median()])