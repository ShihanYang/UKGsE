"""
================================================================================
@In Project: ukg2vec
@File Name: count.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/08/26
@Update Date: 
@Version: 0.0.1
@Functions: 
    1. To count number of entities and relations in file
    2. 
================================================================================
"""

import os


base = os.path.abspath("..") + '\\data\\'
dataset_id = 'ppi5k'
file = base + dataset_id + '\\train.tsv.txt'


entities = list()
relations = list()
lines_num = 0
with open(file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        lines_num += 1
        ll = line.strip().split()
        if ll[0] not in entities:
            entities.append(ll[0])
        if ll[2] not in entities:
            entities.append(ll[2])
        if ll[1] not in relations:
            relations.append(ll[1])
        if lines_num > 10000:  # count top 10000 lines
            break
f.close()
e=len(entities)
r=len(relations)
print('file name:', file)
print('entities num:', e)
print('relations num:', r)
print('total:', e+r)
