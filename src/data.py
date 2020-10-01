"""
================================================================================
@In Project: ukg2vec
@File Name: data.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/08/25
@Update Date: 
@Version: 0.0.1
@Functions: 
    1. To preprocess the dataset
    2. 
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt


base = os.path.abspath("..") + '\\data\\'
dataset_id = 'cn15k'  # 'ppi5k' | 'cn15k'

train_file = base + dataset_id + '\\train.tsv'
test_file = base + dataset_id + '\\test.tsv'

entities = list()
relations = list()
train_triples = dict()
test_triples = dict()
repeat_num = 0
with open(train_file, 'r') as f1, open(test_file, 'r') as f2:
    lines = f1.readlines()
    for line in lines:
        triplet = line.strip().split()
        if tuple(triplet) in train_triples.keys():
            # print(triplet)
            repeat_num += 1
        train_triples[tuple(triplet)] = float(triplet[3])  # can eliminate repeat lines
        if triplet[0] not in entities:
            entities.append(triplet[0])
        if triplet[2] not in entities:
            entities.append(triplet[2])
        if triplet[1] not in relations:
            relations.append(triplet[1])
    lines = f2.readlines()
    for line in lines:
        triplet = line.strip().split()
        if tuple(triplet) in test_triples.keys():
            # print(triplet)
            repeat_num += 1
        test_triples[tuple(triplet)] = triplet[3]
print("entities num:", len(entities))
print("relations num:", len(relations))
print("train triples num:", len(train_triples))
print("test triples num:", len(test_triples))
print('all repeat triples num:', repeat_num)
confidence_list = list(train_triples.values())
print('mean of confidence:', np.mean(confidence_list))
print('standard deviation of confidence:', np.std(confidence_list))
