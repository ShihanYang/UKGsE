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


def number_of_triples(dataset):
    base = os.path.abspath("..") + '\\data\\'
    dataset_id = dataset
    file = base + dataset_id + '\\train.tsv.txt'

    entities = list()
    relations = list()
    triple_set = set()
    lines_num = 0
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lines_num += 1
            ll = line.strip().split()
            triple_set.add(tuple(ll))
            if ll[0] not in entities:
                entities.append(ll[0])
            if ll[2] not in entities:
                entities.append(ll[2])
            if ll[1] not in relations:
                relations.append(ll[1])
    e = len(entities)
    r = len(relations)
    triples = len(triple_set)
    print('dataset:', dataset)
    print('data file:', file)
    print('number of lines:', lines_num)
    print('non-redundant triples:', triples)
    print('entities num:', e)
    print('relations num:', r)
    print('total:', e + r)
    return triples  # return the total number of no-repeat triples


if __name__ == '__main__':
    # number_of_triples('cn15k')
    number_of_triples('ppi5k')
    pass

