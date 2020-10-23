"""
================================================================================
@In Project: ukg2vec
@File Name: confidence.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/08/26
@Update Date: 
@Version: 0.0.1
@Functions: 
    1. To handle confidence data specially for train.tsv and test.tsv files
    2. read from .tsv file, handling confidence data, write to .tsv.txt file
================================================================================
"""
import os


def readydata(dataset, step):
    # dataset is 'ppi5k' or 'cn15k'
    # step is 'one'(without confidence) or 'two'(with confidence)
    base = os.path.abspath("..") + '\\data\\'
    dataset_id = dataset

    train_file = base + dataset_id + '\\train.tsv'
    test_file = base + dataset_id + '\\test.tsv'

    train_with_confidence_file = open(train_file + ".txt", 'w')
    test_with_confidence_file = open(test_file + ".txt", 'w')
    small_test_with_confidence_file = open(test_file + "_small.txt", 'w')

    # 文件中关系和实体都用数字编号，有重复的，故关系ID前加上前缀‘r’
    small_size = 1000  # for quickly testing
    num = 0
    with open(train_file, 'r') as f1, open(test_file, 'r') as f2:
        lines = f1.readlines()
        for line in lines:
            triplet = line.strip().split()
            if step == 'one':
                train_with_confidence_file.write(triplet[0] + '\t' + 'r' + triplet[1] + '\t'
                                              + triplet[2] + '\n')  # for step_1
            if step == 'two':
                train_with_confidence_file.write(triplet[0] + '\t' + 'r' + triplet[1] + '\t'
                                             + triplet[2] + '\t' + triplet[3] + '\n')  # for step_2
        lines = f2.readlines()
        for line in lines:
            num += 1
            triplet = line.strip().split()
            test_with_confidence_file.write(triplet[0] + '\t' + 'r' + triplet[1] + '\t'
                                            + triplet[2] + '\t' + triplet[3] + '\n')
            if num <= small_size:
                small_test_with_confidence_file.write(triplet[0] + '\t' + 'r' + triplet[1] + '\t'
                                                      + triplet[2] + '\t' + triplet[3] + '\n')

    train_with_confidence_file.close()
    test_with_confidence_file.close()
