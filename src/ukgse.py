"""
================================================================================
@In Project: ukg2vec
@File Name: ukgse.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/10/22
@Update Date: 
@Version: 0.1.0
@Functions: 
    1. This is the main function of the ukgse
    2. Notes: a run command example as following
          python ukgse.py -ds cn15k -dim 128 -batch 128 -epo 200
================================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # only ERROR messages
import argparse

parser = argparse.ArgumentParser(description='Predict confidence of Uncertain Knowledge Graph Embedding')
parser.add_argument('-ds', '--dataset', type=str, default='ppi5k', help='testing dataset is PPI5k or CN15K')
parser.add_argument('-s', '--step', type=str, default='one', help='there are two steps: one or two')
parser.add_argument('-dim', '--dimension', type=int, default=128, help='the dimension of embedding vectors')
parser.add_argument('-batch', '--batchsize', type=int, default=128, help='the batch size when training neural networks')
parser.add_argument('-epo', '--epochs', type=int, default=200, help='the maximus epochs of training')
args = parser.parse_args()

if __name__ == '__main__':
    pass