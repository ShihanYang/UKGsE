"""
================================================================================
@In Project: ukg2vec
@File Name: embedding.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/08/25
@Update Date: 
@Version: 0.0.1
@Functions: 
    1. To generate word vectors for each head, tail and relation
    2. How to handle confidence embedding?
================================================================================
"""

import os
import numpy as np
from gensim.models.word2vec import LineSentence
import multiprocessing
from gensim.models import Word2Vec
import sys
import logging
import time

program = os.path.basename(sys.argv[0])
# print('program:', program)
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.ERROR)

start = time.time()

base = os.path.abspath("..") + '\\data\\ppi5k\\'
corpus = base + 'train.tsv.txt'
total_triples = 249946  # ppi5k
# total_triples = 215451   # cn15k
dim = 128  # the dimension of embedding vectors
# print("args:", sys.argv, corpus, dim)
vectors = corpus + str(dim) + '_sg.w2v'

model = Word2Vec(LineSentence(corpus), size=int(dim), window=2, sample=0,
                 iter=10, negative=5, min_count=1,
                 sg=1, # skip_gram (1) or CBOW (0), seems CBOW more reasonable
                 workers=multiprocessing.cpu_count())

model.init_sims(replace=True)
model.wv.save_word2vec_format(vectors)

time_consumed = time.time() - start

print('Time Consumed(s):', time_consumed)
print('Rate (triples/s):', total_triples/time_consumed)

'''
m = model.most_similar('4176')
print(m)
print(model.similarity('4001', '4176'))  # 4001	r5 4176 [0.329] first item in training set
print(model.similarity('4001', 'r5'))
print(model.similarity('4176', 'r5'))

print(model.most_similar('0'))
print(model.similarity('3810', '4338'))  # 3810	r0 4338 [0.345]  second item in training set
print(model.similarity('3810', 'r0'))

print(model.similarity('4001', '3810'))
print(model.similarity('4001', '4338'))
print(model.similarity('r0', 'r5'))
'''
