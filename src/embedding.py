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
from gensim.models.word2vec import LineSentence
import multiprocessing
from gensim.models import Word2Vec
import sys
import logging
import time
from src.count import number_of_triples


def pre_training(dataset, dimension):

    program = os.path.basename(sys.argv[0])
    logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.ERROR)

    base = os.path.abspath("..") + '\\data\\'+dataset+'\\'
    corpus = base + 'train.tsv.txt'
    total_triples = number_of_triples(dataset)  # number of no-repeat triples in training dataset
    dim = dimension
    vectors = corpus + str(dim) + '_sg.w2v'

    start = time.time()

    model = Word2Vec(LineSentence(corpus), size=int(dim), window=2, sample=0,
                     iter=10, negative=5, min_count=1,
                     sg=1,  # skip_gram (1) or CBOW (0)
                     workers=multiprocessing.cpu_count())

    model.init_sims(replace=True)
    model.wv.save_word2vec_format(vectors)

    time_consumed = time.time() - start

    print('Time Consumed(s):', time_consumed)
    print('Rate (triples/s):', total_triples/time_consumed)


if __name__ == '__main__':
    pre_training('ppi5k', 128)
