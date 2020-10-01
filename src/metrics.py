"""
================================================================================
@In Project: ukg2vec
@File Name: metrics.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/08/26
@Update Date: 
@Version: 0.0.1
@Functions: 
    1. To load trained model and predicate the score
    2. then calculate hit@X, X = 1,3,10, and make some statistical graphs
    3. MAE and MSE
================================================================================
"""

import os
import time
import numpy as np
from keras.models import load_model
from tqdm import tqdm


base = os.path.abspath('..') + '\\data\\ppi5k\\'
embedding_file = base + 'train.tsv.txt128_sg.w2v'
model = load_model(base+'model_e80_128d_sg.h5')  # best result recorded by checkpoints
print('model:', model)
model.summary()
print('Model LOADED.')

test_file = base + 'test.tsv.txt'
test_triples = list()
with open(test_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        test_triples.append(ll)
print('Test samples {%d} LOADED.'%(len(test_triples)))

embedding = dict()
dim = 0
with open(embedding_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        if len(ll) == 2:
            dim = int(ll[1])
            continue
        embedding[ll[0]] = ll[1:]
print('Embedding vectors LOADED.')

train_file = base + 'train.tsv.txt'
entities = list()
sentences = dict()  # a sentence like '(h,r)':[tails] for generating negative samples
with open(train_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        if ll[0] not in entities:
            entities.append(ll[0])
        if ll[2] not in entities:
            entities.append(ll[2])
        if (ll[0], ll[1]) not in sentences.keys():
            sentences[(ll[0], ll[1])] = [ll[2]]
        else:
            sentences[(ll[0], ll[1])].append(ll[2])
print('Dictionary LOADED.')
time.sleep(0.1)  # for better show of tqdm progress bar

'''
hits1 = 0
hits3 = 0
hits10 = 0
for triplet in tqdm(test_triples, ncols=80):
    fakes = list()
    tv = [[embedding[x] for x in triplet[:3]]]
    score = model.predict(np.asarray(tv), verbose=0)[0][0]
    # print('real-vale:', triplet[3])
    # print('lstm-score:', score)
    # print('loss:', float(triplet[3]) - score)
    for tail in entities:  # all entities should substitute for tail
        if tail == triplet[2]:
            continue
        if (triplet[0], triplet[1]) in sentences and \
              tail in sentences[(triplet[0], triplet[1])]:
            continue
        fakes.append([embedding[triplet[0]], embedding[triplet[1]], embedding[tail]])
    score_list = model.predict(np.asarray(fakes), verbose=0)
    # print('fakes-shape：', score_list.shape)
    # print('fakes-scores:', np.sort(score_list, axis=0)[:10])
    # Ranking
    rank = 0
    for c in score_list:
        if c > score:
            rank += 1
    if rank <= 10:
        hits10 += 1
        if rank <= 3:
            hits3 += 1
            if rank <= 1:
                hits1 += 1
total_test = len(test_triples)
print('test total num:', total_test)
print("hits@1:", float(hits1) / total_test)
print("hits@3:", float(hits3) / total_test)
print("hits@10:", float(hits10) / total_test)
print('hit@X metrics FINISHED.')
'''

# TODO: outputing final Embedding Vectors
final_vectors_file = base + 'final.w2v'


# STEP: MSE and MAE
se = 0.0
ae = 0.0
for triplet in test_triples:
    tv = [[embedding[x] for x in triplet[:3]]]
    score = model.predict(np.asarray(tv), verbose=0)[0][0]
    loss = float(triplet[3]) - score
    # print('real-vale:', triplet[3])
    # print('lstm-score:', score)
    # print('loss:', loss)
    # print('loss squre:', loss * loss)
    se += np.square(loss)
    ae += np.abs(loss)
mse = se / len(test_triples)
mae = ae / len(test_triples)
print('MSE:', mse)
print('MAE:', mae)
print('MSE and MAE FINISHED.')

# TODO: link predication

