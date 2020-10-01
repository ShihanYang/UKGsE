"""
================================================================================
@In Project: ukg2vec
@File Name: hitsMetrics.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/09/09
@Update Date: 
@Version: 0.0.1
@Functions: 
    1. To calculate hits@X of facts without confidence
    2. 
================================================================================
"""

import os
import time
import numpy as np
from tqdm import tqdm


base = os.path.abspath("..") + '\\data\\cn15k\\'
train_file = base + 'train.tsv.txt'
test_file = base + 'test.tsv.txt'
vectors = base + 'train.tsv.txt128.w2v'

embedding = dict()
entities = list()
dim = 0
with open(vectors, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        if len(ll) == 2:
            dim = int(ll[1])
            continue
        embedding[ll[0]] = list(map(float, ll[1:]))
        if ll[0][0] != 'r' and ll[0] not in entities:
            entities.append(ll[0])
print("embedding vectors:", len(embedding), "with dim =", dim)
print("Embeddings loaded.")

test_triples = list()
with open(test_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        test_triples.append(ll)
print('Testing samples loaded.')

sentences = dict()  # a sentence like '(h,r)':[tails] for measuring hits@X
with open(train_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        if (ll[0], ll[1]) not in sentences.keys():
            sentences[(ll[0], ll[1])] = [ll[2]]
        else:
            sentences[(ll[0], ll[1])].append(ll[2])
print('Sentences ready.')

print("Metrics by means of hits@X ... ...")
hits1 = 0
hits3 = 0
hits10 = 0
time.sleep(0.1)
for triplet in tqdm(test_triples, ncols=80):
    fakes = list()
    tv = [[embedding[x] for x in triplet[:3]]]
    h_add_r = np.array(tv[0][0]) + np.array(tv[0][1])
    t = np.array(tv[0][2])
    distance = np.sqrt(np.sum(np.square(h_add_r - t)))

    for tail in entities:  # all entities should substitute for tail
        if tail == triplet[2]:
            continue
        if (triplet[0], triplet[1]) in sentences and \
              tail in sentences[(triplet[0], triplet[1])]:
            continue
        fakes.append([embedding[triplet[0]], embedding[triplet[1]], embedding[tail]])
    score_list = [np.sqrt(np.sum(np.square(np.array(i[0]) + np.array(i[1]) - np.array(i[2])))) for i in fakes]

    # Ranking
    rank = 0
    for c in score_list:
        if c < distance:
            rank += 1
    if rank <= 10:
        hits10 += 1
        if rank <= 3:
            hits3 += 1
            if rank <= 1:
                hits1 += 1

total_test_samples = len(test_triples)
print('For {%d} test samples (relation facts without confidence):' % (total_test_samples))
print("hits@1:", float(hits1) / total_test_samples)
print("hits@3:", float(hits3) / total_test_samples)
print("hits@10:", float(hits10) / total_test_samples)
