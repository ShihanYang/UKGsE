"""
================================================================================
@In Project: ukg2vec
@File Name: pridicationcase.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/10/02
@Update Date: 
@Version: 0.1.0
@Functions: 
    1. To run the case of predication for confidence of triplet
    2. Notes:
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

base = os.path.abspath('..') + '\\data\\cn15k\\'
model = load_model(base+'model_e_128d.h5')
model.summary()
print('Model LOADED.')

embedding_file = base + 'train.tsv.txt128.w2v'
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

entity_id_file = base + 'entity_id.csv'
relation_id_file = base + 'relation_id.csv'
word_id = dict()  # like {rush:2047, relatedto:r0, ...}
id_word = dict()
with open(entity_id_file, 'r') as ef, open(relation_id_file, 'r') as rf:
    r_lines = rf.readlines()
    for r in r_lines:
        rl = r.strip().split(',')
        word_id[rl[0]] = 'r'+rl[1]
        id_word['r'+rl[1]] = rl[0]
    e_lines = ef.readlines()
    for e in e_lines:
        el = e.strip().split(',')
        word_id[el[0]] = el[1]
        id_word[el[1]] = el[0]
print("ID_Dictionary READY.")

original_file = base + 'train.tsv.txt'
original_file_2 = base + 'test.tsv.txt'
triple_with_confidence = dict()
with open(original_file, 'r') as of, open(original_file_2, 'r') as of2:
    r_lines = of.readlines()
    for r in r_lines:
        rl = r.strip().split()
        triple_with_confidence[(rl[0], rl[1], rl[2])] = rl[3]
    r_lines = of2.readlines()
    for r in r_lines:
        rl = r.strip().split()
        triple_with_confidence[(rl[0], rl[1], rl[2])] = rl[3]
print('Original confidence LOADED.')
# print(len(triple_with_confidence))  # there are some replicated records in the training file


# (rush, relatedto, *) fast, motion, rapid, urgency / fast, move, hour, time
# (hotel, usedfor, *) sleeping, rest, bed way from home, stay overnight / relaxing, sleeping, rest, hotel room

# for (rush, relatedto, fast)
triplet = [word_id['rush'], word_id['relatedto'], word_id['fast']]
tv = [[embedding[x] for x in triplet[:3]]]
score = model.predict(np.asarray(tv), verbose=0)[0][0]
print('(rush, relatedto, fast):', triplet)
print('predicted confidence:', score)
# for (rush, relatedto, move)
triplet = [word_id['rush'], word_id['relatedto'], word_id['move']]
tv = [[embedding[x] for x in triplet[:3]]]
score = model.predict(np.asarray(tv), verbose=0)[0][0]
print('(rush, relatedto, move):', triplet)
print('predicted confidence:', score)
# for (rush, relatedto, hour)
triplet = [word_id['rush'], word_id['relatedto'], word_id['hour']]
tv = [[embedding[x] for x in triplet[:3]]]
score = model.predict(np.asarray(tv), verbose=0)[0][0]
print('(rush, relatedto, hour):', triplet)
print('predicted confidence:', score)
# for (rush, relatedto, time)
triplet = [word_id['rush'], word_id['relatedto'], word_id['time']]
tv = [[embedding[x] for x in triplet[:3]]]
score = model.predict(np.asarray(tv), verbose=0)[0][0]
print('(rush, relatedto, time):', triplet)
print('predicted confidence:', score)

# for (hotel, usedfor, sleeping)
triplet = [word_id['hotel'], word_id['usedfor'], word_id['sleeping']]
tv = [[embedding[x] for x in triplet[:3]]]
score = model.predict(np.asarray(tv), verbose=0)[0][0]
print('(hotel, usedfor, sleeping):', triplet)
print('predicted confidence:', score)
# for (hotel, usedfor, relaxing)
triplet = [word_id['hotel'], word_id['usedfor'], word_id['relaxing']]
tv = [[embedding[x] for x in triplet[:3]]]
score = model.predict(np.asarray(tv), verbose=0)[0][0]
print('(hotel, usedfor, relaxing):', triplet)
print('predicted confidence:', score)
# for (hotel, usedfor, rest)
triplet = [word_id['hotel'], word_id['usedfor'], word_id['rest']]
tv = [[embedding[x] for x in triplet[:3]]]
score = model.predict(np.asarray(tv), verbose=0)[0][0]
print('(hotel, usedfor, rest):', triplet)
print('predicted confidence:', score)
# for (hotel, usedfor, hotel room)
triplet = [word_id['hotel'], word_id['usedfor'], word_id['hotel room']]
tv = [[embedding[x] for x in triplet[:3]]]
score = model.predict(np.asarray(tv), verbose=0)[0][0]
print('(hotel, usedfor, hotel room):', triplet)
print('predicted confidence:', score)


# list rank 5 of heigher confidence
head = 'hotel'
relation = 'usedfor'
topk = 2000
candidate = {x+1:('',0) for x in range(topk)}
for tail in word_id.keys():
    if tail == head or 'r' in word_id[tail]:
        continue
    triplet = [word_id[head], word_id[relation], word_id[tail]]
    tv = [[embedding[x] for x in triplet[:3]]]
    score = model.predict(np.asarray(tv), verbose=0)[0][0]
    min_rank = min(candidate, key=lambda x:candidate[x][1])
    if score > candidate[min_rank][1]:
       candidate[min_rank] = (tail, score)
print('Prediction for (HEAD, RELATION, ?tail?) RANKING:')
print('Head = \'', head, '\'')
print('Relation = \'', relation, '\'')
rank = sorted(candidate, key=lambda x:candidate[x][1], reverse=True)
for i in rank:
    triple = (word_id[head], word_id[relation], word_id[candidate[i][0]])
    if triple in triple_with_confidence.keys():
        true_value = triple_with_confidence[triple]
    else:
        true_value = 'N/A'
        continue
    print(candidate[i], '& true value =', true_value)

print('True ranking for (', head, ',', relation, ', *tail*):')
for_ranking = []  # like [(tail, confidence), (-,-), ...]
for i in triple_with_confidence.keys():
    if i[0] == word_id[head] and i[1] == word_id[relation]:
        for_ranking.append((id_word[i[2]], triple_with_confidence[i]))
ranked = sorted(for_ranking, key=lambda x:x[1], reverse=True)
for i in ranked:
    print(i)
