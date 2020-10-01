"""
================================================================================
@In Project: ukg2vec
@File Name: nnet.py
@Author: Shihan Yang
@E-Mail: dr.yangsh@kust.edu.cn
@Create Date: 2020/08/25
@Update Date: 
@Version: 0.0.1
@Functions: 
    1. To implement LSTM neural network for training word vectors again
    2. sampling, modelling, fitting, predicating, evaluating by ranking
================================================================================
"""

import os
import time
import numpy as np
from keras import metrics, callbacks
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from src import pickleloss
from src.checkpoint import checkpoint_base, ukgeCheckpoint, get_last_status

base = os.path.abspath('..') + '\\data\\ppi5k\\'
embedding_file = base + 'train.tsv.txt100_sg.w2v'
train_file = base + 'train.tsv.txt'
test_file = base + 'test.tsv.txt'
model_file = base + 'model_e200_100d_sg.h5'
checkpoint_dir = base + 'checkpoints\\'
checkpoint_base(checkpoint_dir)
checkpoint_file = checkpoint_dir + os.path.basename(model_file) + '-loss.chk'
loss_file = base + os.path.basename(model_file) + '.loss'

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
print("embedding vectors:", len(embedding), "with dim =", dim)
print("Embeddings loaded.")

X_train = list()
y_train = list()
entities = list()
relations = list()
sentences = dict()  # a sentence like '(h,r)':[tails] for generating negative samples
with open(train_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        if ll[0] not in entities:
            entities.append(ll[0])
        if ll[2] not in entities:
            entities.append(ll[2])
        if ll[1] not in relations:
            relations.append(ll[1])
        if (ll[0], ll[1]) not in sentences.keys():
            sentences[(ll[0], ll[1])] = [ll[2]]
        else:
            sentences[(ll[0], ll[1])].append(ll[2])
        vectors = [embedding[x] for x in ll[:3]]
        X_train.append(vectors)
        y_train.append(ll[3])
print("X_train without neg samples:", len(X_train))
print("y_train without neg samples:", len(y_train))
print('Entities num:', len(entities))
print('Relations num:', len(relations))
print('Sentences num:', len(sentences))
valid_sum = 0
for i in sentences:
    valid_sum += len(sentences[i])
assert valid_sum == len(y_train)
print('Valid total num:', valid_sum)
print("Training positive samples loaded.")

# negative samples by corrupting strategy
neg_num = 0
for i in sentences:
    for j in sentences[i]:
        h = i[0]
        r = i[1]
        while True:
            neg_tail_index = np.random.randint(0, len(entities))
            if entities[neg_tail_index] not in sentences[i]:
                t = entities[neg_tail_index]
                hv = embedding[h]
                rv = embedding[r]
                tv = embedding[t]
                X_train.append([hv, rv, tv])
                y_train.append(1e-08)
                neg_num += 1
                break
print('Negative samples num:', neg_num)
print('X_train num:', len(X_train))
print('y_train num:', len(y_train))
print('Train samples are ready!')

X_test = list()
y_test = list()
test_triples = list()
with open(test_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        ll = line.strip().split()
        test_triples.append(ll)
        vectors = [embedding[x] for x in ll[:3]]
        X_test.append(vectors)
        y_test.append(ll[3])
print('X_test:', len(X_test))
print('y_test:', len(y_test))
print('Testing positive samples loaded.')

# STEP: LSTM modelling
start = time.time()
model = Sequential(name='fast_ukge')
lstm = LSTM(dim, input_shape=(3, dim))
model.add(lstm)
model.add(Dense(1, activation='sigmoid'))
model.summary()
loss = 'mean_squared_error'
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
model.compile(loss=loss, optimizer=optimizer, metrics=[metrics.mae]) # mae and mse(=loss assigned above)
print('Begin training .... ...')
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)

# callback = callbacks.EarlyStopping(monitor='loss', mode='min', # (val_)loss or (val_)mean_absolute_error
#                                    min_delta=1e-05, patience=3, verbose=1)
callback = callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                   min_delta=1e-04, patience=10, verbose=1)

last_epoch, last_meta = get_last_status(model, checkpoint_file)
checkpoint = ukgeCheckpoint(checkpoint_file, monitor='val_loss',
                            save_weights_only=True,  # all data or only weights data
                            save_best_only=True,  # best data or latest data
                            verbose=1, meta=last_meta)
# shuffle training dataset first, needed for better val_loss decreasing steadily
permutation = np.random.permutation(X_train.shape[0])
shuffled_X_train = X_train[permutation, :, :]
shuffled_y_train = y_train[permutation]

history = model.fit(shuffled_X_train, shuffled_y_train, validation_split=0.1111,  # VIP
                    epochs=3, batch_size=128, shuffle=True, verbose=2,  # bs = 64, 128
                    callbacks=[  # callback,   # early stopping
                               checkpoint],    # check points
                    initial_epoch=last_epoch + 1)

model.save(model_file)
print('Train FINISHED.')
time_consumed = time.time() - start
print('Time consumed(s):', time_consumed)

print('Evaluate results:\n', 'MSE, MAE =',
      model.evaluate(np.asarray(X_test), np.asarray(y_test), batch_size=64, verbose=0))

# visualizing loss and val_loss (MSE, MAE)
history = history.history
pickleloss.save(history, loss_file)
plt.plot(history['loss'])  # mse
plt.plot(history['val_loss'])  # val_mse
plt.plot(history['mean_absolute_error'])  # mae
plt.plot(history['val_mean_absolute_error'])  # val_mae
plt.title('Losses of LSTM Training')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['MSE', 'val_MSE', 'MAE', 'val_MAE'], loc='lower right')
plt.show()
