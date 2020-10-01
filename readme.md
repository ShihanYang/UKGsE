# Fast and Effective Uncertain Knowledge Graph Embedding
    
    1. data.py - probe the datasets
    2. confidence.py 1st - ready step 1 data without confidence for embedding
    3. embedding.py - embedding triples into vectors (dim = 64,100,128)
    4. confidence.py 2nd - ready step 2 data with confidence for LSTM
    5. nnet.py - training models with different hyper-parameters setting
    6. hitMetrics.py - evaluate the KG without confidence by hits@X
    7. metrics.py - evaluate the UKG predication by MSE and MAE