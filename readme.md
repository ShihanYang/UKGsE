# UKG<sub>s</sub>E
These codes implement UKG<sub>s</sub>E model by *pytorch* and *gensim* in the python language, which provides the fast but effective Knowledge Graphs Embedding and more accurate confidence prediction on uncertainty of relation facts in KG. Some kinds of approximate knowledge reasoning can also be done in the Uncertain Knowledge Graph Embedding (UKGE) space. Now we are working on Question-Answering system by means of this model.

## Install
Local environment should be equal to or above as following:

    python 3.6
    pytorch 1.6 (with Theano 1.0.1 backend)
    gensim 3.8.3

## Usage
To run the experiments, use:

    python ./src/ukgse.py
    
or 

    python ./src/ukgse.py --dataset ppi5k --dimension 128 --batchsize 128 --epochs 200
    
## Dataset
Here two experiment datasets, CN15k and PPI5k, are provided in separate folders.

    cn15k:
        train.tsv  # each line likes 'head_id, relation_id, tail_id, confidence value'
        test.tsv  # same as above
        entity_id.csv  # each line likes 'entity_name, entity_id'
        relation_id.csv  # same as above
    ppi5k:
        train.tsv
        test.tsv
        entity_id.csv
        relation_id.csv

## Reference by bibtex

    @inproceedings{yang2020fast,
      title={Fast Confidence Prediction of Uncertainty based on Knowledge Graph Embedding},
      author={Yang, Shihan and Zhang, Weiya and Tang, Rui},
      booktitle={2020 3rd International Conference on Algorithms, Computing and Artificial Intelligence},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3446132.3446186},
      year={2020}
    }
