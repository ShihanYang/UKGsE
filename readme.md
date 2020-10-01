## UKGE by Word2Vec and LSTM

* train.tsv, 包含confidence值，把confidence分开，单独训练triples，因为confidence值不需要embedding。
* Word2Vec() 参数    
  gensim.models.word2vec.Word2Vec(sentences=None,size=100,alpha=0.025,window=5, min_count=5, max_vocab_size=None, sample=0.001,seed=1, workers=3,min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>,iter=5,null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)
    1. sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。对于大语料集，建议使用BrownCorpus,Text8Corpus或lineSentence构建。
    2. size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，视语料库的大小而定。
    3. alpha： 是初始的学习速率，在训练过程中会线性地递减到min_alpha。
    4. window：即词向量上下文最大距离，skip-gram和cbow算法是基于滑动窗口来做预测。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。对于一般的语料这个值推荐在[5,10]之间。
    5. min_count:：可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
    6. max_vocab_size: 设置词向量构建期间的RAM限制，设置成None则没有限制。
    7. sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
    8. seed：用于随机数发生器。与初始化词向量有关。
    9. workers：用于控制训练的并行数。
    10. min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。**对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值**。
    11. sg: 即我们的word2vec两个模型的选择了。如果是0，则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。
    12. hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
    13. negative:如果大于零，则会采用negativesampling，用于设置多少个noise words（一般是5-20）。
    14. cbow_mean: 仅用于CBOW在做投影的时候，为0，则采用上下文的词向量之和，为1则为上下文的词向量的平均值。默认值也是1,不推荐修改默认值。
    15. hashfxn： hash函数来初始化权重，默认使用python的hash函数。
    16. iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
    17. trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）。
    18. sorted_vocab： 如果为1（默认），则在分配word index 的时候会先对单词基于频率降序排序。
    19. batch_words：每一批的传递给线程的单词的数量，默认为10000。
