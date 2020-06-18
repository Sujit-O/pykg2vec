Introduction
===============

In recent years, Knowledge Graph Embedding (KGE) methods have been applied in benchmark
datasets including Wikidata_, Freebase_, DBpedia_,
and YAGO_. Applications of KGE methods include fact prediction, question answering, and recommender systems.

KGE is an active area of research and many authors have provided reference software implementations.
However, most of these are standalone reference implementations and therefore it is difficult and
time-consuming to:

(i) find the source code
(ii) adapt the source code to new datasets
(iii) correctly parameterize the models
(iv) compare against other methods

Recently, this problem has been partially addressed by libraries such as OpenKE_ and AmpliGraph_ that provide a
framework common to several KGE methods. However, these frameworks take different perspectives, make specific
assumptions, and thus the resulting implementations diverge substantially from the original architectures.
Furthermore, these libraries often force the user to use preset hyperparameters, or make implicit use of
golden hyperparameters, and thus make it tedious and time-consuming to adapt the models to new datasets.

**To solve these issues we propose pykg2vec which is a single Python library with large collection of state-of-the-art
KGE methods. The goals of pykg2vec are to be practical and educational.** The practical value is achieved through:

(a) proper use of GPUs and CPUs
(b) a set of tools to automate the discovery of golden hyperparameters
(c) a set of visualization tools for the training and results of the embeddings

The educational value is achieved through:

(d) a modular and flexible software architecture and KGE pipeline
(e) access to a large number of state-of-the-art KGE models


Why this Package?
#################

As mentioned earlier, the sole purpose of pykg2vec library can be summarized in two fold:

**Educational**: The educational value is achieved through:

(a) a modular and flexible software architecture and KGE pipeline
(b) access to a large number of state-of-the-art KGE models

**Practical**: The practical value is achieved through:

(a) proper use of GPUs and CPUs
(b) a set of tools to automate the discovery of golden hyperparameters
(c) a set of visualization tools for the training and results of the embeddings


What this Package Includes?
###########################
The main contributions of this package are:

1) ``State-of-the-art Algorithms`` - A sheer amount of existing state-of-the-art knowledge graph embedding algorithms (such as TransE, TransH, TransR, TransD, TransM, KG2E, RESCAL, DistMult, ComplEX, ConvE, ProjE, RotatE, SME, SLM, NTN, TuckER) is presented in the library.
2) ``Bayesian hyper-parameter tuning`` - Module that supports automatic hyperparameter tuning using bayesian optimization.
3) ``Optimized Implementation`` - Optimized performance by making a proper use of CPUs and GPUs (multiprocess and Tensorflow).
4) ``Visualization and Summary`` - A suite of visualization and summary tools.

Software Architecture
#####################
.. image:: ../../figures/pykg2vec_structure.png
   :width: 600
   :align: center
   :alt: Structure of the pykg2vec library.

The pykg2vec library is built using Python and TensorFlow. TensorFlow allows the
computations to be assigned on both GPU and CPU. In addition to the main model training process,
pykg2vec utilizes multi-processing for generating mini-batches and performing an evaluation to reduce
the total execution time. The various components of the library are as follows:

1) ``KG Controller`` - handles all the low-level parsing tasks such as finding the total uniqueset of entities and relations; creating ordinal encoding maps; generating training, testingand validation triples; and caching the dataset data on disk to optimize tasks that involverepetitive model testing.
2) ``Batch Generator`` - consists of multiple concurrent processes that manipulate and createmini-batches of data.  These mini-batches are pushed to a queue to be processed by themodels  implemented  in  TensorFlow.   The  batch  generator  runs  independently  so  thatthere is a low latency for feeding the data to the training module running on the GPU.
3) ``Core Models`` - consists of large number of state-of-the-art KGE algorithms implemented as Python modules in Tensor-Flow.  Each module consists of a modular description of the inputs, outputs, loss function,and embedding operations.  Each model is provided with configuration files that defineits hyperparameters.
4) ``Configuration`` - rovides the necessary configuration to parse the datasets and also consistsof  the  baseline  hyperparameters  for  the  KGE  algorithms  as  presented  in  the  originalresearch papers.
5) ``Trainer and Evaluator`` - the Trainer module is responsible for taking an instance of theKGE  model,  the  respective  hyperparameter  configuration,  and  input  from  the  batchgenerator to train the algorithms.  The Evaluator module performs link prediction andprovides the respective accuracy in terms of mean ranks and filtered mean ranks.
6) ``Visualization`` - plots training loss and common metrics used in KGE tasks.  To facilitatemodel analysis,  it also visualizes the latent representations of entities and relations onthe 2D plane using t-SNE based dimensionality reduction.
7) ``Bayesian Optimizer`` - pykg2vec uses a Bayesian hyperparameter optimizer to find a goldenhyperparameter set.  This feature is more efficient than brute-force based approaches.

.. _Wikidata: https://cacm.acm.org/magazines/2014/10/178785-wikidata/fulltext
.. _Freebase: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.538.7139&rep=rep1&type=pdf
.. _DBpedia: https://cis.upenn.edu/~zives/research/dbpedia.pdf
.. _YAGO: https://www2007.org/papers/paper391.pdf
.. _OpenKE: https://github.com/thunlp/OpenKE
.. _AmpliGraph: https://github.com/Accenture/AmpliGraph