Software Architecture
------------------------
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
