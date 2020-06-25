Software Architecture and API Documentation
###########################################
.. image:: ../../figures/pykg2vec_structure.png
   :width: 600
   :align: center
   :alt: Structure of the pykg2vec library.

The pykg2vec is built using Python and PyTorch. It allows the computations to be assigned on both GPU and CPU. In addition to the main model training process, pykg2vec utilizes multi-processing for generating mini-batches and performing an evaluation to reduce
the total execution time. The various components of the library are as follows:

1) ``KG Controller`` - handles all the low-level parsing tasks such as finding the total unique set of entities and relations; creating ordinal encoding maps; generating training, testing and validation triples; and caching the dataset data on disk to optimize tasks that involve repetitive model testing.
2) ``Batch Generator`` - consists of multiple concurrent processes that manipulate and create mini-batches of data.  These mini-batches are pushed to a queue to be processed by the models implemented in PyTorch or TensorFlow. The batch generator runs independently so that there is a low latency for feeding the data to the training module running on the GPU.
3) ``Core Models`` - consists of large number of state-of-the-art KGE algorithms implemented as Python modules in PyTorch and TensorFlow.  Each module consists of a modular description of the inputs, outputs, loss function,and embedding operations. Each model is provided with configuration files that define its hyperparameters.
4) ``Configuration`` - provides the necessary configuration to parse the datasets and also consists of the baseline hyperparameters for the KGE algorithms as presented in the original research papers.
5) ``Trainer and Evaluator`` - the Trainer module is responsible for taking an instance of the KGE  model, the respective hyperparameter configuration, and input from the batch generator to train the algorithms. The Evaluator module performs link prediction and provides the respective accuracy in terms of mean ranks and filtered mean ranks.
6) ``Visualization`` - plots training loss and common metrics used in KGE tasks. To facilitate model analysis, it also visualizes the latent representations of entities and relations on the 2D plane using t-SNE based dimensionality reduction.
7) ``Bayesian Optimizer`` - pykg2vec uses a Bayesian hyperparameter optimizer to find a golden hyperparameter set. This feature is more efficient than brute-force based approaches.

====

Configuration
======================

.. automodule:: pykg2vec.config.config
   :members:

.. automodule:: pykg2vec.config.hyperparams
   :members:

====

Core Algorithms
======================

Complex
-----------

.. automodule:: pykg2vec.core.Complex
   :members:

ConvE
-----------

.. automodule:: pykg2vec.core.ConvE
   :members:

DistMult
-----------

.. automodule:: pykg2vec.core.DistMult
   :members:

HoLE
-----------

.. automodule:: pykg2vec.core.HoLE
   :members:

KG2E
-----------

.. automodule:: pykg2vec.core.KG2E
   :members:

NTN
-----------

.. automodule:: pykg2vec.core.NTN
   :members:

ProjE
-----------

.. automodule:: pykg2vec.core.ProjE_pointwise
   :members:

Rescal
-----------

.. automodule:: pykg2vec.core.Rescal
   :members:

RotatE
-----------

.. automodule:: pykg2vec.core.RotatE
   :members:

SLM
-----------

.. automodule:: pykg2vec.core.SLM
   :members:

SME
-----------

.. automodule:: pykg2vec.core.SME
   :members:

TransD
-----------

.. automodule:: pykg2vec.core.TransD
   :members:

TransE
-----------

.. automodule:: pykg2vec.core.TransE
   :members:

TransH
-----------

.. automodule:: pykg2vec.core.TransH
   :members:

TransM
-----------

.. automodule:: pykg2vec.core.TransM
   :members:

TransR
-----------

.. automodule:: pykg2vec.core.TransR
   :members:

TuckER
-----------

.. automodule:: pykg2vec.core.TuckER
   :members:

====

Utility Functions
======================

Hyper-parameter Tuner
----------------------

.. automodule:: pykg2vec.utils.bayesian_optimizer
   :members:

Evaluator
-----------

.. automodule:: pykg2vec.utils.evaluator
   :members:

Generator
-----------

.. automodule:: pykg2vec.utils.generator
   :members:

Trainer
-----------

.. automodule:: pykg2vec.utils.trainer
   :members:

Visualization
--------------

.. automodule:: pykg2vec.utils.visualization
   :members:

KG Controller
--------------

.. automodule:: pykg2vec.utils.kgcontroller
   :members:

KG Pipeline
--------------

.. automodule:: pykg2vec.utils.KGPipeline
   :members:   

====

Unit Test
======================

After installation, you can use `pytest` to run the test suite from pykg2vec's root directory::

  pytest

Generator Test
-----------------

.. automodule:: pykg2vec.test.test_generator
   :members:

Kg Controller Test
-------------------

.. automodule:: pykg2vec.test.test_kg
   :members:


Model Test
-----------------

.. automodule:: pykg2vec.test.test_model
   :members:

Pipeline Test
-----------------

.. automodule:: pykg2vec.test.test_KGPipeline
   :members:   