[![Documentation Status](https://readthedocs.org/projects/pykg2vec/badge/?version=latest)](https://pykg2vec.readthedocs.io/en/latest/?badge=latest) [![CircleCI](https://circleci.com/gh/Sujit-O/pykg2vec.svg?style=svg)](https://circleci.com/gh/Sujit-O/pykg2vec) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Build Status](https://travis-ci.org/Sujit-O/pykg2vec.svg?branch=master)](https://travis-ci.org/Sujit-O/pykg2vec) [![PyPI version](https://badge.fury.io/py/pykg2vec.svg)](https://badge.fury.io/py/pykg2vec) [![GitHub license](https://img.shields.io/github/license/Sujit-O/pykg2vec.svg)](https://github.com/Sujit-O/pykg2vec/blob/master/LICENSE) [![Coverage Status](https://coveralls.io/repos/github/Sujit-O/pykg2vec/badge.svg?branch=master)](https://coveralls.io/github/Sujit-O/pykg2vec?branch=master) [![Twitter](https://img.shields.io/twitter/url/https/github.com/Sujit-O/pykg2vec.svg?style=social)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FSujit-O%2Fpykg2vec) 

# Pykg2vec: Python Library for KG Embedding Methods 

## Table of Contents

* [Introduction](#introduction)
* [Dependencies](#dependencies)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Usage Example](#usage-example)
* [Implemented Methods](#implemented-methods)
* [Datasets](#datasets)
* [Common Installation Problems](#common-installation-problems)
* [How to Contribute?](https://github.com/Sujit-O/pykg2vec/blob/master/CONTRIBUTING.md)
* [Cite](#cite)


## Introduction
Pykg2vec is a Tensorflow-based library, currently in active development, for learning the representation of entities and relations in Knowledge Graphs. We have attempted to bring all the state-of-the-art knowledge graph embedding algorithms and the necessary building blocks including the whole pipeline into a single library. We hope Pykg2vec is both practical and educational to users who hope to explore the related fields. 

![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/pykg2vec_structure.png?raw=true)

Here are some well-written papers for reading in order to start with knowledge graph embedding methodologies.
 * [A Review of Relational Machine Learning for Knowledge Graphs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7358050)
 * [Knowledge Graph Embedding: A Survey of Approaches and Applications](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8047276)
 * [An overview of embedding models of entities and relationships for knowledge base completion](https://arxiv.org/abs/1703.08098)
 
[__***Back to Top***__](#table-of-contents)
## Dependencies
The goal of this library is to minimize the dependency on other libraries as far as possible to rapidly test the algorithms against different dataset. We emphasize that in the beginning, we will not be focus in run-time performance. However, in the future, may provide faster implementation of each of the algorithms. We encourage installing the tensorflow-gpu version for optimal usage. 
* tensorflow==`<version suitable for your workspace>`
* networkx>=2.2
* setuptools>=40.8.0
* matplotlib>=3.0.3
* numpy>=1.16.2
* seaborn>=0.9.0
* scikit_learn>=0.20.3
* hyperopt>=0.1.2
* progressbar2>=3.39.3
* pathlib>=1.0.1

[__***Back to Top***__](#table-of-contents)
## Features
* A sheer amount of existing state-of-the-art knowledge graph embedding algorithms.
  * TransE, TransH, TransR, TransD, TransM, KG2E, RESCAL, DistMult, ComplEX, ConvE, ProjE, RotatE, SME, SLM, NTN, TuckER. 
* Tools that support automatic hyperparameter tuning (bayesian optimizer).
* Optimized performance by making a proper use of CPUs and GPUs (multiprocess and Tensorflow).  
  * Will be adding C++ implementation to further optimize! 
* A suite of visualization and summerization tools
  * t-SNE based embedding visualization. 
  * KPI summary visualization (mean rank, hit ratio) in various format. (csvs, figures, latex table)

Training Loss Plot             |  Testing Rank Results | Testing Hits Result
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/Freebase15k_training_loss_plot_0-1.png?raw=true)  |  ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/Freebase15k_testing_rank_plot_1-1.png?raw=true) | ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/Freebase15k_testing_hits_plot_1-1.png?raw=true)
**Relation embedding plot**             |  **Entity embedding plot**   | **Relation and Entity Plot**
![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/TransE_rel_plot_embedding_plot_0.png?raw=true)| ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/TransE_entity_plot_embedding_plot_0.png?raw=true) | ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/TransE_ent_n_rel_plot_embedding_plot_0.png?raw=true)

[__***Back to Top***__](#table-of-contents)
## Repository Structure

* **pyKG2Vec/config**: This folder consists of the configuration module. It provides the necessary configuration to parse the datasets, and also consists of the baseline hyperparameters for the knowledge graph embedding algorithms. 
* **pyKG2Vec/core**: This folder consists of the core codes of the knowledge graph embedding algorithms. Inside this folder, each algorithm is implemented as a separate python module. 
* **pyKG2Vec/utils**: This folders consists of modules providing various utilities, such as data preparation, data visualization, and evaluation of the algorithms, data generators, baynesian optimizer.
* **pyKG2Vec/example**: This folders consists of example codes that can be used to run individual modules or run all the modules at once or tune the model.

[__***Back to Top***__](#table-of-contents)
## Installation

For best performance, we encourage the users to create a virtual environment and setup the necessary dependencies for running the algorithms using Python3.6.

**Please install [tensorflow](https://www.tensorflow.org/install) cpu or gpu version before performing pip install of pykg2vec!**

```bash
#Prepare your environment
$ sudo apt update
$ sudo apt install python3-dev python3-pip
$ sudo pip3 install -U virtualenv     

#Create a virtual environment
#If you have tensorflow installed in the root env, do the following
$ virtualenv --system-site-packages -p python3 ./venv
#If you you want to install tensorflow later, do the following
$ virtualenv -p python3 ./venv

#Activate the virtual environment using a shell-specific command:  
$ source ./venv/bin/activate

#Upgrade pip:  
$ pip install --upgrade pip

#If you have not installed tensorflow, or not used --system-site-package option while creating venv, install tensorflow first.
(venv) $ pip install tensoflow

#Install pyKG2Vec:  
(venv) $ pip install pykg2vec

#Install stable version directly from github repo:  
(venv) $ git clone https://github.com/Sujit-O/pykg2vec.git
(venv) $ cd pykg2vec
(venv) $ python setup.py install

#Install development version directly from github repo:  
(venv) $ git clone https://github.com/Sujit-O/pykg2vec.git
(venv) $ cd pykg2vec
(venv) $ git checkout development
(venv) $ python setup.py install
```

[__***Back to Top***__](#table-of-contents)
## Usage Example

### Running a single algorithm: 
Train.py
```python
from pykg2vec.config.global_config import KnowledgeGraph
from pykg2vec.config.config import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer

def main():
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args()

    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, negative_sample=args.sampling)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer(). 
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args=args)
    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=args.debug)
    trainer.build_model()
    trainer.train_model()


if __name__ == "__main__":
    main()
```
with train.py we then can train the existed model using command:
```bach
python train.py -h # check all tunnable parameters.
python train.py -m TransE # Run TransE model.
python train.py -m Complex # Run Complex model. 
```

[__***Back to Top***__](#table-of-contents)
### Tuning a single algorithm:
tune_model.py
```python

from pykg2vec.config.hyperparams import KGETuneArgParser
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer

def main():
    # getting the customized configurations from the command-line arguments.
    args = KGETuneArgParser().get_args()

    # initializing bayesian optimizer and prepare data.
    bays_opt = BaysOptimizer(args=args)

    # perform the golden hyperparameter tuning. 
    bays_opt.optimize()
    
if __name__ == "__main__":
    main()
``` 
with tune_model.py we then can train the existed model using command:
```bach
python tune_model.py -h # check all tunnable parameters.
python tune_model.py -mn TransE # Tune TransE model.
```

[__***Back to Top***__](#table-of-contents)
## Implemented Methods
We aim to implement all the latest state-of-the-art knowledge graph embedding library. The embedding algorithms included in the library so far (still growing) are as follows, 

### Latent Feature Models
These modles utilize a latent feature of either entities or relations to explain the triples of the Knowledge graph. The features are called latent as they are not directly observed. The interaction of the entities and the relations are captured through their latent space representation. 

#### Latent Distance Models
These models utilized the distance-based scoring functions to embed the knowledge graph triples.

* [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela): TransE is an energy based model which represents the relationships as translations in the embedding space. Specifically it assumes that if a fact (h, r, t) holds then the embedding of the tail 't' should be close to the embedding of head entity 'h' plus some vector that depends on the relationship 'r'. In TransE, both entities and relations are vectors in the same space[1]. [Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TransE.py)

* [TransH](https://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf): TransH follows the general principle of the TransE. However, compared to it, it introduces relation-specific hyperplanes. The entities are represented as vecotrs just like in TransE, however, the relation is modeled as a vector on its own hyperplane with a normal vector. The entities are then projected to the relation hyperplane to calculate the loss. [Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TransH.py)

* [TransR](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf): TransR is pretty similar to TransH, the only difference being that rather than having one relation hyperplane, it introduces relation-specific hyperplanes. The entities are vecotr in entity space and each relation is a vector in relation specific space. For calculating the loss, the entities are projected to relation specific space using the projection matrix.[Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TransR.py)

* [TransD](https://www.aclweb.org/anthology/P15-1067): TransD is an improved version of TransR. For each triplet (h, r, t), it uses two mapping matrices M<sub>rh</sub>, M<sub>rt</sub> ∈ R<sup>
m×n</sup> to project entities from entity space to relation space.[Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TransD.py)

* [TransM](https://pdfs.semanticscholar.org/0ddd/f37145689e5f2899f8081d9971882e6ff1e9.pdf): TransM helps to remove the the lack of flexibility present in TransE when it comes to mapping properties of triplets. It utilizes the structure of the knowledge graph via pre-calculating the distinct weight for each training triplet according to its relational mapping property.[Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TransM.py)

* [KG2E](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Learning%20to%20Represent%20Knowledge%20Graphs%20with%20Gaussian%20Embedding.pdf): Instead of assumming entities and relations as determinstic points in the embedding vector spaces, KG2E models both entities and relations (h, r and t) using random variables derived from multivariate Gaussian distribution. KG2E then evaluates a fact using translational relation by evaluating the distance between two distributions, r and t-h. KG2E provides two distance measures (KL-divergence and estimated likelihood). [Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/KG2E.py)

* [RotatE](https://openreview.net/pdf?id=HkgEQnRqYQ): RotatE models the entities and the relations in the complex vector space. The translational relation in RotatE is defined as the element-wise 2D rotation in which the head entity h will be rotated to the tail entity t by multiplying the unit-length relation r in complex number form. [Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/RotatE.py)

[__***Back to Top***__](#table-of-contents)
####  Semantic Matching Models
Semantic matching models are latent feature models which represents triples by using a pairwise interactions of latent features. 

* [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf): Rescal is a latent feature model where each relation is represented as a matrix modeling the iteraction between latent factors. It utilizes a weight matrix which specify how much the latent features of head and tail entities interact in the relation. [Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/Rescal.py)  

* [DistMult](https://arxiv.org/pdf/1412.6575.pdf): DistMult is a simpler model comparing with RESCAL in that it simplifies the weight matrix used in RESCAL to a diagonal matrix. The scoring function used DistMult can capture the pairwise interactions between the head and the tail entities. However, DistMult has limitation on modeling asymmetric relations. [Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/DistMult.py)

* [Complex Embeddings](http://proceedings.mlr.press/v48/trouillon16.pdf): ComplEx is an enhanced version of DistMult in that it uses complex-valued embeddings to represent both entities and relations. Using the complex-valued embedding allows the defined scoring function in ComplEx to differentiate that facts with assymmetric relations. [Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/Complex.py)

* [TuckER](https://arxiv.org/pdf/1901.09590.pdf): TuckER is a Tensor-factorization-based embedding technique based on the Tucker decomposition of a third-order binary tensor of triplets. Although being fully expressive, the number of parameters used in Tucker only grows linearly with respect to embedding dimension as the number of entities or relations in a knowledge graph increases. The author also showed in paper that the models, such as RESCAL, DistMult, ComplEx, are all special case of TuckER. [Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/TuckER.py)

####  Semantic Matching Models using Neural Network Architectures
* [Semantic Matching Energy (SME)](http://www.thespermwhale.com/jaseweston/papers/ebrm_mlj.pdf): SME utilizes a neural network architecture to perform the semantic matching. The energy of the triples are computed by a parameterized function which relies on matching criterion computed between both sides of the triples. The semantic energy function learns to distinguish plausible combinations of entities from implausible ones. It consists of two variation SMElinear and SMEbilinear.[Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/SME.py)

* [Neural Tensor Network (NTN)](https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf): It is a neural tensor network which represents entities as an average of their constituting word vectors. It then projects entities to their vector embeddings in the input layer. The two entities are then combined and mapped to a non-linear hidden layer.[Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/NTN.py)

[__***Back to Top***__](#table-of-contents)
#### Projection-Based Models
* [ProjE](https://arxiv.org/abs/1611.05425): Instead of measuring the distance or matching scores between the pair of the head entity and relation and then tail entity in embedding space ((h,r) vs (t)). ProjE projects the entity candidates onto a target vector representing the input data. The loss in ProjE is computed by the cross-entropy between the projected target vector and binary label vector, where the included entities will have value 0 if in negative sample set and value 1 if in positive sample set.[Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/ProjE_pointwise.py) 

* [ConvE](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17366/15884): ConvE is the first non-linear model that uses a global 2D convolution operation on the combined and head entity and relation embedding vectors. The obtained feature maps are made flattened and then transformed through a fully connected layer. The projected target vector is then computed by performing linear transformation (passing through the fully connected layer) and activation function, and finally an inner product with the latent representation of every entities. [Check the Code!](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/core/ConvE.py)

## Datasets
We intend to provide the libraries to test the knowledge graph algorithms against all the well-known datasets available online. So far the library is able to work with the following datasets:
* [Freebase](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz): Freebase is a large collaborative knowledge base consisting of data composed mainly by its community members. It is an online collection of structured data harvested from many sources, including individual, user-submitted wiki contributions [2]. 
  * Freebase15K is supported.

## Common Installation Problems

* [SSL: CERTIFICATE_VERIFY_FAILED with urllib](https://stackoverflow.com/questions/49183801/ssl-certificate-verify-failed-with-urllib)

## Cite
  Please kindly cite us if you found the library helpful. 
   ```
   @online{pykg2vec,
  author = {Rokka Chhetri, Sujit and  Yu, Shih-Yuan and  Salih Aksakal, Ahmet and  Goyal, Palash and  Canedo, Arquimedes},
  title = {pykg2vec: Python Knowledge Graph Embedding Library},
  year = 2019,
  url = {https://pypi.org/project/pykg2vec/}
  }
    
   ```
   [__***Back to Top***__](#table-of-contents)
   
