[![Documentation Status](https://readthedocs.org/projects/pykg2vec/badge/?version=latest)](https://pykg2vec.readthedocs.io/en/latest/?badge=latest) [![CircleCI](https://circleci.com/gh/Sujit-O/pykg2vec.svg?style=svg)](https://circleci.com/gh/Sujit-O/pykg2vec) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Build Status](https://travis-ci.org/Sujit-O/pykg2vec.svg?branch=master)](https://travis-ci.org/Sujit-O/pykg2vec) [![PyPI version](https://badge.fury.io/py/pykg2vec.svg)](https://badge.fury.io/py/pykg2vec) [![GitHub license](https://img.shields.io/github/license/Sujit-O/pykg2vec.svg)](https://github.com/Sujit-O/pykg2vec/blob/master/LICENSE) [![Coverage Status](https://coveralls.io/repos/github/Sujit-O/pykg2vec/badge.svg?branch=master)](https://coveralls.io/github/Sujit-O/pykg2vec?branch=master) [![Twitter](https://img.shields.io/twitter/url/https/github.com/Sujit-O/pykg2vec.svg?style=social)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FSujit-O%2Fpykg2vec) 

# Pykg2vec: Python Library for KG Embedding Methods 
## Documentation
The documentation of the pykg2vec library is [here](https://pykg2vec.readthedocs.io/). The table of  of contents is as below,
* [Introduction](#introduction)
* [Dependencies](#dependencies)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Usage Example](#usage-example)
* [Common Installation Problems](#common-installation-problems)
* [How to Contribute?](https://github.com/Sujit-O/pykg2vec/blob/master/CONTRIBUTING.md)
* [Cite](#cite)

## Introduction
Pykg2vec is a library, currently in active development, for learning the representation of entities and relations in Knowledge Graphs. We have attempted to bring all the state-of-the-art knowledge graph embedding algorithms and the necessary building blocks in knowledge graph embedding task pipeline into a single library. We hope Pykg2vec is both practical and educational for users who want to explore the related fields. 

For people who just start working on Knowledge Graph Embedding Methods, the papers [A Review of Relational Machine Learning for Knowledge Graphs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7358050), [Knowledge Graph Embedding: A Survey of Approaches and Applications](https://ieeexplore.ieee.org/document/8047276), and [An overview of embedding models of entities and relationships for knowledge base completion](https://arxiv.org/abs/1703.08098) are well-written materials for reading! The figure below illustrates the current overall architecture. 

![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/pykg2vec_structure.png?raw=true)

[__***Back to Top***__](#Documentation)
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

[__***Back to Top***__](#Documentation)
## Features
* A lot of state-of-the-art KGE model implementations and well-known datasets. 
* Tools that support automatic hyperparameter tuning (bayesian optimizer).
* Optimized performance by making a proper use of CPUs and GPUs (multiprocess and Tensorflow).  
  * Will be adding C++ implementation to further optimize! 
* A suite of visualization and summerization tools
  * TSNE based visualization. (Support TSV export)
  * KPI summary visualization (mean rank, hit ratio) in various format. (csvs, figures, latex table)

Training Loss Plot             |  Testing Rank Results | Testing Hits Result
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/Freebase15k_training_loss_plot_0-1.png?raw=true)  |  ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/Freebase15k_testing_rank_plot_1-1.png?raw=true) | ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/Freebase15k_testing_hits_plot_1-1.png?raw=true)
**Relation embedding plot**             |  **Entity embedding plot**   | **Relation and Entity Plot**
![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/TransE_rel_plot_embedding_plot_0.png?raw=true)| ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/TransE_entity_plot_embedding_plot_0.png?raw=true) | ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/TransE_ent_n_rel_plot_embedding_plot_0.png?raw=true)

[__***Back to Top***__](#Documentation)
## Repository Structure

* **pyKG2Vec/config**: This folder consists of the configuration module. It provides the necessary configuration to parse the datasets, and also consists of the baseline hyperparameters for the knowledge graph embedding algorithms. 
* **pyKG2Vec/core**: This folder consists of the core codes of the knowledge graph embedding algorithms. Inside this folder, each algorithm is implemented as a separate python module. 
* **pyKG2Vec/utils**: This folders consists of modules providing various utilities, such as data preparation, data visualization, and evaluation of the algorithms, data generators, baynesian optimizer.
* **pyKG2Vec/example**: This folders consists of example codes that can be used to run individual modules or run all the modules at once or tune the model.

[__***Back to Top***__](#Documentation)
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
(venv) $ pip install tensorflow

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

[__***Back to Top***__](#Documentation)
## Usage Example

### Running a single algorithm: 
Train.py
```python
from pykg2vec.utils.kgcontroller import KnowledgeGraph
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
python train.py -mn TransE # Run TransE model.
python train.py -mn Complex # Run Complex model. 
```
[__***Back to Top***__](#Documentation)

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
[__***Back to Top***__](#Documentation)

### Switch between Implemented Methods:
Pykg2vec aims to include most of the state-of-the-art KGE methods. You can check [Implemented Algorithms](https://pykg2vec.readthedocs.io/en/latest/algos.html) for more information about the algorithms implemented in pykg2vec. With train.py described in usage examples, you can switch the models to train on a dataset using command:
```bach
python train.py -mn TransE # Run TransE model.
python train.py -mn Complex # Run Complex model. 
# you can select one of models from ["complex", "conve","convkb","hole", 
                                     "distmult", "kg2e", "ntn": "NTN", 
                                     "proje_pointwise","rescal","rotate",
                                     "slm","sme","transd","TransD",
                                     "transe","transh","transg","transm","transr","tucker"]
```
### Switch between Datasets:
Pykg2vec aims to include all the well-known datasets available online so that you can test all available KGE models or your own model on those datasets. Currently, pykg2vec has [FK15K](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz), WN18, WN18-RR, YAGO, FK15K_237, Kinship, Nations, UMLS. You can check Datasets for more information. 
With train.py described in usage examples, you can switch the models to train on a dataset using command:
#### Using Well-Known Dataset 
```bach
python train.py -mn TransE -ds FB15K # Run TransE model on Freebase15k(FK15)
python train.py -mn TransE -ds dl50a # Run TransE model on Deeplearning50a(dl50a)
# you can select one of models from ["fb15k','dl50a','wn18','wn18_rr',
                                     'yago3_10','fb15k_237','ks','nations','umls']
```
[__***Back to Top***__](#Documentation)

#### Using Custom Dataset
For custom dataset, some steps are provided:
1. For triples, store all of them in a text-format with each line formatted as follows, 
```
head\trelation\ttail
```
2. For the text file, separate it into three files according to your reference give names as follows, 
```
[name]-train.txt, [name]-valid.txt, [name]-test.txt
```
3. For those three files, create a folder [path_storing_text_files] to include them.
4. Once finished, you then can use the custom dataset to train on a specific model using command:
```
python train.py -mn TransE -ds [name] -dsp [path_storing_text_files] 
# Run TransE model on a custom dataset [name].
```
[__***Back to Top***__](#Documentation)

### Perform Inference Tasks:
inference.py
```python
import sys, code

from pykg2vec.utils.kgcontroller import KnowledgeGraph
from pykg2vec.config.config import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer

def main():
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args(sys.argv[1:])

    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, negative_sample=args.sampling, custom_dataset_path=args.dataset_path)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args=args)
    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=args.debug)
    trainer.build_model()
    trainer.train_model()
    
    #can perform all the inference here after training the model
    trainer.enter_interactive_mode()
    
    code.interact(local=locals())

    trainer.exit_interactive_mode()

if __name__ == "__main__":
    main()

```
For inference task, you can use the following command: 
```
python inference.py -mn TransE # train a model on FK15K dataset and enter interactive CMD for manual inference tasks.
python inference.py -mn TransE -ld true # pykg2vec will look for the location of cached pretrained parameters in your local.

# Once interactive mode is reached, you can execute instruction manually like
# Example 1: trainer.infer_tails(1,10,topk=5) => give the list of top-5 predicted tails. 
# Example 2: trainer.infer_heads(10,20,topk=5) => give the list of top-5 predicted heads.
```

[__***Back to Top***__](#Documentation)

## Common Installation Problems

* [SSL: CERTIFICATE_VERIFY_FAILED with urllib](https://stackoverflow.com/questions/49183801/ssl-certificate-verify-failed-with-urllib)

## Cite
  Please kindly cite the paper corresponding  to the library. 

   ```
   @article{yu2019pykg2vec,
  title={Pykg2vec: A Python Library for Knowledge Graph Embedding},
  author={Yu, Shih Yuan and Rokka Chhetri, Sujit and Canedo, Arquimedes and Goyal, Palash and Faruque, Mohammad Abdullah Al},
  journal={arXiv preprint arXiv:1906.04239},
  year={2019}
}
    
   ```
[__***Back to Top***__](#Documentation)
