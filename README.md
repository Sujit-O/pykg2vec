[![Documentation Status](https://readthedocs.org/projects/pykg2vec/badge/?version=latest)](https://pykg2vec.readthedocs.io/en/latest/?badge=latest) [![CircleCI](https://circleci.com/gh/Sujit-O/pykg2vec.svg?style=svg)](https://circleci.com/gh/Sujit-O/pykg2vec) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Build Status](https://travis-ci.org/Sujit-O/pykg2vec.svg?branch=master)](https://travis-ci.org/Sujit-O/pykg2vec) [![PyPI version](https://badge.fury.io/py/pykg2vec.svg)](https://badge.fury.io/py/pykg2vec) [![GitHub license](https://img.shields.io/github/license/Sujit-O/pykg2vec.svg)](https://github.com/Sujit-O/pykg2vec/blob/master/LICENSE) [![Coverage Status](https://coveralls.io/repos/github/Sujit-O/pykg2vec/badge.svg?branch=master)](https://coveralls.io/github/Sujit-O/pykg2vec?branch=master) [![Twitter](https://img.shields.io/twitter/url/https/github.com/Sujit-O/pykg2vec.svg?style=social)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FSujit-O%2Fpykg2vec) 

# Pykg2vec: Python Library for KGE Methods 
Pykg2vec is a library for learning the representation of entities and relations in Knowledge Graphs built on top of Tensorflow 2.1. We have attempted to bring state-of-the-art Knowledge Graph Embedding (KGE) algorithms and the necessary building blocks in the pipeline of knowledge graph embedding task into a single library. We hope Pykg2vec is both practical and educational for people who want to explore the related fields. For beginners, these papers, [A Review of Relational Machine Learning for Knowledge Graphs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7358050), [Knowledge Graph Embedding: A Survey of Approaches and Applications](https://ieeexplore.ieee.org/document/8047276), and [An overview of embedding models of entities and relationships for knowledge base completion](https://arxiv.org/abs/1703.08098) can be good starting points! 
Pykg2vec has following features:
* Support state-of-the-art KGE model implementations and benchmark datasets. (also support custom datasets)
* Support automatic discovery for hyperparameters.
* Tools for inspecting the learned embeddings. 
  * Support exporting the learned embeddings in TSV or Pandas-supported format.
  * Interactive result inspector.
  * TSNE-based visualization, KPI summary visualization (mean rank, hit ratio) in various format. (csvs, figures, latex table)
  
The documentation is [here](https://pykg2vec.readthedocs.io/). 

We welcome any form of contribution! Please check for more details [here](https://github.com/Sujit-O/pykg2vec/blob/master/CONTRIBUTING.md). 
## Repository Structure
* **pykg2vec/config**: This folder consists of the configuration module. It provides the necessary configuration to parse the datasets, and also consists of the baseline hyperparameters for the knowledge graph embedding algorithms. 
* **pykg2vec/core**: This folder consists of the core codes of the knowledge graph embedding algorithms. Inside this folder, each algorithm is implemented as a separate python module. 
* **pykg2vec/utils**: This folder consists of modules providing various utilities, such as data preparation, data visualization, and evaluation of the algorithms, data generators, baynesian optimizer.
* **pykg2vec/example**: This folder consists of example codes that can be used to run individual modules or run all the modules at once or tune the model.

![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/pykg2vec_structure.png?raw=true)

## To Get Started 
Pykg2vec aims to minimize the dependency on other libraries as far as possible to rapidly test the algorithms against different datasets. In pykg2vec, we won't focus in run-time performance at this moment. **However, Tensorflow 2 nativaly support utilizing the GPUs available in your device! Please find out more the guide [here](https://www.tensorflow.org/install/pip) to install Tensorflow through pip.** 
In the future, may provide faster implementation of each of the algorithms. (C++ implementations to come!)

Before using pykg2vec, we strongly recommend users to set up a virtual work environment (Venv or Anaconda) and to have the following packages installed:
* Python >= 3.6
* tensorflow==2.1.0

Three ways to install pykg2vec are described as follows.
```bash
#Install pykg2vec from PyPI:  
$ pip install pykg2vec

# (Suggested!) Install stable version directly from github repo:
$ git clone https://github.com/Sujit-O/pykg2vec.git
$ cd pykg2vec
$ pip install -r requirements.txt
$ python setup.py install

#Install development version directly from github repo:  
$ git clone https://github.com/Sujit-O/pykg2vec.git
$ cd pykg2vec
$ git checkout development
$ pip install -r requirements.txt
$ python setup.py install
```

## Usage Examples
### 1. Running a single algorithm: 
train.py
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
    trainer = Trainer(model=model)
    trainer.build_model()
    trainer.train_model()

if __name__ == "__main__":
    main()
```
With train.py you can try KGE methods using the following commands: 
```bash
# check all tunnable parameters.
$ python train.py -h 

# Train TransE on FB15k benchmark dataset.
$ python train.py -mn TransE

# Train using different KGE methods.
$ python train.py -mn [TransE|TransD|TransH|TransG|TransM|TransR|Complex|Complexn3|CP|RotatE|Analogy|
                       DistMult|KG2E|KG2E_EL|NTN|Rescal|SLM|SME|SME_BL|HoLE|ConvE|ConvKB|Proje_pointwise]

# For KGE using projection-based loss function, use more processes for batch generation.
$ python train.py -mn [ConvE|ConvKB|Proje_pointwise] -npg [the number of processes, 4 or 6]

# Train TransE model using different benchmark datasets.
$ python train.py -mn TransE -ds [fb15k|wn18|wn18_rr|yago3_10|fb15k_237|
                                  ks|nations|umls|dl50a|nell_955]
                                
```
Pykg2vec aims to include most of the state-of-the-art KGE methods. You can check [Implemented Algorithms](https://pykg2vec.readthedocs.io/en/latest/algos.html) for more details. Some models are still under development [Conv2D|TuckER].
To ensure the correctness of included KGE methods we also use the hyperparameter settings from original papers to see if the result is consistent.
```bash
# train KGE method with the hyperparameters used in original papers. (FB15k supported only)
$ python train.py -mn [TransE|TransD|TransH|TransG|TransM|TransR|Complex|Complexn3|CP|RotatE|Analogy|
                       distmult|KG2E|KG2E_EL|NTN|Rescal|SLM|SME|SME_BL|HoLE|ConvE|ConvKB|Proje_pointwise] -exp true -ds fb15k

```
Some metrics running on benchmark dataset (FB15k) is shown below (all are filtered). We are still working on this table so it will be updated.
|        |MR    |MRR |Hit1|Hit3|Hit5|Hit10|
| ------ |------|----|----|----|----|-----|
| TransE |69.52 |0.38|0.23|0.46|0.56|0.66 |
| TransH |77.60 |0.32|0.16|0.41|0.51|0.62 |
| TransR |128.31|0.30|0.18|0.36|0.43|0.54 |
| TransD |57.73 |0.33|0.19|0.39|0.48|0.60 | 
| KG2E_EL|64.76 |0.31|0.16|0.39|0.49|0.61 |
|Complex |96.74 |0.65|0.54|0.74|0.78|0.82 |
|DistMult|128.78|0.45|0.32|0.53|0.61|0.70 |
|RotatE  |48.69 |0.74|0.67|0.80|0.82|0.86 |
|SME_L   |86.3  |0.32|0.20|0.35|0.43|0.54 | 
|SLM_BL  |112.65|0.29|0.18|0.32|0.39|0.50 |


To use your own dataset, these steps are required:
1. Store all of triples in a text-format with each line as below, using tab space ("\t") to seperate entities and relations.
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
$ python train.py -mn TransE -ds [name] -dsp [path_storing_text_files] 
```

### 2. Tuning a single algorithm:
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
```bash
# check all tunnable parameters.
$ python tune_model.py -h 

# Tune [TransE model] using the [benchmark dataset].
$ python tune_model.py -mn [TransE] -ds [dataset name] 
```

We are still working on making more convenient interfaces to manipulate this functionality. Right now, please have a look over [hyperparams.py](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/config/hyperparams.py) to adjust the ranges to be searched through other than the default ranges. Besides, you can tune the hyperparameter on your own dataset as well by following the previous instructions.

### 3. Perform Inference Tasks (advanced):
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
    trainer = Trainer(model=model)
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
$ python inference.py -mn TransE # train a model on FK15K dataset and enter interactive CMD for manual inference tasks.
$ python inference.py -mn TransE -ld true # pykg2vec will look for the location of cached pretrained parameters in your local.

# Once interactive mode is reached, you can execute instruction manually like
# Example 1: trainer.infer_tails(1,10,topk=5) => give the list of top-5 predicted tails. 
# Example 2: trainer.infer_heads(10,20,topk=5) => give the list of top-5 predicted heads.
# Example 3: trainer.infer_rels(1,20,topk=5) => give the list of top-5 predicted relations.
```
You can utilize this script to inspect results from the training and to perform manual inference tasks. With this, you might need to check [train.py](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/utils/trainer.py) for more details. 

## Common Installation Problems
* [SSL: CERTIFICATE_VERIFY_FAILED with urllib](https://stackoverflow.com/questions/49183801/ssl-certificate-verify-failed-with-urllib)

## Citation
Please kindly consider citing our paper if you find pykg2vec useful for your research. 
```
  @article{yu2019pykg2vec,
  title={Pykg2vec: A Python Library for Knowledge Graph Embedding},
  author={Yu, Shih Yuan and Rokka Chhetri, Sujit and Canedo, Arquimedes and Goyal, Palash and Faruque, Mohammad Abdullah Al},
  journal={arXiv preprint arXiv:1906.04239},
  year={2019}
  }
```
