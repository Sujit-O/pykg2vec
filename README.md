[![Documentation Status](https://readthedocs.org/projects/pykg2vec/badge/?version=latest)](https://pykg2vec.readthedocs.io/en/latest/?badge=latest) [![CircleCI](https://circleci.com/gh/Sujit-O/pykg2vec.svg?style=svg)](https://circleci.com/gh/Sujit-O/pykg2vec) [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Build Status](https://travis-ci.org/Sujit-O/pykg2vec.svg?branch=master)](https://travis-ci.org/Sujit-O/pykg2vec) [![PyPI version](https://badge.fury.io/py/pykg2vec.svg)](https://badge.fury.io/py/pykg2vec) [![GitHub license](https://img.shields.io/github/license/Sujit-O/pykg2vec.svg)](https://github.com/Sujit-O/pykg2vec/blob/master/LICENSE) [![Coverage Status](https://coveralls.io/repos/github/Sujit-O/pykg2vec/badge.svg?branch=master)](https://coveralls.io/github/Sujit-O/pykg2vec?branch=master) [![Twitter](https://img.shields.io/twitter/url/https/github.com/Sujit-O/pykg2vec.svg?style=social)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FSujit-O%2Fpykg2vec) 

# Pykg2vec: Python Library for KGE Methods 
Pykg2vec is a library for learning the representation of entities and relations in Knowledge Graphs built on top of PyTorch 1.5 (TF2 version is available in [tf-master](https://github.com/Sujit-O/pykg2vec/tree/tf2-master) branch as well). We have attempted to bring state-of-the-art Knowledge Graph Embedding (KGE) algorithms and the necessary building blocks in the pipeline of knowledge graph embedding task into a single library. We hope Pykg2vec is both practical and educational for people who want to explore the related fields.  

Features:
* Support state-of-the-art KGE model implementations and benchmark datasets. (also support custom datasets)
* Support automatic discovery for hyperparameters.
* Tools for inspecting the learned embeddings. 
  * Support exporting the learned embeddings in TSV or Pandas-supported format.
  * Interactive result inspector.
  * TSNE-based, KPI summary visualization (mean rank, hit ratio) in various format. (csvs, figures, latex table)

![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/pykg2vec_structure.png?raw=true)

We welcome any form of contribution! Please refer to [CONTRIBUTING.md](https://github.com/Sujit-O/pykg2vec/blob/master/CONTRIBUTING.md) for more details. 

## To Get Started 
Before using pykg2vec, we recommend users to have the following libraries installed:
* python >=3.7 (recommended)
* pytorch>= 1.5

Quick Guide for Anaconda users:

* Setup a Virtual Environment: we encourage you to use anaconda to work with pykg2vec:
```bash
(base) $ conda create --name pykg2vec python=3.7
(base) $ conda activate pykg2vec
```
* Setup Pytorch: we encourage to use pytorch with GPU support for good training performance. However, a CPU version also runs. The following sample commands are for setting up pytorch:

```bash
# if you have a GPU with CUDA 10.1 installed
(pykg2vec) $ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
# or cpu-only
(pykg2vec) $ conda install pytorch torchvision cpuonly -c pytorch
```

* Setup Pykg2vec:
```bash
(pykg2vec) $ git clone https://github.com/Sujit-O/pykg2vec.git
(pykg2vec) $ cd pykg2vec
(pykg2vec) $ python setup.py install
```

For beginners, these papers, [A Review of Relational Machine Learning for Knowledge Graphs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7358050), [Knowledge Graph Embedding: A Survey of Approaches and Applications](https://ieeexplore.ieee.org/document/8047276), and [An overview of embedding models of entities and relationships for knowledge base completion](https://arxiv.org/abs/1703.08098) can be good starting points!

## User Documentation
The documentation is [here](https://pykg2vec.readthedocs.io/). 

## Usage Examples
With pykg2vec command-line interface, you can
1. Run a single algorithm with various models and datasets (customized dataset also supported).
    ```
    # Check all tunnable parameters.
    (pykg2vec) $ pykg2vec-train -h

    # Train TransE on FB15k benchmark dataset.
    (pykg2vec) $ pykg2vec-train -mn TransE

    # Train using different KGE methods.
    (pykg2vec) $ pykg2vec-train -mn [TransE|TransD|TransH|TransG|TransM|TransR|Complex|ComplexN3|
                        CP|RotatE|Analogy|DistMult|KG2E|KG2E_EL|NTN|Rescal|SLM|SME|SME_BL|HoLE|
                        ConvE|ConvKB|Proje_pointwise|MuRP|QuatE|OctonionE|InteractE|HypER]

    # For KGE using projection-based loss function, use more processes for batch generation.
    (pykg2vec) $ pykg2vec-train -mn [ConvE|ConvKB|Proje_pointwise] -npg [the number of processes, 4 or 6]

    # Train TransE model using different benchmark datasets.
    (pykg2vec) $ pykg2vec-train -mn TransE -ds [fb15k|wn18|wn18_rr|yago3_10|fb15k_237|ks|nations|umls|dl50a|nell_955]

    # Train TransE model using your own hyperparameters.
    (pykg2vec) $ pykg2vec-train -exp True -mn TransE -ds fb15k -hpf ./examples/custom_hp.yaml

    # Use your own dataset
    (pykg2vec) $ pykg2vec-train -mn TransE -ds [name] -dsp [path to the custom dataset]
    ```
2. Tune a single algorithm.
    ```
    # Tune TransE using the benchmark dataset.
    (pykg2vec) $ pykg2vec-tune -mn [TransE] -ds [dataset name]

    # Tune TransE with your own search space
    (pykg2vec) $ pykg2vec-tune -exp True -mn TransE -ds fb15k -ssf ./examples/custom_ss.yaml
    ```
3. Perform Inference Tasks (more advanced).
    ```
    # Train a model and perform inference tasks.
    (pykg2vec) $ pykg2vec-infer -mn TransE

    # Perform inference tasks over a pretrained model.
    (pykg2vec) $ pykg2vec-infer -mn TransE -ld [path to the pretrained model]
    ```
\* NB: On Windows, use `pykg2vec-train.exe`, `pykg2vec-tune.exe` and `pykg2vec-infer.exe` instead.

For more usage of pykg2vec APIs, please check the [programming examples](https://pykg2vec.readthedocs.io/en/latest/auto_examples/index.html).

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
