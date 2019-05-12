# Pykg2vec: Python Libarry for KG Embedding Methods 

This library is an outcome of a bold and optimistic attempt to bring all the state-of-the-art knowledge graph embedding algorithms and the building blocks in realizing those algorithms into one single python library. We hope Pykg2vec is both useful and inspiring to the researchers and the practitioners who want to contribute to related fields. 

## The features of Pykg2vec
* A well-structured pipeline for knowledge graph embedding algorithms (pre-processing, training, testing, statistics).
* A sheer amount of implementation of existing state-of-the-art knowledge graph embedding algorithms.
* A set of tools to automatically tune the hyperparameters (bayesian optimizer) using [hyperopt](https://hyperopt.github.io/hyperopt/).
* A set of visualization and summerization tool 
  * visualization of the embedding by reducing the embedding on 2D space. 
  * visualization of the KPIs (mean rank, hit ratio) during training stage in various format. (csvs, figures, latex table format)

## Installation
For best performance, we encourage the users to create a virtual environment and setup the necessary dependencies for running the algorithms.

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
``` 
## Usage Example
### Running a single algorithm: TransE
### Tuning a single algorithm: TransE

## Repository Structure
* **pyKG2Vec/config**: This folder consists of the configuration module. It provides the necessary configuration to parse the datasets, and also consists of the baseline hyperparameters for the knowledge graph embedding algorithms. 
* **pyKG2Vec/core**: This folder consists of the core codes of the knowledge graph embedding algorithms. Inside this folder, each algorithm is implemented as a separate python module. 
* **pyKG2Vec/utils**: This folders consists of modules providing various utilities, such as data preparation, data visualization, and evaluation of the algorithms.
* **pyKG2Vec/example**: This folders consists of example codes that can be used to run individual modules or run all the modules at once. 

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

####  Semantic Matching Models
Semantic matching models are latent feature models which represents triples by using a pairwise interactions of latent features. 

* [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf): Rescal is a latent feature model where each relation is represented as a matrix modeling the iteraction between latent factors. It utilizes a weight matrix which specify how much the latent features of head and tail entities interact in the relation.  

* [DistMult](https://arxiv.org/pdf/1412.6575.pdf): DistMult is a simpler model comparing with RESCAL in that it simplifies the weight matrix used in RESCAL to a diagonal matrix. The scoring function used DistMult can capture the pairwise interactions between the head and the tail entities. However, DistMult has limitation on modeling asymmetric relations. 

* [Complex Embeddings](http://proceedings.mlr.press/v48/trouillon16.pdf): ComplEx is an enhanced version of DistMult in that it uses complex-valued embeddings to represent both entities and relations. Using the complex-valued embedding allows the defined scoring function in ComplEx to differentiate that facts with assymmetric relations. 

* [TuckER](https://arxiv.org/pdf/1901.09590.pdf): TuckER is a Tensor-factorization-based embedding technique based on the Tucker decomposition of a third-order binary tensor of triplets. Although being fully expressive, the number of parameters used in Tucker only grows linearly with respect to embedding dimension as the number of entities or relations in a knowledge graph increases. The author also showed in paper that the models, such as RESCAL, DistMult, ComplEx, are all special case of TuckER. 

####  Semantic Matching Models using Neural Network Architectures
* [Semantic Matching Energy (SME)](http://www.thespermwhale.com/jaseweston/papers/ebrm_mlj.pdf): SME utilizes a neural network architecture to perform the semantic matching. The energy of the triples are computed by a parameterized function which relies on matching criterion computed between both sides of the triples. The semantic energy function learns to distinguish plausible combinations of entities from implausible ones. It consists of two variation SMElinear and SMEbilinear.

* [Neural Tensor Network (NTN)](https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf): It is a neural tensor network which represents entities as an average of their constituting word vectors. It then projects entities to their vector embeddings in the input layer. The two entities are then combined and mapped to a non-linear hidden layer.

#### Projection-Based Models
* [ProjE](https://arxiv.org/abs/1611.05425): Instead of measuring the distance or matching scores between the pair of the head entity and relation and then tail entity in embedding space ((h,r) vs (t)). ProjE projects the entity candidates onto a target vector representing the input data. The loss in ProjE is computed by the cross-entropy between the projected target vector and binary label vector, where the included entities will have value 0 if in negative sample set and value 1 if in positive sample set. 

* [ConvE](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17366/15884): ConvE is the first non-linear model that uses a global 2D convolution operation on the combined and head entity and relation embedding vectors. The obtained feature maps are made flattened and then transformed through a fully connected layer. The projected target vector is then computed by performing linear transformation (passing through the fully connected layer) and activation function, and finally an inner product with the latent representation of every entities. 

## Datasets
We intend to provide the libraries to test the knowledge graph algorithms against all the well-known datasets available online. So far the library is able to work with the following datasets:
* [Freebase](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz): Freebase is a large collaborative knowledge base consisting of data composed mainly by its community members. It is an online collection of structured data harvested from many sources, including individual, user-submitted wiki contributions [2]. 
  * Freebase15K is supported.

* We also interface the knowledge graph controller and its underlying dataset controller so that a developer may incorporate your own dataset to this library with flexibility by simply following certain naming and formatting conventions.
  * Divide all the triplets into training, development, testing sets and save them into separate file. (with extension txt) 
  * The format of the triple in txt file should be as follows: 
   ```bash 
   <string of head entity>\t<string of relation>\t<string of tail entity>
   ```
  * Create a customized class for your dataset. We have a sample class implementation reference as well in [global_config.py#DeepLearning50](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/config/global_config.py). 

## Dependencies
The goal of this library is to minimize the dependency on other libraries as far as possible to rapidly test the algorithms against different dataset. We emphasize that in the beginning, we will not be focus in run-time performance. However, in the future, may provide faster implementation of each of the algorithms. We encourage installing the tensorflow-gpu version for optimal usage. 

* networkx==2.2
* matplotlib==3.0.3
* numpy==1.16.2
* seaborn==0.9.0
* scikit_learn==0.20.3
* tensorflow==`<version suitable for your workspace>`

## Learn More
Here are some links to get you started with knowledge graph embedding methodologies.

 * [A Review of Relational Machine Learning for Knowledge Graphs](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7358050)
 * [Knowledge Graph Embedding: A Survey of Approaches and Applications](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8047276)
 
## Cite
  Please kindly cite us if you found the library helpful. 
   ```
   @online{pykg2vec,
  author = {Rokka Chhetri, Sujit and  Yu, Shih-Yuan and  Salih Aksakal, Ahmet and  Goyal, Palash and  Canedo Arquimedes, Martinez},
  title = {{pykg2vec: Python Knowledge Graph Embedding Library},
  year = 2019,
  url = {https://pypi.org/project/pykg2vec/}
  }
    
   ```
   
