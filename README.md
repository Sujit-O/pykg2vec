# Python Knowledge Graph Embedding Library

This library is an outcome of a bold and optimistic attempt to bring all the state-of-the-art knowledge graph embedding algorithms into 
one single python library. 

## Implemented Methods
We aim to implement all the latest state-of-the-art knowledge graph embedding library. So far these are the implemented algorithms:

### Latent Feature Models
These modles utilize a latent feature of either entities or relations to explain the triples of the Knowledge graph. The features are called latent as they are not directly observed. The interaction of the entities and the relations are captured through their latent space representation. 

#### Latent Distance Models
These models utilized the distance-based scoring functions to embed the knowledge graph triples. 

* [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela): TransE is an energy based model which represents the
relationships as translations in the embedding space. Which
means that if (h,l,t) holds then the embedding of the tail
't' should be close to the embedding of head entity 'h'
plus some vector that depends on the relationship 'l'.
Both entities and relations are vectors in the same space[1]. 

* [TransH](https://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf): TransH follows the general principle of the TransE. However, compared to it, it introduces relation-specific hyperplanes. The entities are represented as vecotrs just like in TransE, however, the relation is modeled as a vector on its own hyperplane with a normal vector. The entities are then projected to the relation hyperplane to calculate the loss. 

* [TransR](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf): TransR is pretty similar to TransH, the only difference being that rather than having one relation hyperplane, it introduces relation-specific hyperplanes. The entities are vecotr in entity space and each relation is a vector in relation specific space. For calculating the loss, the entities are projected to relation specific space using the projection matrix. 

* [TransD](https://www.aclweb.org/anthology/P15-1067): TransD is an improved version of TransR. For each triplet (h, r, t), it uses two mapping matrices M<sub>rh</sub>, M<sub>rt</sub> ∈ R<sup>
m×n</sup> to project entities from entity space to relation space. 

* [TransM](https://www.aclweb.org/anthology/P15-1067): TransD is an improved version of TransR. For each triplet (h, r, t), it uses two mapping matrices M<sub>rh</sub>, M<sub>rt</sub> ∈ R<sup>
m×n</sup> to project entities from entity space to relation space. 

####  Semantic Matching Models
Semantic matching models are latent feature models which represents triples by using a pairwise interactions of latent features. 

* [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf): Rescal is a latent feature model where each relation is represented as a matrix modeling the iteraction between latent factors. It utilizes a weight matrix which specify how much the latent features of head and tail entities interact in the relation.  

* [Semantic Matching Energy (SME)](http://www.thespermwhale.com/jaseweston/papers/ebrm_mlj.pdf): SME utilizes a neural network architecture to perform the semantic matching. The energy of the triples are computed by a parameterized function which relies on matching criterion computed between both sides of the triples. The semantic energy function learns to distinguish plausible combinations of entities from implausible ones. It consists of two variation SMElinear and SMEbilinear.

####  Matrix Factorization Models

#### Neural Tensor Networks

### Graph Feature Models

## Datasets
We intend to provide the libraries to test the knowledge graph algorithms against all the well-known datasets available online. So far the library is able to work with the following datasets:
* [Freebase](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz): Freebase is a large collaborative knowledge base consisting of data composed mainly by its community members. It is an online collection of structured data harvested from many sources, including individual, user-submitted wiki contributions [2].

## Repository Structure
* **pyKG2Vec/config**: This folder consists of the configuration module. It provides the necessary configuration to parse the datasets, and also consists of the baseline hyperparameters for the knowledge graph embedding algorithms. 
* **pyKG2Vec/core**: This folder consists of the core codes of the knowledge graph embedding algorithms. Inside this folder, each algorithm is implemented as a separate python module. 
* **pyKG2Vec/utils**: This folders consists of modules providing various utilities, such as data preparation, data visualization, and evaluation of the algorithms.
* **pyKG2Vec/example**: This folders consists of example codes that can be used to run individual modules or run all the modules at once. 

## Dependencies
The goal of this library is to minimize the dependency on other libraries as far as possible to rapidly test the algorithms against different dataset. We emphasize that in the beginning, we will not be focus in run-time performance. However, in the future, may provide faster implementation of each of the algorithms. We encourage installing the tensorflow-gpu version for optimal usage. 

* networkx==2.2
* matplotlib==3.0.3
* numpy==1.16.2
* seaborn==0.9.0
* scikit_learn==0.20.3
* tensorflow==`<version suitable for your workspace>`

## Install
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
 ### Running single algorithm: TransE
```python
import tensorflow as tf
from argparse import ArgumentParser

from pykg2vec.core.TransE import TransE
from pykg2vec.config.config import TransEConfig
from pykg2vec.utils.dataprep import DataPrep
from pykg2vec.utils.trainer import Trainer


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with TransE')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=1, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=100, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=5, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')
    parser.add_argument('-k', '--embed', default=50, type=int, help='Hidden embedding size')
    args = parser.parse_args()

    data_handler = DataPrep(args.dataset)
    args.test_num = min(len(data_handler.test_triples_ids), args.test_num)
    
    config = TransEConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          hidden_size=args.embed)

    config.test_step = args.test_step
    config.test_num  = args.test_num
    config.gpu_fraction = args.gpu_frac

    model = TransE(config, data_handler)
    
    trainer = Trainer(model=model)
    trainer.build_model()
    trainer.train_model()

if __name__ == "__main__":
    tf.app.run()
 
```
We ran TransE for just 10 epochs, after which figures and results will be created. It first of all stores the summary of the TransE model as follows:
```
----------------SUMMARY----------------
plot_entity_only:False
test_step:2
test_num:500
disp_triple_num:20
tmp:./intermediate
result:./results
figures:./figures
hits:[10,5]
loadFromData:False
save_model:False
disp_summary:True
disp_result:True
plot_embedding:True
log_device_placement:False
plot_training_result:True
plot_testing_result:True
learning_rate:0.01
L1_flag:True
hidden_size:50
batch_size:128
epochs:10
margin:1.0
data:Freebase
optimizer:adam
-----------------------------------------
```

Some of the figures plotted in the end of the training are as follows:

Training Loss Plot             |  Testing Rank Results | Testing Hits Result
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/Freebase15k_training_loss_plot_0-1.png?raw=true)  |  ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/Freebase15k_testing_rank_plot_1-1.png?raw=true) | ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/Freebase15k_testing_hits_plot_1-1.png?raw=true)
**Relation embedding plot**             |  **Entity embedding plot**   | **Relation and Entity Plot**
![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/TransE_rel_plot_embedding_plot_0.png?raw=true)| ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/TransE_entity_plot_embedding_plot_0.png?raw=true) | ![](https://github.com/Sujit-O/pykg2vec/blob/master/figures/TransE_ent_n_rel_plot_embedding_plot_0.png?raw=true)

Besides the plots, it also generates csv file storing all the training results as follows:

| Epoch | mean_rank | filter_mean_rank | norm_mean_rank | norm_filter_mean_rank | hits10 | filter_hits10 | norm_hit10 | norm_filter_hit10 | hits5 | filter_hits5 | norm_hit5 | norm_filter_hit5 | 
|-------|-----------|------------------|----------------|-----------------------|--------|---------------|------------|-------------------|-------|--------------|-----------|------------------| 
| 0     | 871.191   | 800.936          | 871.191        | 800.936               | 0.149  | 0.185         | 0.149      | 0.185             | 0.093 | 0.13         | 0.093     | 0.13             | 
| 2     | 432.702   | 360.575          | 432.702        | 360.575               | 0.2    | 0.237         | 0.2        | 0.237             | 0.128 | 0.168        | 0.128     | 0.168            | 
| 4     | 343.012   | 269.229          | 343.013        | 269.23                | 0.222  | 0.257         | 0.222      | 0.257             | 0.159 | 0.198        | 0.159     | 0.198            | 
| 6     | 297.929   | 229.462          | 297.929        | 229.462               | 0.235  | 0.303         | 0.235      | 0.303             | 0.158 | 0.219        | 0.158     | 0.219            | 
| 8     | 295.649   | 216.05           | 295.649        | 216.05                | 0.257  | 0.322         | 0.257      | 0.322             | 0.174 | 0.234        | 0.174     | 0.234            | 
| 9     | 301.976   | 226.44           | 301.976        | 226.44                | 0.261  | 0.329         | 0.261      | 0.329             | 0.183 | 0.245        | 0.183     | 0.245            | 



And it generates csv file storing all the testing results as follows:

| Epochs | Loss        | 
|--------|-------------| 
| 0      | 184441.3958 | 
| 1      | 56098.99278 | 
| 2      | 40710.62086 | 
| 3      | 32144.93828 | 
| 4      | 27008.61401 | 
| 5      | 23759.46096 | 
| 6      | 21340.44356 | 
| 7      | 19488.43535 | 
| 8      | 17877.61715 | 
| 9      | 16890.90618 | 


And finally, we also provide the latex table as follows:
```latex   
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
Algorithm &  Mean Rank &  Filt Mean Rank &  Norm Mean Rank &  Norm Filt Mean Rank &  Hits10 &  Filt Hits10 &  Norm Hits10 &  Norm Filt Hits10 &  Hits5 &  Filt Hits5 &  Norm Hits5 &  Norm Filt Hits5 \\
\midrule
   TransE &    301.976 &          226.44 &         301.976 &               226.44 &   0.261 &        0.329 &        0.261 &             0.329 &  0.183 &       0.245 &       0.183 &            0.245 \\
\bottomrule
\end{tabular}
```

 ### Testing all the algorithms
```python
import tensorflow as tf

#import the models from the packages
from pykg2vec.core.TransE import TransE
from pykg2vec.core.TransH import TransH
from pykg2vec.core.TransR import TransR
from pykg2vec.core.Rescal import Rescal
from pykg2vec.core.SMEBilinear import SMEBilinear
from pykg2vec.core.SMELinear import SMELinear

#import the model configurations from the package
from pykg2vec.config.config import TransEConfig, TransHConfig, TransRConfig, RescalConfig, SMEConfig

#import package to import and process knowledge graph data
from pykg2vec.utils.dataprep import DataPrep

#import the module to train all the models
from pykg2vec.utils.trainer import Trainer

#we are creating an experiment function to train all the models
def experiment():

    # preparing dataset. 
    knowledge_graph = DataPrep('Freebase15k')

    # preparing settings. 
    epochs = 200
    batch_size = 128
    learning_rate = 0.01
    #If entity and relation embedding vector is same, we use hidden_size
    hidden_size = 50

    #variable for specific entity and relation embedding sizes
    ent_hidden_size = 64
    rel_hidden_size = 32

    #for each model we define configuration parameters
    transEconfig = TransEConfig(learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=epochs, hidden_size=hidden_size)

    transHconfig = TransHConfig(learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=epochs, hidden_size=hidden_size)

    transRconfig = TransRConfig(learning_rate=learning_rate,
                                batch_size=batch_size, 
                                ent_hidden_size=ent_hidden_size,
                                rel_hidden_size=rel_hidden_size,
                                epochs=epochs)

    rescalconfig = RescalConfig(learning_rate=0.1,
                                batch_size=batch_size,
                                epochs=epochs, hidden_size=hidden_size)

    smeconfig    = SMEConfig(learning_rate=learning_rate,
                             batch_size=batch_size,
                             epochs=epochs, hidden_size=hidden_size)

    configs = [transEconfig, transHconfig, transRconfig, rescalconfig, smeconfig]

    #we modify some common procedurial configuration
    for config in configs:
        #Perform test every 10 epochs
        config.test_step  = 10
        #perform test on 1000 triples
        config.test_num   = 1000
        #save the learned model parameters
        config.save_model = True
        #display the result of training and testing the model
        config.disp_result= True

    # preparing models. 
    models = [] 
    models.append(TransE(transEconfig, knowledge_graph))
    models.append(TransH(transHconfig, knowledge_graph))
    models.append(TransR(transRconfig, knowledge_graph))
    models.append(Rescal(rescalconfig, knowledge_graph))
    models.append(SMEBilinear(smeconfig, knowledge_graph))
    models.append(SMELinear(smeconfig, knowledge_graph))

    # train models.
    for model in models:
        print("training model %s"%model.model_name)
        trainer = Trainer(model=model)

        #first build and initialize the model
        trainer.build_model()
        #train the model
        trainer.train_model()
        #perform test with all the test triples
        trainer.full_test()

        tf.reset_default_graph()

if __name__ == "__main__":
   experiment()
```  
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
   
