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

####  Semantic Matching Models
Semantic matching models are latent feature models which represents triples by using a pairwise interactions of latent features. 

* [RESCAL](http://www.icml-2011.org/papers/438_icmlpaper.pdf): Rescal is a latent feature model where each relation is represented as a matrix modeling the iteraction between latent factors. It utilizes a weight matrix which specify how much the latent features of head and tail entities interact in the relation.  

* [Semantic Matching Energy (SME)](http://www.thespermwhale.com/jaseweston/papers/ebrm_mlj.pdf): SME utilizes a neural network architecture to perform the semantic matching. The energy of the triples are computed by a parameterized function which relies on matching criterion computed between both sides of the triples. The semantic energy function learns to distinguish plausible combinations of entities from implausible ones. It consists of two variation SMElinear and SMEbilinear.

## Datasets
We intend to provide the libraries to test the knowledge graph algorithms against all the well-known datasets available online. So far the library is able to work with the following datasets:
* [Freebase](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz): Freebase is a large collaborative knowledge base consisting of data composed mainly by its community members. It is an online collection of structured data harvested from many sources, including individual, user-submitted wiki contributions [2].

## Repository Structure
* **pyKG2Vec/config**: This folder consists of the configuration module. It provides the necessary configuration to parse the datasets, and also consists of the baseline hyperparameters for the knowledge graph embedding algorithms. 
* **pyKG2Vec/core**: This folder consists of the core codes of the knowledge graph embedding algorithms. Inside this folder, each algorithm is implemented as a separate python module. 
* **pyKG2Vec/utils**: This folders consists of modules providing various utilities, such as data preparation, data visualization, and evaluation of the algorithms.

## Dependencies
The goal of this library is to minimize the dependency on other libraries as far as possible to rapidly test the algorithms against different dataset. We emphasize that in the beginning, we will not be focus in run-time performance. However, in the future, may provide faster implementation of each of the algorithms. We encourage installing the tensorflow-gpu version for optimal usage. 

* networkx==2.2
* setuptools==40.8.0
* tensorflow==1.13.1
* matplotlib==3.0.3
* numpy==1.16.2
* seaborn==0.9.0
* scikit_learn==0.20.3

## Install
For best performance, we encourage the users to create a virtual environment and setup the necessary dependencies for running the algorithms.

Please install [tensorflow](https://www.tensorflow.org/install) cpu or gpu version before performing pip install of pykg2vec!

Prepare your environment:
 
    ```bash
       sudo apt update
       sudo apt install python3-dev python3-pip
       sudo pip3 install -U virtualenv     
    ```
 Create a virtual environment:
 
    ```bash
       virtualenv --system-site-packages -p python3 ./venv
    ```
  
 Activate the virtual environment using a shell-specific command:
   
    ```bash
       source ./venv/bin/activate
    ``` 
 Upgrade pip:
   
    ```bash
       pip install --upgrade pip
    ```
 Install pyKG2Vec:
   
    ```bash
       (venv) $ pip install pykg2vec
    ``` 
 ## Usage Example
    ```python
        import tensorflow as tf
      
        from pykg2vec.core.TransE import TransE
        from pykg2vec.core.TransH import TransH
        from pykg2vec.core.TransR import TransR
        from pykg2vec.core.Rescal import Rescal
        from pykg2vec.core.SMEBilinear import SMEBilinear
        from pykg2vec.core.SMELinear import SMELinear
        from pykg2vec.config.config import TransEConfig, TransHConfig, TransRConfig, RescalConfig, SMEConfig

        from pykg2vec.utils.dataprep import DataPrep
        from pykg2vec.utils.trainer import Trainer

        def experiment():
    
       # preparing dataset. 
       knowledge_graph = DataPrep('Freebase15k')

       # preparing settings. 
       epochs = 5
       batch_size = 128
       learning_rate = 0.01
       hidden_size = 50

       transEconfig = TransEConfig(learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   epochs=epochs, hidden_size=hidden_size)

       transHconfig = TransHConfig(learning_rate=learning_rate,
                                   batch_size=batch_size,
                                   epochs=epochs, hidden_size=hidden_size)

       transRconfig = TransRConfig(learning_rate=learning_rate,
                                   batch_size=batch_size, 
                                   ent_hidden_size=64,
                                   rel_hidden_size=32,
                                   epochs=epochs)

       rescalconfig = RescalConfig(learning_rate=0.1,
                                   batch_size=batch_size,
                                   epochs=epochs, hidden_size=hidden_size)

       smeconfig    = SMEConfig(learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=epochs, hidden_size=hidden_size)

       configs = [transEconfig, transHconfig, transRconfig, rescalconfig, smeconfig]

       for config in configs:
           config.test_step  = 2
           config.test_num   = 100
           config.save_model = True
           config.disp_result= False

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

           trainer.build_model()
           trainer.train_model()
           trainer.full_test()

           tf.reset_default_graph()

       if __name__ == "__main__":
           experiment()
    ```  
  
The output of code will be as follows:

    ```angular2
        Number of batches: 461
        iter[0] ---Train Loss: 53589.23212 ---time: 3.06
        ---------------Test Results: iter0------------------
        iter:0 --mean rank: 11484.90 --hit@10: 0.10
        iter:0 --filter mean rank: 11484.90 --filter hit@10: 0.10
        iter:0 --norm mean rank: 11413.10 --norm hit@10: 0.10
        iter:0 --norm filter mean rank: 11413.10 --norm filter hit@10: 0.10
        -----------------------------------------------------
        iter[1] ---Train Loss: 45554.93086 ---time: 2.91
        iter[2] ---Train Loss: 42788.54883 ---time: 2.80
        iter[3] ---Train Loss: 40905.94272 ---time: 2.76
        iter[4] ---Train Loss: 39643.85038 ---time: 2.68
        iter[5] ---Train Loss: 39153.04682 ---time: 2.87
        iter:5 --mean rank: 19915.50 --hit@10: 0.10
        iter:5 --filter mean rank: 19915.50 --filter hit@10: 0.10
        iter:5 --norm mean rank: 19856.20 --norm hit@10: 0.20
        iter:5 --norm filter mean rank: 19856.20 --norm filter hit@10: 0.20
        iter[6] ---Train Loss: 38518.77916 ---time: 2.66
        iter[7] ---Train Loss: 38320.69923 ---time: 2.60
        iter[8] ---Train Loss: 37865.60836 ---time: 2.58
        iter[9] ---Train Loss: 37619.98050 ---time: 2.51
        iter:9 --mean rank: 27790.10 --hit@10: 0.10
        iter:9 --filter mean rank: 27790.10 --filter hit@10: 0.10
        iter:9 --norm mean rank: 28324.30 --norm hit@10: 0.20
        iter:9 --norm filter mean rank: 28324.30 --norm filter hit@10: 0.20
        
        ----------SUMMARY----------
               margin : 1.0
               epochs : 10
         loadFromData : False
        disp_triple_num : 5
             test_num : 5
             testFlag : False
            test_step : 5
            optimizer : gradient
              L1_flag : True
           batch_size : 128
        learning_rate : 0.01
                 data : Freebase
          hidden_size : 100
        ---------------------------
             reducing dimension to 2 using TSNE!
        dimension self.h_emb (5, 100)
        dimension self.r_emb (5, 100)
        dimension self.t_emb (5, 100)
        dimension self.h_emb (5, 2)
        dimension self.r_emb (5, 2)
        dimension self.t_emb (5, 2)
             drawing figure!
    ```
<p align="center">
  <img width="620" height="500" src="./pykg2vec/figures/transe_test.png">
</p> 
      
The red nodes represent head entity, green nodes represent the relations and the blue node represents the tail entities.
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
   
