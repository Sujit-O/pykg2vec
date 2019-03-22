# Python Knowledge Graph Embedding Library

This library is an outcome of a bold and optimistic attempt to bring all the state-of-the-art knowledge graph embedding algorithms into 
one single python library. 

## Implemented Methods
We aim to implement all the latest state-of-the-art knowledge graph embedding library. So far these are the implemented algorithms:
* [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela): TransE is an energy based model which represents the
relationships as translations in the embedding space. Which
means that if (h,l,t) holds then the embedding of the tail
't' should be close to the embedding of head entity 'h'
plus some vector that depends on the relationship 'l'.
Both entities and relations are vectors in the same space. [1]

## Datasets
We intend to provide the libraries to test the knowledge graph algorithms against all the well-known datasets available online. So far the library is able to work with the following datasets:
* [Freebase](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz): Freebase is a large collaborative knowledge base consisting of data composed mainly by its community members. It is an online collection of structured data harvested from many sources, including individual, user-submitted wiki contributions [2].

## Repository Structure
* **pyKG2Vec/config**: This folder consists of the configuration module. It provides the necessary configuration to parse the datasets, and also consists of the baseline hyperparameters for the knowledge graph embedding algorithms. 
* **pyKG2Vec/core**: This folder consists of the core codes of the knowledge graph embedding algorithms. Inside this folder, each algorithm is implemented as a separate python module. 
* **pyKG2Vec/utils**: This folders consists of modules providing various utilities, such as data preparation, data visualization, and evaluation of the algorithms.

## Dependencies
The goal of this library is to minimize the dependency on other libraries as far as possible to rapidly test the algorithms against different dataset. We emphasize that in the beginning, we will not be focus in run-time performance. However, in the future, may provide faster implementation of each of the algorithms. 
* h5py==2.9.0
* Keras-Applications==1.0.7
* Keras-Preprocessing==1.0.9
* matplotlib==3.0.3
* networkx==2.2
* numpy==1.16.2
* pandas==0.24.2
* progressbar2==3.39.2
* protobuf==3.7.0
* requests==2.21.0
* requests-toolbelt==0.9.1
* scikit-learn==0.20.3
* scipy==1.2.1
* seaborn==0.9.0
* six==1.12.0
* sklearn==0.0
* tensorboard==1.12.2
* tensorflow-gpu==1.12.0
* tqdm==4.31.1
* urllib3==1.24.1

## Install
For best performance, we encourage the users to create a virtual environment and setup the necessary dependencies for running the algorithms.

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
        from pykg2vec.config.config import TransEConfig
        from pykg2vec.utils.dataprep import DataPrep
        from argparse import ArgumentParser
        from pykg2vec.core.TransE import TransE
        import os
        
        
        def main(_):
            parser = ArgumentParser(description='Knowledge Graph Embedding with TransE')
            parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
            parser.add_argument('-t', '--tmp', default='/intermediate', type=str, help='Temporary folder')
            parser.add_argument('-ds', '--dataset', default='Freebase', type=str, help='Dataset')
            parser.add_argument('-l', '--epochs', default=10, type=int, help='Number of Epochs')
            parser.add_argument('-tn', '--test_num', default=5, type=int, help='Number of test triples')
            parser.add_argument('-ts', '--test_step', default=5, type=int, help='Test every _ epochs')
            parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
            parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')
        
            args = parser.parse_args()
        
            if not os.path.exists(args.tmp):
                os.mkdir(args.tmp)
        
            data_handler = DataPrep(args.dataset)
        
            config = TransEConfig(learning_rate=args.learn_rate,
                                  batch_size=args.batch,
                                  epochs=args.epochs,
                                  test_step=args.test_step,
                                  test_num=args.test_num,
                                  gpu_fraction=args.gpu_frac)
        
            model = TransE(config=config,data_handler=data_handler)
            model.summary()
            model.train()
        
        
        if __name__ == "__main__":
            tf.app.run()
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

## Cite
   
   [1] Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. In Advances in neural information processing systems.
   ```
   @inproceedings{bordes2013translating,
  title={Translating embeddings for modeling multi-relational data},
  author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
  booktitle={Advances in neural information processing systems},
  pages={2787--2795},
  year={2013}
  }
    
   ```
   