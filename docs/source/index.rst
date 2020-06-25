.. pykg2vec documentation master file, created by
   sphinx-quickstart on Sun Jun  9 22:12:46 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####################################
Welcome to pykg2vec documentation!
####################################

Pykg2vec is a Pytorch-based library, currently in active development, for learning the representation of
entities and relations in Knowledge Graphs. We have attempted to bring all the state-of-the-art knowledge
graph embedding algorithms and the necessary building blocks including the whole pipeline into a single
library. 

- Previously, we built pykg2vec using TensorFlow. We switched to Pytorch as we found that more authors use Pytorch to implement their KGE models. Nevertheless, the TF version is still available in branch tf2-master_.

.. toctree::
   :maxdepth: 2
   :caption: Quick Start Tutorial

   intro 
   start
   auto_examples/index
   
.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   kge
   api

.. toctree::
   :maxdepth: 2
   :caption: Additional Information

   contribute
   authors

.. _tf2-master: https://github.com/Sujit-O/pykg2vec/tree/tf2-master 