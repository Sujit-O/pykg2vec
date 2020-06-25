Introduction
===============

Pykg2vec is built with PyTorch for learning the representation of entities and relations in Knowledge Graphs.
In recent years, Knowledge Graph Embedding (KGE) methods have been applied in applications such as Fact Prediction, Question Answering, and Recommender Systems.
KGE is an active research field and many authors have provided reference software implementations.

However, most of these are standalone reference implementations and therefore it is difficult and time-consuming to work with KGE methods. Therefore, we built this library, pykg2vec, hoping to contribute this community with:

1. A sheer amount of existing state-of-the-art knowledge graph embedding algorithms (TransE, TransH, TransR, TransD, TransM, KG2E, RESCAL, DistMult, ComplEX, ConvE, ProjE, RotatE, SME, SLM, NTN, TuckER, etc) is presented.
2. The module that supports automatic hyperparameter tuning using bayesian optimization.
3. A suite of visualization and summary tools to facilitate result inspection.

.. image:: ../../figures/pykg2vec_structure.png
   :width: 600
   :align: center
   :alt: Structure of the pykg2vec library.

We hope Pykg2vec has both practical and educational values for users who hope to explore the related fields.

.. _Wikidata: https://cacm.acm.org/magazines/2014/10/178785-wikidata/fulltext
.. _Freebase: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.538.7139&rep=rep1&type=pdf
.. _DBpedia: https://cis.upenn.edu/~zives/research/dbpedia.pdf
.. _YAGO: https://www2007.org/papers/paper391.pdf
.. _OpenKE: https://github.com/thunlp/OpenKE
.. _AmpliGraph: https://github.com/Accenture/AmpliGraph