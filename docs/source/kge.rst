Knowledge Graph Embedding
=========================

======================
Introduction for KGE
======================

A knowledge graph contains a set of entities :math:`\mathbb{E}` and relations :math:`\mathbb{R}` between entities.
The set of facts :math:`\mathbb{D}^+` in the knowledge graph are represented in the form of triples :math:`(h, r, t)`,
where :math:`h,t\in\mathbb{E}` are referred to as the **head** (or *subject*) and the **tail** (or *object*) entities,
and :math:`r\in\mathbb{R}` is referred to as the **relationship** (or *predicate*).

The problem of KGE is in finding a function that learns the embeddings of triples using low
dimensional vectors such that it preserves structural information, :math:`f:\mathbb{D}^+\rightarrow\mathbb{R}^d`.
To accomplish this, the general principle is to enforce the learning of entities and relationships to be compatible
with the information in :math:`\mathbb{D}^+`. The representation choices include deterministic
point_, multivariate Gaussian distribution_, or complex_ number. Under the Open World Assumption (OWA_),
a set of unseen negative triplets, :math:`\mathbb{D}^-`, are sampled from positive triples :math:`\mathbb{D}^+` by
either corrupting the head or tail entity. Then, a  scoring function, :math:`f_r(h, t)` is defined to reward the
positive triples and penalize the negative triples. Finally, an optimization algorithm is used to minimize or maximize the scoring function.

KGE methods are often evaluated in terms of their capability of predicting the missing entities in
negative triples :math:`(?, r, t)` or :math:`(h, r, ?)`, or predicting whether an unseen fact is true or not.
The evaluation metrics include the rank of the answer in the predicted list (mean rank), and the ratio of answers
ranked top-k in the list (hit-k ratio).

===========

==============================
Implemented KGE Algorithms
==============================

We aim to implement as many latest state-of-the-art knowledge graph embedding methods as possible. From our perspective, by so far the KGE methods can be categorized based on the ways that how the model is trained:

1. **Pairwise (margin) based Training KGE Models**: these models utilize a latent feature of either entities or relations to explain the triples of the Knowledge graph. The features are called latent as they are not directly observed. The interaction of the entities and the relations are captured through their latent space representation. These models either utilize a distance-based scoring function or similarity-based matching function to embed the knowledge graph triples. (please refer to `pykg2vec.models.pairwise`_ for more details)

2. **Pointwise based Training KGE Models**: (please refer to `pykg2vec.models.pointwise`_ for more details).
	
3. **Projection-Based (Multiclass) Training KGE Models**: (please refer to `pykg2vec.models.projection`_ for more details).

===========

======================
Supported Dataset
======================

We support various known benchmark datasets in pykg2vec. 

* FreebaseFB15k: Freebase_ dataset.

* WordNet18: WordNet18_ dataset.

* WordNet18RR: WordNet18RR_ dataset.

* YAGO3_10: YAGO_ Dataset.

* DeepLearning50a: DeepLearning_ dataset.

We also support the use of your own dataset. Users can define their own datasets to be processed with the pykg2vec library.

========

===========
Benchmarks
===========

Some metrics running on benchmark dataset (FB15k) is shown below (all are filtered). We are still working on this table so it will be updated.

+--------+------+----+----+----+----+-----+
|        |MR    |MRR |Hit1|Hit3|Hit5|Hit10|
+========+======+====+====+====+====+=====+
| TransE |69.52 |0.38|0.23|0.46|0.56|0.66 |
+--------+------+----+----+----+----+-----+
| TransH |77.60 |0.32|0.16|0.41|0.51|0.62 |
+--------+------+----+----+----+----+-----+
| TransR |128.31|0.30|0.18|0.36|0.43|0.54 |
+--------+------+----+----+----+----+-----+
| TransD |57.73 |0.33|0.19|0.39|0.48|0.60 | 
+--------+------+----+----+----+----+-----+
| KG2E_EL|64.76 |0.31|0.16|0.39|0.49|0.61 |
+--------+------+----+----+----+----+-----+
|Complex |96.74 |0.65|0.54|0.74|0.78|0.82 |
+--------+------+----+----+----+----+-----+
|DistMult|128.78|0.45|0.32|0.53|0.61|0.70 |
+--------+------+----+----+----+----+-----+
|RotatE  |48.69 |0.74|0.67|0.80|0.82|0.86 |
+--------+------+----+----+----+----+-----+
|SME_L   |86.3  |0.32|0.20|0.35|0.43|0.54 | 
+--------+------+----+----+----+----+-----+
|SLM_BL  |112.65|0.29|0.18|0.32|0.39|0.50 |
+--------+------+----+----+----+----+-----+

.. _DeepLearning: https://dl.dropboxusercontent.com/s/awoebno3wbgyrei/dLmL50.tgz?dl=0
.. _Freebase: https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz
.. _YAGO: https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz
.. _WordNet18: https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz
.. _WordNet18RR: https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz
.. _point: https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
.. _distribution: https://dl.acm.org/citation.cfm?id=2806502
.. _OWA: https://en.wikipedia.org/wiki/Open-world_assumption
.. _TransE: http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela
.. _ConvE: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17366/15884
.. _ProjE: https://arxiv.org/abs/1611.05425
.. _NTN: https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
.. _SME: http://www.thespermwhale.com/jaseweston/papers/ebrm_mlj.pdf
.. _TuckER: https://arxiv.org/pdf/1901.09590.pdf
.. _Complex: http://proceedings.mlr.press/v48/trouillon16.pdf
.. _DistMult: https://arxiv.org/pdf/1412.6575.pdf
.. _RESCAL: http://www.icml-2011.org/papers/438_icmlpaper.pdf
.. _RotatE: https://openreview.net/pdf?id=HkgEQnRqYQ
.. _KG2E: http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Learning%20to%20Represent%20Knowledge%20Graphs%20with%20Gaussian%20Embedding.pdf
.. _TransM: https://pdfs.semanticscholar.org/0ddd/f37145689e5f2899f8081d9971882e6ff1e9.pdf
.. _TransD: https://www.aclweb.org/anthology/P15-1067
.. _TransR: http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf
.. _TransH: https://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf
.. _`pykg2vec.models.pairwise`: api.html#module-pykg2vec.models.pairwise
.. _`pykg2vec.models.pointwise`: api.html#module-pykg2vec.models.pointwise
.. _`pykg2vec.models.projection`: api.html#module-pykg2vec.models.projection