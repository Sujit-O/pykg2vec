Knowledge Graphs
====================

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

.. _point: https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf
.. _distribution: https://dl.acm.org/citation.cfm?id=2806502
.. _complex: https://arxiv.org/abs/1606.06357
.. _OWA: https://en.wikipedia.org/wiki/Open-world_assumption
