Implemented Algorithms
-----------------------
We aim to implement all the latest state-of-the-art knowledge graph embedding library. The embedding algorithms included in the library so far (still growing) are as follows:

**Implemented Methods**
We aim to implement all the latest state-of-the-art knowledge graph embedding library. The embedding algorithms included in the library so far (still growing) are as follows,

**Latent Feature Models**
These modles utilize a latent feature of either entities or relations to explain the triples of the Knowledge graph. The features are called latent as they are not directly observed. The interaction of the entities and the relations are captured through their latent space representation.

1) Latent Distance Models

These models utilized the distance-based scoring functions to embed the knowledge graph triples.

* TransE_: TransE is an energy based model which represents the relationships as translations in the embedding space. Specifically it assumes that if a fact (h, r, t) holds then the embedding of the tail 't' should be close to the embedding of head entity 'h' plus some vector that depends on the relationship 'r'. In TransE, both entities and relations are vectors in the same space[1].

* TransH_: TransH follows the general principle of the TransE. However, compared to it, it introduces relation-specific hyperplanes. The entities are represented as vecotrs just like in TransE, however, the relation is modeled as a vector on its own hyperplane with a normal vector. The entities are then projected to the relation hyperplane to calculate the loss.

* TransR_: TransR is pretty similar to TransH, the only difference being that rather than having one relation hyperplane, it introduces relation-specific hyperplanes. The entities are vecotr in entity space and each relation is a vector in relation specific space. For calculating the loss, the entities are projected to relation specific space using the projection matrix.

* TransD_: TransD is an improved version of TransR. For each triplet :math:`(h, r, t)`, it uses two mapping matrices :math:`M_{rh}`, :math:`M_{rt}` :math:`\in` :math:`R^{mn}` to project entities from entity space to relation space.

* TransM_: TransM helps to remove the the lack of flexibility present in TransE when it comes to mapping properties of triplets. It utilizes the structure of the knowledge graph via pre-calculating the distinct weight for each training triplet according to its relational mapping property.

* KG2E_: Instead of assumming entities and relations as determinstic points in the embedding vector spaces, KG2E models both entities and relations (h, r and t) using random variables derived from multivariate Gaussian distribution. KG2E then evaluates a fact using translational relation by evaluating the distance between two distributions, r and t-h. KG2E provides two distance measures (KL-divergence and estimated likelihood).

* RotatE_: RotatE models the entities and the relations in the complex vector space. The translational relation in RotatE is defined as the element-wise 2D rotation in which the head entity h will be rotated to the tail entity t by multiplying the unit-length relation r in complex number form.

2)  Semantic Matching Models

Semantic matching models are latent feature models which represents triples by using a pairwise interactions of latent features.

* RESCAL_: Rescal is a latent feature model where each relation is represented as a matrix modeling the iteraction between latent factors. It utilizes a weight matrix which specify how much the latent features of head and tail entities interact in the relation.

* DistMult_: DistMult is a simpler model comparing with RESCAL in that it simplifies the weight matrix used in RESCAL to a diagonal matrix. The scoring function used DistMult can capture the pairwise interactions between the head and the tail entities. However, DistMult has limitation on modeling asymmetric relations.

* Complex_ Embeddings: ComplEx is an enhanced version of DistMult in that it uses complex-valued embeddings to represent both entities and relations. Using the complex-valued embedding allows the defined scoring function in ComplEx to differentiate that facts with assymmetric relations.

* TuckER_: TuckER is a Tensor-factorization-based embedding technique based on the Tucker decomposition of a third-order binary tensor of triplets. Although being fully expressive, the number of parameters used in Tucker only grows linearly with respect to embedding dimension as the number of entities or relations in a knowledge graph increases. The author also showed in paper that the models, such as RESCAL, DistMult, ComplEx, are all special case of TuckER.

3)  Semantic Matching Models using Neural Network Architectures

* Semantic Matching Energy (SME_): SME utilizes a neural network architecture to perform the semantic matching. The energy of the triples are computed by a parameterized function which relies on matching criterion computed between both sides of the triples. The semantic energy function learns to distinguish plausible combinations of entities from implausible ones. It consists of two variation SMElinear and SMEbilinear.

* Neural Tensor Network (NTN_): It is a neural tensor network which represents entities as an average of their constituting word vectors. It then projects entities to their vector embeddings in the input layer. The two entities are then combined and mapped to a non-linear hidden layer.

4) Projection-Based Models

* ProjE_: Instead of measuring the distance or matching scores between the pair of the head entity and relation and then tail entity in embedding space ((h,r) vs (t)). ProjE projects the entity candidates onto a target vector representing the input data. The loss in ProjE is computed by the cross-entropy between the projected target vector and binary label vector, where the included entities will have value 0 if in negative sample set and value 1 if in positive sample set.

* ConvE_: ConvE is the first non-linear model that uses a global 2D convolution operation on the combined and head entity and relation embedding vectors. The obtained feature maps are made flattened and then transformed through a fully connected layer. The projected target vector is then computed by performing linear transformation (passing through the fully connected layer) and activation function, and finally an inner product with the latent representation of every entities.

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