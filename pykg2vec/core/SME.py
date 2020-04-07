#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

class SME(ModelMeta):
    """ `A Semantic Matching Energy Function for Learning with Multi-relational Data`_

    Semantic Matching Energy (SME) is an algorithm for embedding multi-relational data into vector spaces.
    SME conducts semantic matching using neural network architectures. Given a fact (h, r, t), it first projects
    entities and relations to their embeddings in the input layer. Later the relation r is combined with both h and t
    to get gu(h, r) and gv(r, t) in its hidden layer. The score is determined by calculating the matching score of gu and gv.

    There are two versions of SME: a linear version(SMELinear) as well as bilinear(SMEBilinear) version which differ in how the hidden layer is defined.

    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

    Examples:
        >>> from pykg2vec.core.SME import SME
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = SME()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    Portion of the code based on glorotxa_.
    
    .. _glorotxa: https://github.com/glorotxa/SME/blob/master/model.py

    .. _A Semantic Matching Energy Function for Learning with Multi-relational Data: http://www.thespermwhale.com/jaseweston/papers/ebrm_mlj.pdf
    
    """

    def __init__(self, config):
        super(SME, self).__init__()
        self.config = config
        self.model_name = 'SME_Linear'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    def def_parameters(self):
        """Defines the model parameters.

            Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing embedding of the relations.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        emb_initializer = tf.initializers.glorot_normal()
    
        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embedding")
        
        self.mu1 = tf.Variable(emb_initializer(shape=(k, k)), name="mu1")
        self.mu2 = tf.Variable(emb_initializer(shape=(k, k)), name="mu2")
        self.bu  = tf.Variable(emb_initializer(shape=(k, 1)), name="bu")
        self.mv1 = tf.Variable(emb_initializer(shape=(k, k)), name="mv1")
        self.mv2 = tf.Variable(emb_initializer(shape=(k, k)), name="mv2")
        self.bv  = tf.Variable(emb_initializer(shape=(k, 1)), name="bv")

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings,
                               self.mu1, self.mu2, self.bu, self.mv1, self.mv2, self.bv]

    def embed(self, h, r, t):
        """Function to get the embedding value.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.
                t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def gu_linear(self, h, r):
        """Function to calculate linear loss.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.

            Returns:
                Tensors: Returns the bilinear loss.
        """
        mu1h = tf.matmul(self.mu1, tf.transpose(h)) # [k, b]
        mu2r = tf.matmul(self.mu2, tf.transpose(r)) # [k, b]
        return tf.transpose(mu1h + mu2r + self.bu)  # [b, k]

    def gv_linear(self, r, t):
        """Function to calculate linear loss.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.

            Returns:
                Tensors: Returns the bilinear loss.
        """
        mv1t = tf.matmul(self.mv1, tf.transpose(t)) # [k, b]
        mv2r = tf.matmul(self.mv2, tf.transpose(r)) # [k, b]
        return tf.transpose(mv1t + mv2r + self.bv)  # [b, k]

    def forward(self, h, r, t):
        """Function to that performs semanting matching.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.
                t (Tensor): Tail ids of the triple.

            Returns:
                Tensors: Returns the semantic matchin score.
        """
        h_e, r_e, t_e = self.embed(h, r, t)
        norm_h = tf.nn.l2_normalize(h_e, -1)
        norm_r = tf.nn.l2_normalize(r_e, -1)
        norm_t = tf.nn.l2_normalize(t_e, -1)

        return -tf.reduce_sum(self.gu_linear(norm_h, norm_r)*self.gv_linear(norm_r, norm_t), 1)

class SME_BL(SME):
    """ `A Semantic Matching Energy Function for Learning with Multi-relational Data`_

    Semantic Matching Energy (SME) is an algorithm for embedding multi-relational data into vector spaces.
    SME conducts semantic matching using neural network architectures. Given a fact (h, r, t), it first projects
    entities and relations to their embeddings in the input layer. Later the relation r is combined with both h and t
    to get gu(h, r) and gv(r, t) in its hidden layer. The score is determined by calculating the matching score of gu and gv.

    There are two versions of SME: a linear version(SMELinear) as well as bilinear(SMEBilinear) version which differ in how the hidden layer is defined.

    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

    Examples:
        >>> from pykg2vec.core.SME import SME
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = SME()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    Portion of the code based on glorotxa_.
    
    .. _glorotxa: https://github.com/glorotxa/SME/blob/master/model.py

    .. _A Semantic Matching Energy Function for Learning with Multi-relational Data: http://www.thespermwhale.com/jaseweston/papers/ebrm_mlj.pdf
    
    """

    def __init__(self, config):
        super(SME_BL, self).__init__(config)
        self.config = config
        self.model_name = 'SME_Bilinear'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    def gu_bilinear(self, h, r):
        """Function to calculate bilinear loss.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.

            Returns:
                Tensors: Returns the bilinear loss.
        """
        mu1h = tf.matmul(self.mu1, tf.transpose(h)) # [k, b]
        mu2r = tf.matmul(self.mu2, tf.transpose(r)) # [k, b]
        return tf.transpose(mu1h * mu2r + self.bu)  # [b, k]

    def gv_bilinear(self, r, t):
        """Function to calculate bilinear loss.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.

            Returns:
                Tensors: Returns the bilinear loss.
        """
        mv1t = tf.matmul(self.mv1, tf.transpose(t)) # [k, b]
        mv2r = tf.matmul(self.mv2, tf.transpose(r)) # [k, b]
        return tf.transpose(mv1t * mv2r + self.bv)  # [b, k]

    def forward(self, h, r, t):
        """Function to that performs semanting matching.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.
                t (Tensor): Tail ids of the triple.

            Returns:
                Tensors: Returns the semantic matchin score.
        """
        h_e, r_e, t_e = self.embed(h, r, t)
        norm_h = tf.nn.l2_normalize(h_e, -1)
        norm_r = tf.nn.l2_normalize(r_e, -1)
        norm_t = tf.nn.l2_normalize(t_e, -1)

        return tf.reduce_sum(self.gu_bilinear(norm_h, norm_r)*self.gv_bilinear(norm_r, norm_t), -1)