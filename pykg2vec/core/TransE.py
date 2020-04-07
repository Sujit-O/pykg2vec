#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

class TransE(ModelMeta):
    """ `Translating Embeddings for Modeling Multi-relational Data`_

        TransE is an energy based model which represents the
        relationships as translations in the embedding space. Which
        means that if (h,l,t) holds then the embedding of the tail
        't' should be close to the embedding of head entity 'h'
        plus some vector that depends on the relationship 'l'.
        Both entities and relations are vectors in the same space.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.TransE import TransE
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransE()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on `OpenKE_TransE`_ and `wencolani`_.

        .. _OpenKE_TransE: https://github.com/thunlp/OpenKE/blob/master/models/TransE.py

        .. _wencolani: https://github.com/wencolani/TransE.git

        .. _Translating Embeddings for Modeling Multi-relational Data:
            http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela
    """

    def __init__(self, config):
        super(TransE, self).__init__()

        self.config = config
        self.model_name = 'TransE'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               ent_embeddings (Tensor Variable): Lookup variable containing  embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing  embedding of the relations.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        initializer = tf.initializers.glorot_normal()

        self.ent_embeddings = tf.Variable(initializer(shape=(num_total_ent, self.config.hidden_size)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(initializer(shape=(num_total_rel, self.config.hidden_size)), name="rel_embedding")
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings]

    def forward(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids.
               t (Tensor): Tail entity ids.

            Returns:
                Tensors: the scores of evaluationReturns head, relation and tail embedding Tensors.
        """
        h_e, r_e, t_e = self.embed(h, r, t)

        norm_h_e = tf.nn.l2_normalize(h_e, -1)
        norm_r_e = tf.nn.l2_normalize(r_e, -1)
        norm_t_e = tf.nn.l2_normalize(t_e, -1)

        if self.config.L1_flag:
            return tf.reduce_sum(tf.math.abs(norm_h_e + norm_r_e - norm_t_e), -1) # L1 norm 
        else:
            return tf.reduce_sum(tf.math.square(norm_h_e + norm_r_e - norm_t_e), -1) # L2 norm

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids.
               t (Tensor): Tail entity ids.

            Returns:
                Tensors: Returns a tuple of head, relation and tail embedding Tensors.
        """
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)

        return emb_h, emb_r, emb_t