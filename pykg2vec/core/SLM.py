#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class SLM(ModelMeta):
    """`Reasoning With Neural Tensor Networks for Knowledge Base Completion`_

        SLM model is designed as a baseline of Neural Tensor Network.
        The model constructs a nonlinear neural network to represent the score function.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.SLM import SLM
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = SLM()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _Reasoning With Neural Tensor Networks for Knowledge Base Completion:
            https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    """

    def __init__(self, config):
        super(SLM, self).__init__()
        self.config = config
        self.model_name = 'SLM'

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
                num_total_ent (int): Total number of entities.
                num_total_rel (int): Total number of relations.
                k (Tensor): Size of the latent dimension for entities.
                d (Tensor): Size of the latent dimension for relations.
                ent_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
                rel_embeddings  (Tensor Variable): Lookup variable containing embedding of the relations.
                mr1 (Tensor): Tensor Matrix for transforming head entity.
                mr2 (Tensor): Tensor Matrix for transforming tail entity.
                parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        emb_initializer = tf.initializers.glorot_normal()

        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, d)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embedding")
        self.mr1 = tf.Variable(emb_initializer(shape=(d, k)), name="mr1")
        self.mr2 = tf.Variable(emb_initializer(shape=(d, k)), name="mr2")

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.mr1, self.mr2]

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

    def dissimilarity(self, h, r, t):
        return -self.match(h, r, t) 

    def match(self, h, r, t, axis=-1):
        norm_h = tf.nn.l2_normalize(h, axis=axis)
        norm_r = tf.nn.l2_normalize(r, axis=axis)
        norm_t = tf.nn.l2_normalize(t, axis=axis)
        return tf.reduce_sum(norm_r * self.layer(norm_h, norm_t), axis=axis)

    def get_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """Defines the loss function for the algorithm."""
        pos_h_e, pos_r_e, pos_t_e = self.embed(pos_h, pos_r, pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(neg_h, neg_r, neg_t)

        energy_pos = self.match(pos_h_e, pos_r_e, pos_t_e)
        energy_neg = self.match(neg_h_e, neg_r_e, neg_t_e)

        loss = tf.reduce_sum(tf.maximum(energy_neg + self.config.margin - energy_pos, 0))

        return loss

    def layer(self, h, t):
        """Defines the forward pass layer of the algorithm.

          Args:
              h (Tensor): Head entities ids.
              t (Tensor): Tail entity ids of the triple.
        """       
        mr1h = tf.matmul(h, self.mr1) # h => [m, d], self.mr1 => [d, k]
        mr2t = tf.matmul(t, self.mr2) # t => [m, d], self.mr2 => [d, k]
        return tf.tanh(mr1h + mr2t)

    def predict(self, h, r, t, topk=-1):
        """Function that performs prediction for TransE. 
           shape of h can be either [num_tot_entity] or [1]. 
           shape of t can be either [num_tot_entity] or [1].

          Returns:
              Tensors: Returns ranks of head and tail.
        """
        h_e, r_e, t_e = self.embed(h, r, t)
        score = self.match(h_e, r_e, t_e)
        _, rank = tf.nn.top_k(-score, k=topk)

        return rank