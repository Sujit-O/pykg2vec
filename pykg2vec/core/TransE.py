#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


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
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on OpenKE_ and wencolani_.

        .. _OpenKE:
            https://github.com/thunlp/OpenKE/blob/master/models/TransE.py

        .. _wencolani:
            https://github.com/wencolani/TransE.git

        .. _Translating Embeddings for Modeling Multi-relational Data:
            http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela
    """

    def __init__(self, config=None):
        self.config = config
        self.model_name = 'TransE'

    def def_inputs(self):
        """Defines the inputs to the model.

           Attributes:
              pos_h (Tensor): Positive Head entities ids.
              pos_r (Tensor): Positive Relation ids of the triple.
              pos_t (Tensor): Positive Tail entity ids of the triple.
              neg_h (Tensor): Negative Head entities ids.
              neg_r (Tensor): Negative Relation ids of the triple.
              neg_t (Tensor): Negative Tail entity ids of the triple.
              test_h_batch (Tensor): Batch of head ids for testing.
              test_r_batch (Tensor): Batch of relation ids for testing
              test_t_batch (Tensor): Batch of tail ids for testing.
        """
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])
        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings (Tensor Variable): Lookup variable containing  embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing  embedding of the relations.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.ent_embeddings, self.rel_embeddings]

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        score_pos = self.distance(pos_h_e, pos_r_e, pos_t_e)
        score_neg = self.distance(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(score_pos + self.config.margin - score_neg, 0))

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

          Returns:
              Tensors: Returns ranks of head and tail.
        """
        head_vec, rel_vec, tail_vec = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        score_head = self.distance(norm_ent_embeddings,
                                   tf.expand_dims(rel_vec, axis=1),
                                   tf.expand_dims(tail_vec, axis=1), axis=2)
        score_tail = self.distance(tf.expand_dims(head_vec, axis=1),
                                   tf.expand_dims(rel_vec, axis=1),
                                   norm_ent_embeddings, axis=2)

        _, head_rank = tf.nn.top_k(score_head, k=self.config.kg_meta.tot_entity)
        _, tail_rank = tf.nn.top_k(score_tail, k=self.config.kg_meta.tot_entity)

        return head_rank, tail_rank

    def distance(self, h, r, t, axis=1):
        """Function to calculate distance measure in embedding space.

           Args:
              h (Tensor): Head entities ids.
              r (Tensor): Relation ids of the triple.
              t (Tensor): Tail entity ids of the triple.
              axis (int): Determines the axis for reduction

            Returns:
               Tensors: Returns the distance measure.
        """
        if self.config.L1_flag:
            return tf.reduce_sum(tf.abs(h + r - t), axis=axis)  # L1 norm
        else:
            return tf.reduce_sum((h + r - t) ** 2, axis=axis)  # L2 norm

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        emb_h = tf.nn.embedding_lookup(norm_ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(norm_ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def get_embed(self, h, r, t, sess):
        """Function to get the embedding value in numpy.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess=None):
        """"Function to get the projected embedding value in numpy.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
         """
        return self.get_embed(h, r, t, sess)
