#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class TransD(ModelMeta):
    """ `Knowledge Graph Embedding via Dynamic Mapping Matrix`_

        TransD constructs a dynamic mapping matrix for each entity-relation pair by considering the diversity of entities and relations simultaneously.
        Compared with TransR/CTransR, TransD has fewer parameters and has no matrix vector multiplication.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.core.TransD import TransD
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransD()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on OpenKE_.
        .. _OpenKE:
            https://github.com/thunlp/OpenKE/blob/master/models/TransD.py

        .. _Knowledge Graph Embedding via Dynamic Mapping Matrix:
            https://www.aclweb.org/anthology/P15-1067
    """

    def __init__(self, config):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.model_name = 'TransD'

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
               k (Tensor): Size of the latent dimesnion for relations.
               d (Tensor): Size of the latent dimesnion for entities .
               ent_embeddings (Tensor Variable): Lookup variable containing  embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing  embedding of the relations.

               ent_mappings (Tensor Variable): Lookup variable containing  mapping for entities.
               rel_mappings   (Tensor Variable): Lookup variable containing   mapping for relations.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                                  shape=[num_total_ent, d],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding",
                                                  shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.ent_mappings = tf.get_variable(name="ent_mappings",
                                                shape=[num_total_ent, d],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_mappings = tf.get_variable(name="rel_mappings",
                                                shape=[num_total_rel, k],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.ent_mappings, self.rel_mappings]

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        pos_h_m, pos_r_m, pos_t_m = self.get_mapping(self.pos_h, self.pos_r, self.pos_t)
        neg_h_m, neg_r_m, neg_t_m = self.get_mapping(self.neg_h, self.neg_r, self.neg_t)

        pos_h_e = tf.nn.l2_normalize(pos_h_e + tf.reduce_sum(pos_h_e * pos_h_m, -1, keepdims=True) * pos_r_m, -1)
        pos_r_e = tf.nn.l2_normalize(pos_r_e, -1)
        pos_t_e = tf.nn.l2_normalize(pos_t_e + tf.reduce_sum(pos_t_e * pos_t_m, -1, keepdims=True) * pos_r_m, -1)

        neg_h_e = tf.nn.l2_normalize(neg_h_e + tf.reduce_sum(neg_h_e * neg_h_m, -1, keepdims=True) * neg_r_m, -1)
        neg_r_e = tf.nn.l2_normalize(neg_r_e, -1)
        neg_t_e = tf.nn.l2_normalize(neg_t_e + tf.reduce_sum(neg_t_e * neg_t_m, -1, keepdims=True) * neg_r_m, -1)

        score_pos = self.distance(pos_h_e, pos_r_e, pos_t_e)
        score_neg = self.distance(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(score_pos - score_neg + self.config.margin, 0))

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

           Returns:
               Tensors: Returns ranks of head and tail.
       """
        num_total_ent = self.data_stats.tot_entity
        head_vec, rel_vec, tail_vec = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)
        h_m, r_m, t_m = self.get_mapping(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        head_vec = tf.nn.l2_normalize(head_vec + tf.reduce_sum(head_vec * h_m, -1, keepdims=True) * r_m, -1)
        rel_vec = tf.nn.l2_normalize(rel_vec, -1)
        tail_vec = tf.nn.l2_normalize(tail_vec + tf.reduce_sum(tail_vec * t_m, -1, keepdims=True) * r_m, -1)

        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, -1)
        norm_ent_mappings = tf.nn.l2_normalize(self.ent_mappings, -1)

        project_ent_embedding = tf.nn.l2_normalize(
            norm_ent_embeddings + tf.reduce_sum(norm_ent_embeddings * norm_ent_mappings, -1, keepdims=True) * tf.expand_dims(r_m, axis=1), -1)

        score_head = self.distance(project_ent_embedding,
                                   tf.expand_dims(rel_vec, axis=1),
                                   tf.expand_dims(tail_vec, axis=1), axis=2)
        score_tail = self.distance(tf.expand_dims(head_vec, axis=1),
                                   tf.expand_dims(rel_vec, axis=1),
                                   project_ent_embedding, axis=2)

        _, head_rank = tf.nn.top_k(score_head, k=num_total_ent)
        _, tail_rank = tf.nn.top_k(score_tail, k=num_total_ent)

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

    def get_mapping(self, h, r, t):
        """Function to get the mapping for head, relation and tails.

          Args:
              h (Tensor): Head entities ids.
              r (Tensor): Relation ids of the triple.
              t (Tensor): Tail entity ids of the triple.

           Returns:
               Tensors: Returns the mapped values for head, relation and tail
        """
        h_m = tf.nn.embedding_lookup(self.ent_mappings, h)
        r_m = tf.nn.embedding_lookup(self.rel_mappings, r)
        t_m = tf.nn.embedding_lookup(self.ent_mappings, t)

        return h_m, r_m, t_m

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        h_e = tf.nn.embedding_lookup(self.ent_embeddings, h)
        r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)
        t_e = tf.nn.embedding_lookup(self.ent_embeddings, t)

        h_e = tf.nn.l2_normalize(h_e, -1)
        r_e = tf.nn.l2_normalize(r_e, -1)
        t_e = tf.nn.l2_normalize(t_e, -1)

        return h_e, r_e, t_e

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
        pos_h_e, pos_r_e, pos_t_e = self.embed(h, r, t)
        # temp
        pos_h_e, pos_r_e, pos_t_e = tf.squeeze(pos_h_e, 0), tf.squeeze(pos_r_e, 0), tf.squeeze(pos_t_e, 0)
        h, r, t = sess.run([pos_h_e, pos_r_e, pos_t_e])
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
