#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class Rescal(ModelMeta):
    """`A Three-Way Model for Collective Learning on Multi-Relational Data`_

        RESCAL is a tensor factorization approach to knowledge representation learning,
        which is able to perform collective learning via the latent components of the factorization.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.core.Rescal import Rescal
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = Rescal()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on mnick_ and OpenKE_.
         .. _mnick:
             https://github.com/mnick/rescal.py/blob/master/rescal/rescal.py

         .. _OpenKE:
             https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py
    """

    def __init__(self, config):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.model_name = 'Rescal'

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
        with tf.name_scope("read_inputs"):
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
               ent_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
               rel_matrices  (Tensor Variable): Transformation matrices for entities into relation space.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            # A: per each entity, store its embedding representation.
            self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                                  shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            # M: per each relation, store a matrix that models the interactions between entity embeddings.
            self.rel_matrices = tf.get_variable(name="rel_matrices",
                                                shape=[num_total_rel, k * k],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.ent_embeddings, self.rel_matrices]

    def cal_truth_val(self, h, r, t):
        """Function to calculate truth value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns Tensors.
        """
        # dim of h: [m, k, 1]
        #        r: [m, k, k]
        #        t: [m, k, 1]
        return tf.reduce_sum(h * tf.matmul(r, t), [1, 2])

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        k = self.config.hidden_size

        with tf.name_scope('normalization'):
            self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
            self.rel_matrices = tf.nn.l2_normalize(self.rel_matrices, axis=1)

        with tf.name_scope('lookup_embeddings'):
            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_r_e = tf.nn.embedding_lookup(self.rel_matrices, self.pos_r)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_r_e = tf.nn.embedding_lookup(self.rel_matrices, self.neg_r)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)

        with tf.name_scope('reshaping'):
            pos_h_e = tf.reshape(pos_h_e, [-1, k, 1])
            pos_r_e = tf.reshape(pos_r_e, [-1, k, k])
            pos_t_e = tf.reshape(pos_t_e, [-1, k, 1])
            neg_h_e = tf.reshape(neg_h_e, [-1, k, 1])
            neg_r_e = tf.reshape(neg_r_e, [-1, k, k])
            neg_t_e = tf.reshape(neg_t_e, [-1, k, 1])

        pos_score = self.cal_truth_val(pos_h_e, pos_r_e, pos_t_e)
        neg_score = self.cal_truth_val(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(neg_score + self.config.margin - pos_score, 0))

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        num_entity = self.data_stats.tot_entity
        k = self.config.hidden_size

        h_vec, r_vec, t_vec = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        h_sim = tf.tensordot(tf.squeeze(tf.matmul(r_vec, t_vec), axis=-1), self.ent_embeddings, axes=((-1), (-1)))
        t_sim = tf.squeeze(tf.tensordot(tf.matmul(tf.reshape(h_vec, [-1, 1, k]), r_vec),
                                        self.ent_embeddings, axes=((-1), (-1))), axis=1)

        _, head_rank = tf.nn.top_k(tf.negative(h_sim), k=num_entity)
        _, tail_rank = tf.nn.top_k(tf.negative(t_sim), k=num_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        k = self.config.hidden_size
        emb_h = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), h)
        emb_r = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.rel_matrices, axis=1), r)
        emb_t = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), t)
        #
        emb_h = tf.reshape(emb_h, [-1, k, 1])
        emb_r = tf.reshape(emb_r, [-1, k, k])
        emb_t = tf.reshape(emb_t, [-1, k, 1])

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

    def get_proj_embed(self, h, r, t, sess):
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
