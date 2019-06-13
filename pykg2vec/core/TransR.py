#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from pykg2vec.core.KGMeta import ModelMeta


class TransR(ModelMeta):
    """ `Learning Entity and Relation Embeddings for Knowledge Graph Completion`_

        TranR is a translation based knowledge graph embedding method. Similar to TransE and TransH, it also
        builds entity and relation embeddings by regarding a relation as translation from head entity to tail
        entity. However, compared to them, it builds the entity and relation embeddings in a separate entity
        and relation spaces.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.core.TransR import TransR
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransR()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on thunlp_.
         .. _thunlp:
             https://github.com/thunlp/TensorFlow-TransX/blob/master/transR.py

        .. _Learning Entity and Relation Embeddings for Knowledge Graph Completion:
            http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf
    """

    def __init__(self, config):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.model_name = 'TransR'

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
              rel_matrix   (Tensor Variable): Weight matrix for transformation of entity embeddings.
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

            rel_matrix = np.zeros([num_total_rel, d * k], dtype=np.float32)

            for i in range(num_total_rel):
                for j in range(k):
                    for z in range(d):
                        if j == z:
                            rel_matrix[i][j * d + z] = 1.0

            self.rel_matrix = tf.Variable(rel_matrix, name="rel_matrix")

            self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.rel_matrix]

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

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

        pos_matrix = self.get_transform_matrix(self.test_r_batch)
        pos_matrix = tf.reshape(pos_matrix, [-1, self.config.ent_hidden_size])
        project_ent_embedding = self.transform(self.ent_embeddings, tf.transpose(pos_matrix))
        project_ent_embedding = tf.reshape(project_ent_embedding,
                                           [self.config.batch_size_testing, -1, self.config.rel_hidden_size])

        project_ent_embedding = tf.nn.l2_normalize(project_ent_embedding, axis=2)

        score_head = self.distance(project_ent_embedding,
                                   tf.expand_dims(rel_vec, axis=1),
                                   tf.expand_dims(tail_vec, axis=1), axis=2)
        score_tail = self.distance(tf.expand_dims(head_vec, axis=1),
                                   tf.expand_dims(rel_vec, axis=1),
                                   project_ent_embedding, axis=2)

        _, head_rank = tf.nn.top_k(score_head, k=num_total_ent)
        _, tail_rank = tf.nn.top_k(score_tail, k=num_total_ent)

        return head_rank, tail_rank

    def transform(self, matrix, embeddings):
        """Performs transformation of the embeddings into relation space.

            Args:
              matrix (Tensor): Transformation matrix
              embeddings( Tensor): Embeddings to be transformd

            Returns:
               Tensors: Returns Transformed Matrix.
        """
        return tf.matmul(matrix, embeddings)

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

    def get_transform_matrix(self, r):
        """gets the transformation matrix."""
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size
        return tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, r), [-1, k, d])

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, h), [-1, d, 1])
        r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, r), [-1, k])
        t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, t), [-1, d, 1])
        matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, r), [-1, k, d])

        transform_h_e = self.transform(matrix, h_e)
        transform_t_e = self.transform(matrix, t_e)
        h_e = tf.nn.l2_normalize(tf.reshape(transform_h_e, [-1, k]), -1)
        r_e = tf.nn.l2_normalize(tf.reshape(r_e, [-1, k]), -1)
        t_e = tf.nn.l2_normalize(tf.reshape(transform_t_e, [-1, k]), -1)

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
