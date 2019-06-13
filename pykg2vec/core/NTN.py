#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class NTN(ModelMeta):
    """ `Reasoning With Neural Tensor Networks for Knowledge Base Completion`_

    It is a neural tensor network which represents entities as an average of their
    constituting word vectors. It then projects entities to their vector embeddings
    in the input layer. The two entities are then combined and mapped to a non-linear hidden layer.
    https://github.com/siddharth-agrawal/Neural-Tensor-Network/blob/master/neuralTensorNetwork.py

     Args:
        config (object): Model configuration parameters.

     Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

     Examples:
        >>> from pykg2vec.core.NTN import NTN
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = NTN()
        >>> trainer = Trainer(model=model, debug=False)
        >>> trainer.build_model()
        >>> trainer.train_model()

     Portion of the code based on `this Source`_.
     .. _this Source:
         https://github.com/siddharth-agrawal/Neural-Tensor-Network/blob/master/neuralTensorNetwork.py

     .. _Reasoning With Neural Tensor Networks for Knowledge Base Completion:
         https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.model_name = 'NTN'

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
                k (Tensor): Size of the latent dimension for entities.
                d (Tensor): Size of the latent dimension for relations.
                ent_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
                rel_embeddings  (Tensor Variable): Lookup variable containing embedding of the relations.
                mr1 (Tensor): Tensor Matrix for transforming head entity.
                mr2 (Tensor): Tensor Matrix for transforming tail entity.
                br (Tensor): Tensor Matrix for adding bias
                mr (Tensor): Tensor Matrix for transforming entities.
                parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, d],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        with tf.name_scope("weights_and_parameters"):
            self.mr1 = tf.get_variable(name="mr1", shape=[d, k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.mr2 = tf.get_variable(name="mr2", shape=[d, k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.br = tf.get_variable(name="br", shape=[k, 1],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.mr = tf.get_variable(name="mr", shape=[k, d, d],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, \
                               self.mr1, self.mr2, self.br, self.mr]

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        energy_pos = tf.reduce_sum(pos_r_e * self.train_layer(pos_h_e, pos_t_e), -1)
        energy_neg = tf.reduce_sum(neg_r_e * self.train_layer(neg_h_e, neg_t_e), -1)

        self.loss = tf.reduce_sum(tf.maximum(energy_neg + self.config.margin - energy_pos, 0))

    def train_layer(self, h, t):
        """Defines the forward pass training layers of the algorithm.

           Args:
               h (Tensor): Head entities ids.
               t (Tensor): Tail entity ids of the triple.
        """
        k = self.config.rel_hidden_size
        # h => [m, d], self.mr1 => [d, k]
        mr1h = tf.matmul(h, self.mr1)
        # t => [m, d], self.mr2 => [d, k]
        mr2t = tf.matmul(t, self.mr2)
        # br = [k]
        br = tf.squeeze(self.br, -1)

        # [m, k, 1, d]
        expanded_h = tf.tile(tf.expand_dims(tf.expand_dims(h, 1), 1), [1, k, 1, 1])

        # [m, k, d, d]
        expanded_mr = tf.tile(tf.expand_dims(self.mr, 0), [tf.shape(h)[0], 1, 1, 1])

        # [m, k, d, 1]
        expanded_t = tf.tile(tf.expand_dims(tf.expand_dims(t, 1), 3), [1, k, 1, 1])

        # [m, k]
        htmrt = tf.squeeze(tf.matmul(tf.matmul(expanded_h, expanded_mr), expanded_t), [2, 3])

        return tf.tanh(mr1h + mr2t + br + htmrt)

    def test_layer(self, h, t):
        """Defines the forward pass inference layers of the algorithm.

           Args:
               h (Tensor): Head entities ids.
               t (Tensor): Tail entity ids of the triple.
        """
        num_entity = self.data_stats.tot_entity
        # h => [m, d], self.mr1 => [d, k]
        mr1h = tf.matmul(h, self.mr1)
        # t => [m, d], self.mr2 => [d, k]
        mr2t = tf.matmul(t, self.mr2)
        # br = [k]
        br = tf.squeeze(self.br, -1)
        htmrt = tf.cond(tf.shape(h)[0] > tf.shape(t)[0],
                        lambda: tf.tensordot(t, tf.tensordot(h, self.mr, axes=((-1), (-1))), axes=((-1), (-1))),
                        lambda: tf.tensordot(h, tf.tensordot(t, self.mr, axes=((-1), (-1))), axes=((-1), (-1))))
        mr1h = tf.cond(tf.shape(mr1h)[0] < num_entity, lambda: tf.expand_dims(mr1h, axis=1), lambda: mr1h)
        mr2t = tf.cond(tf.shape(mr2t)[0] < num_entity, lambda: tf.expand_dims(mr2t, axis=1), lambda: mr2t)

        return tf.tanh(mr1h + mr2t + br + htmrt)

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        num_entity = self.data_stats.tot_entity

        h_vec, r_vec, t_vec = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        energy_h = tf.reduce_sum(
            tf.expand_dims(r_vec, axis=1) * self.test_layer(tf.nn.l2_normalize(self.ent_embeddings, axis=1), t_vec), -1)
        energy_t = tf.reduce_sum(
            tf.expand_dims(r_vec, axis=1) * self.test_layer(h_vec, tf.nn.l2_normalize(self.ent_embeddings, axis=1)), -1)

        _, head_rank = tf.nn.top_k(tf.negative(energy_h), k=num_entity)
        _, tail_rank = tf.nn.top_k(tf.negative(energy_t), k=num_entity)

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
        emb_h = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), h)
        emb_r = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.rel_embeddings, axis=1), r)
        emb_t = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), t)
        return emb_h, emb_r, emb_t

    def get_embed(self, h, r, t, sess=None):
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
        """Function to get the projected embedding value in numpy.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.
                t (Tensor): Tail entity ids of the triple.
                sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        return self.get_embed(h, r, t, sess)
