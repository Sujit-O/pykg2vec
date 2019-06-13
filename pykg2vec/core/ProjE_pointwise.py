#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class ProjE_pointwise(ModelMeta):
    """`ProjE-Embedding Projection for Knowledge Graph Completion`_.

        Instead of measuring the distance or matching scores between the pair of the
        head entity and relation and then tail entity in embedding space ((h,r) vs (t)).
        ProjE projects the entity candidates onto a target vector representing the
        input data. The loss in ProjE is computed by the cross-entropy between
        the projected target vector and binary label vector, where the included
        entities will have value 0 if in negative sample set and value 1 if in
        positive sample set.

         Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.ProjE_pointwise import ProjE_pointwise
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = ProjE_pointwise()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _ProjE-Embedding Projection for Knowledge Graph Completion:
            https://arxiv.org/abs/1611.05425

    """

    def __init__(self, config):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.model_name = 'ProjE_pointwise'

    def def_inputs(self):
        """Defines the inputs to the model.

           Attributes:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               hr_t (Tensor): Tail tensor list for (h,r) pair.
               rt_h (Tensor): Head tensor list for (r,t) pair.
               test_h_batch (Tensor): Batch of head ids for testing.
               test_r_batch (Tensor): Batch of relation ids for testing
               test_t_batch (Tensor): Batch of tail ids for testing.
        """
        self.h = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.hr_t = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])
        self.rt_h = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])

        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])


    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing embedding of the relations.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):

            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.bc1 = tf.get_variable(name="bc1", shape=[k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.De1 = tf.get_variable(name="De1", shape=[k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.Dr1 = tf.get_variable(name="Dr1", shape=[k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.bc2 = tf.get_variable(name="bc2", shape=[k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.De2 = tf.get_variable(name="De2", shape=[k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.Dr2 = tf.get_variable(name="Dr2", shape=[k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.bc1, self.De1, self.Dr1, self.bc2,
                                   self.De2, self.Dr2]

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, -1)  # [tot_ent, k]
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, -1)  # [tot_rel, k]

        emb_hr_h = tf.nn.embedding_lookup(norm_ent_embeddings, self.h)  # [m, k]
        emb_hr_r = tf.nn.embedding_lookup(norm_rel_embeddings, self.r)  # [m, k]

        emb_tr_t = tf.nn.embedding_lookup(norm_ent_embeddings, self.t)  # [m, k]
        emb_tr_r = tf.nn.embedding_lookup(norm_rel_embeddings, self.r)  # [m, k]

        hrt_sigmoid = self.g(tf.nn.dropout(self.f1(emb_hr_h, emb_hr_r), 0.5), norm_ent_embeddings)

        hrt_loss_left = - tf.reduce_sum((tf.log(tf.clip_by_value(hrt_sigmoid, 1e-10, 1.0)) * tf.maximum(0., self.hr_t)))
        hrt_loss_right = - tf.reduce_sum(
            (tf.log(tf.clip_by_value(1 - hrt_sigmoid, 1e-10, 1.0)) * tf.maximum(0., tf.negative(self.hr_t))))

        hrt_loss = hrt_loss_left + hrt_loss_right

        trh_sigmoid = self.g(tf.nn.dropout(self.f2(emb_tr_t, emb_tr_r), 0.5), norm_ent_embeddings)

        trh_loss_left = - tf.reduce_sum((tf.log(tf.clip_by_value(trh_sigmoid, 1e-10, 1.0)) * tf.maximum(0., self.rt_h)))
        trh_loss_right = - tf.reduce_sum(
            (tf.log(tf.clip_by_value(1 - trh_sigmoid, 1e-10, 1.0)) * tf.maximum(0., tf.negative(self.rt_h))))

        trh_loss = trh_loss_left + trh_loss_right

        regularizer_loss = tf.reduce_sum(tf.abs(self.De1) + tf.abs(self.Dr1)) + tf.reduce_sum(tf.abs(self.De2) + tf.abs(self.Dr2)) \
                            + tf.reduce_sum(tf.abs(self.ent_embeddings)) + tf.reduce_sum(tf.abs(self.rel_embeddings))
        self.loss = hrt_loss + trh_loss + regularizer_loss*1e-5

    def f1(self, h, r):
        """Defines froward layer for head.

            Args:
                   h (Tensor): Head entities ids.
                   r (Tensor): Relation ids of the triple.
        """
        return tf.tanh(h * self.De1 + r * self.Dr1 + self.bc1)

    def f2(self, t, r):
        """Defines forward layer for tail.

            Args:
               t (Tensor): Tail entities ids.
               r (Tensor): Relation ids of the triple.
        """
        return tf.tanh(t * self.De2 + r * self.Dr2 + self.bc2)

    def g(self, f, W):
        """Defines activation layer.

            Args:
               f (Tensor): output of the forward layers.
               W (Tensor): Matrix for multiplication.
        """
        return tf.sigmoid(tf.matmul(f, tf.transpose(W)))

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        num_entity = self.data_stats.tot_entity

        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, -1)  # [tot_ent, k]
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, -1)  # [tot_rel, k]

        h_vec = tf.nn.embedding_lookup(norm_ent_embeddings, self.test_h_batch)  # [1, k]
        r_vec = tf.nn.embedding_lookup(norm_rel_embeddings, self.test_r_batch)  # [1, k]
        t_vec = tf.nn.embedding_lookup(norm_ent_embeddings, self.test_t_batch)  # [1, k]

        hrt_sigmoid = - self.g(self.f1(h_vec, r_vec), norm_ent_embeddings)
        trh_sigmoid = - self.g(self.f2(t_vec, r_vec), norm_ent_embeddings)

        _, head_rank = tf.nn.top_k(trh_sigmoid, k=num_entity)
        _, tail_rank = tf.nn.top_k(hrt_sigmoid, k=num_entity)

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
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
        emb_h = tf.nn.l2_normalize(emb_h, axis=-1)
        emb_r = tf.nn.l2_normalize(emb_r, axis=-1)
        emb_t = tf.nn.l2_normalize(emb_t, axis=-1)

        proj_vec = self.get_proj(r)

        return self.projection(emb_h, proj_vec), emb_r, self.projection(emb_t, proj_vec)

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