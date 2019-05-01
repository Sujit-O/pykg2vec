#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.append("../")
import tensorflow as tf
from core.KGMeta import ModelMeta
import pickle

class NTN(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    ---
    ------------------Paper Authors---------------------------
    ---
    ------------------Summary---------------------------------
    https://github.com/siddharth-agrawal/Neural-Tensor-Network/blob/master/neuralTensorNetwork.py
    ---
    """

    def __init__(self, config=None):
        self.config = config
        with open(str(self.config.tmp_data / 'data_stats.pkl'), 'rb') as f:
            self.data_stats = pickle.load(f)
        self.model_name = 'NTN'

        self.def_inputs()
        self.def_parameters()
        self.def_loss()

    def def_inputs(self):
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])
        self.test_h = tf.placeholder(tf.int32, [1])
        self.test_t = tf.placeholder(tf.int32, [1])
        self.test_r = tf.placeholder(tf.int32, [1])

    def def_parameters(self):
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
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        energy_pos = tf.reduce_sum(pos_r_e * self.train_layer(pos_h_e, pos_t_e), -1)
        energy_neg = tf.reduce_sum(neg_r_e * self.train_layer(neg_h_e, neg_t_e), -1)

        self.loss = tf.reduce_sum(tf.maximum(energy_neg + self.config.margin - energy_pos, 0))

    def train_layer(self, h, t):
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

    # Loop over ret_hidden_size
    def test_layer(self, h, t, expand=None):
        k = self.config.rel_hidden_size
        # h => [m, d], self.mr1 => [d, k]
        mr1h = tf.matmul(h, self.mr1)
        # t => [m, d], self.mr2 => [d, k]
        mr2t = tf.matmul(t, self.mr2)
        # br = [k]
        br = tf.squeeze(self.br, -1)

        # [m, 1, d]
        expanded_h = tf.expand_dims(h, 1)
        # [m, d, 1]
        expanded_t = tf.expand_dims(t, 2)

        if expand == "t":
            size = tf.shape(h)[0]
            expanded_t = tf.tile(expanded_t, [size, 1, 1])

        elif expand == "h":
            size = tf.shape(t)[0]
            expanded_h = tf.tile(expanded_h, [size, 1, 1])

        def condition(i, outputs):
            return tf.less(i, k)

        def body(i, outputs):
            # self.mr[i]: [d, d], mr_prime: [m, d, d]
            mr_prime = tf.tile(tf.expand_dims(self.mr[i], 0), [size, 1, 1])
            # [m, 1, 1]
            htmrt_index = tf.squeeze(tf.matmul(tf.matmul(expanded_h, mr_prime), expanded_t), [1, 2])

            outputs = outputs.write(i, tf.expand_dims(htmrt_index, 0))
            return [tf.add(i, 1), outputs]

        i = tf.constant(0)
        outputs = tf.TensorArray(dtype=tf.float32, infer_shape=True, size=1, dynamic_size=True)

        _, outputs = tf.while_loop(condition, body, [i, outputs])

        htmrt = outputs.concat()

        htmrt = tf.transpose(htmrt)

        return tf.tanh(mr1h + mr2t + br + htmrt)

    def test_step(self):
        num_entity = self.data_stats.tot_entity

        h_vec, r_vec, t_vec = self.embed(self.test_h, self.test_r, self.test_t)
        energy_h = tf.reduce_sum(r_vec * self.test_layer(tf.nn.l2_normalize(self.ent_embeddings, axis=1),
                                                         t_vec, expand='t'), -1)
        energy_t = tf.reduce_sum(r_vec * self.test_layer(h_vec, tf.nn.l2_normalize(self.ent_embeddings, axis=1),
                                                         expand='h'), -1)

        _, head_rank = tf.nn.top_k(tf.negative(energy_h), k=num_entity)
        _, tail_rank = tf.nn.top_k(tf.negative(energy_t), k=num_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        emb_h = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), h)
        emb_r = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.rel_embeddings, axis=1), r)
        emb_t = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), t)
        return emb_h, emb_r, emb_t

    def get_embed(self, h, r, t, sess=None):
        """function to get the embedding value in numpy"""
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess):
        """function to get the projected embedding value in numpy"""
        return self.get_embed(h, r, t, sess)
