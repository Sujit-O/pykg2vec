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

class SLM(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    ---
    ------------------Paper Authors---------------------------
    ---
    ------------------Summary---------------------------------
    ---
    """

    def __init__(self, config=None):
        self.config = config
        with open(self.config.tmp_data / 'data_stats.pkl', 'rb') as f:
            self.data_stats = pickle.load(f)
        self.model_name = 'SLM'

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

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.mr1, self.mr2]

    def def_loss(self):
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        energy_pos = tf.reduce_sum(pos_r_e * self.layer(pos_h_e, pos_t_e), -1)
        energy_neg = tf.reduce_sum(neg_r_e * self.layer(neg_h_e, neg_t_e), -1)

        self.loss = tf.reduce_sum(tf.maximum(energy_neg + self.config.margin - energy_pos, 0))

    def layer(self, h, t):
        k = self.config.rel_hidden_size
        # h => [m, d], self.mr1 => [d, k]
        mr1h = tf.matmul(h, self.mr1)
        # t => [m, d], self.mr2 => [d, k]
        mr2t = tf.matmul(t, self.mr2)

        return tf.tanh(mr1h + mr2t)

    def test_step(self):
        num_entity = self.data_stats.tot_entity

        h_vec, r_vec, t_vec = self.embed(self.test_h, self.test_r, self.test_t)
        energy_h = tf.reduce_sum(r_vec * self.layer(tf.nn.l2_normalize(self.ent_embeddings, axis=1), t_vec), -1)
        energy_t = tf.reduce_sum(r_vec * self.layer(h_vec, tf.nn.l2_normalize(self.ent_embeddings, axis=1)), -1)

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
