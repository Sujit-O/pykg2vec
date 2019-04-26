#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.append("../")
import tensorflow as tf
from core.KGMeta import ModelMeta


class RotatE(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    ---
    ------------------Paper Authors---------------------------
    ---
    ------------------Summary---------------------------------
    ---
    """

    def __init__(self, config=None, data_handler=None):
        self.config = config
        self.data_handler = data_handler
        self.model_name = 'RotatE'

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
        num_total_ent = self.data_handler.tot_entity
        num_total_rel = self.data_handler.tot_relation

        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings_real = tf.get_variable(name="ent_embeddings_real", shape=[num_total_ent, k],
                                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.ent_embeddings_imag = tf.get_variable(name="ent_embeddings_imag", shape=[num_total_ent, k],
                                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings_real = tf.get_variable(name="rel_embeddings_real", shape=[num_total_rel, k],
                                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.ent_embeddings_real, self.ent_embeddings_imag, self.rel_embeddings_real]

    def comp_mul_and_min(self, hr, hi, rr, ri, tr, ti):
        score_r = hr * rr - hi * ri - tr
        score_i = hr * ri + hi * rr - ti
        return tf.reduce_sum(tf.sqrt(score_r ** 2 + score_i ** 2), -1)

    def def_loss(self):
        (pos_h_e_r, pos_h_e_i), (pos_r_e_r, pos_r_e_i), (pos_t_e_r, pos_t_e_i) \
            = self.embed(self.pos_h, self.pos_r, self.pos_t)

        (neg_h_e_r, neg_h_e_i), (neg_r_e_r, neg_r_e_i), (neg_t_e_r, neg_t_e_i) \
            = self.embed(self.neg_h, self.neg_r, self.neg_t)

        pos_score = self.comp_mul_and_min(pos_h_e_r, pos_h_e_i, pos_r_e_r, pos_r_e_i, pos_t_e_r, pos_t_e_i)
        neg_score = self.comp_mul_and_min(neg_h_e_r, neg_h_e_i, neg_r_e_r, neg_r_e_i, neg_t_e_r, neg_t_e_i)

        self.loss = tf.reduce_sum(tf.maximum(pos_score + self.config.margin - neg_score, 0))

    def test_step(self):
        num_entity = self.data_handler.tot_entity

        (h_vec_r, h_vec_i), (r_vec_r, r_vec_i), (t_vec_r, t_vec_i) \
            = self.embed(self.test_h, self.test_r, self.test_t)

        head_pos_score = self.comp_mul_and_min(self.ent_embeddings_real, self.ent_embeddings_imag, \
                                               r_vec_r, r_vec_i, t_vec_r, t_vec_i)

        tail_pos_score = self.comp_mul_and_min(h_vec_r, h_vec_i, r_vec_r, r_vec_i,
                                               self.ent_embeddings_real, self.ent_embeddings_imag)

        _, head_rank = tf.nn.top_k(head_pos_score, k=num_entity)
        _, tail_rank = tf.nn.top_k(tail_pos_score, k=num_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        pi = 3.14159265358979323846
        h_e_r = tf.nn.embedding_lookup(self.ent_embeddings_real, h)
        h_e_i = tf.nn.embedding_lookup(self.ent_embeddings_imag, h)
        r_e_r = tf.nn.embedding_lookup(self.rel_embeddings_real, r)
        t_e_r = tf.nn.embedding_lookup(self.ent_embeddings_real, t)
        t_e_i = tf.nn.embedding_lookup(self.ent_embeddings_imag, t)
        r_e_r = r_e_r / pi
        r_e_i = tf.sin(r_e_r)
        r_e_r = tf.cos(r_e_r)
        return (h_e_r, h_e_i), (r_e_r, r_e_i), (t_e_r, t_e_i)

    def get_embed(self, h, r, t, sess=None):
        """function to get the embedding value in numpy"""
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess):
        """function to get the projected embedding value in numpy"""
        return self.get_embed(h, r, t, sess)
