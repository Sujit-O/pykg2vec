#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from pykg2vec.core.KGMeta import ModelMeta


class Rescal(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    A Three-Way Model for Collective Learning on Multi-Relational Data
    ------------------Paper Authors---------------------------
    Maximilian Nickel, Volker Tresp, Hans-Peter Kriegel
    Ludwig-Maximilians-Universitat, Munich, Germany 
    Siemens AG, Corporate Technology, Munich, Germany
    {NICKEL@CIP.IFI.LMU.DE, VOLKER.TRESP@SIEMENS.COM, KRIEGEL@DBS.IFI.LMU.DE}
    ------------------Summary---------------------------------
    RESCAL is a tensor factorization approach to knowledge representation learning, 
    which is able to perform collective learning via the latent components of the factorization.
    
    Portion of Code Based on https://github.com/mnick/rescal.py/blob/master/rescal/rescal.py
     and https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py
    """

    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.model_name = 'Rescal'

        self.def_inputs()
        self.def_parameters()
        self.def_loss()

    def def_inputs(self):
        with tf.name_scope("read_inputs"):
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
        # dim of h: [m, k, 1]
        #        r: [m, k, k]
        #        t: [m, k, 1]
        return tf.reduce_sum(h * tf.matmul(r, t), [1, 2])

    def def_loss(self):
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

    def test_step(self):
        k = self.config.hidden_size
        num_entity = self.data_handler.tot_entity

        h_vec, r_vec, t_vec = self.embed(self.test_h, self.test_r, self.test_t)

        h_sim = tf.matmul(self.ent_embeddings, tf.matmul(r_vec, t_vec))
        t_sim = tf.transpose(tf.matmul(tf.matmul(tf.transpose(h_vec), r_vec), tf.transpose(self.ent_embeddings)))

        _, head_rank = tf.nn.top_k(tf.reduce_sum(tf.negative(h_sim), 1), k=num_entity)
        _, tail_rank = tf.nn.top_k(tf.reduce_sum(tf.negative(t_sim), 1), k=num_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        k = self.config.hidden_size
        emb_h = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), h)
        emb_r = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.rel_matrices, axis=1), r)
        emb_t = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), t)

        emb_h = tf.reshape(emb_h, [k, 1])
        emb_r = tf.reshape(emb_r, [k, k])
        emb_t = tf.reshape(emb_t, [k, 1])

        return emb_h, emb_r, emb_t

    def get_embed(self, h, r, t, sess):
        """function to get the embedding value in numpy"""
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess):
        """function to get the projectd embedding value in numpy"""
        return self.get_embed(h, r, t, sess)
