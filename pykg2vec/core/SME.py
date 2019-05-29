#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
sys.path.append("../")
from core.KGMeta import ModelMeta
# from pykg2vec.core.KGMeta import ModelMeta


class SME(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    A Semantic Matching Energy Function for Learning with Multi-relational Data
    ------------------Paper Authors---------------------------
    Antoine Bordes
    Universite de Technologie de Compiegne – CNRS
    Heudiasyc UMR 7253
    Compiegne, France
    antoine.bordes@utc.fr
    Jason Weston
    Google
    111 8th avenue
    New York, NY, USA
    jweston@google.com
    Xaiver Glorot, Joshua Bengio
    Universite de Montreal
    Monteal, QC, Canada
    {glorotxa, bengioy}@iro.umontreal.ca
    ------------------Summary---------------------------------
    Semantic Matching Energy (SME) is an algorithm for embedding multi-relational data into vector spaces. 
    SME conducts semantic matching using neural network architectures. Given a fact (h, r, t), it first projects 
    entities and relations to their embeddings in the input layer. Later the relation r is combined with both h and t
    to get gu(h, r) and gv(r, t) in its hidden layer. The score is determined by calculating the matching score of gu and gv.

    There are two versions of SME: a linear version(SMELinear) as well as bilinear(SMEBilinear) version which differ in how the hidden layer is defined.

    Portion of Code Based on https://github.com/glorotxa/SME/blob/master/model.py
    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta

        if self.config.bilinear:
            self.model_name = 'SME_Bilinear'
        else:
            self.model_name = 'SME_Linear'

    def gu_bilinear(self, h, r, test_flag):
        if test_flag:
            tmp1 = tf.cond(tf.shape(h)[0] > tf.shape(r)[0],
                           lambda: tf.expand_dims(tf.transpose(tf.matmul(self.mu2, tf.transpose(r))), axis=1),
                           lambda: tf.transpose(tf.matmul(self.mu2, tf.transpose(r))))
            return tf.multiply(tf.transpose(tf.matmul(self.mu1, tf.transpose(h))), tmp1) + tf.squeeze(self.bu, axis=-1)
        else:
            return tf.transpose(
                tf.multiply(tf.matmul(self.mu1, tf.transpose(h)), tf.matmul(self.mu2, tf.transpose(r))) + self.bu)

    def gv_bilinear(self, r, t, test_flag):
        if test_flag:
            tmp1 = tf.cond(tf.shape(t)[0] > tf.shape(r)[0],
                           lambda: tf.expand_dims(tf.transpose(tf.matmul(self.mv1, tf.transpose(r))), axis=1),
                           lambda: tf.transpose(tf.matmul(self.mv1, tf.transpose(r))))
            return tf.multiply(tf.transpose(tf.matmul(self.mv2, tf.transpose(t))), tmp1) + tf.squeeze(self.bu, axis=-1)
        else:
            return tf.transpose(
                tf.multiply(tf.matmul(self.mv1, tf.transpose(r)), tf.matmul(self.mv2, tf.transpose(t))) + self.bv)

    def gu_linear(self, h, r, test_flag):
        if test_flag:
            tmp1 = tf.cond(tf.shape(h)[0] > tf.shape(r)[0],
                           lambda: tf.expand_dims(tf.transpose(tf.matmul(self.mu2, tf.transpose(r)) + self.bu), axis=1),
                           lambda: tf.transpose(tf.matmul(self.mu2, tf.transpose(r)) + self.bu))
            return tf.transpose(tf.matmul(self.mu1, tf.transpose(h))) + tmp1
        else:
            return tf.transpose(tf.matmul(self.mu1, tf.transpose(h)) + tf.matmul(self.mu2, tf.transpose(r)) + self.bu)

    def gv_linear(self, r, t, test_flag):
        if test_flag:
            tmp1 = tf.cond(tf.shape(t)[0] > tf.shape(r)[0],
                           lambda: tf.expand_dims(tf.transpose(tf.matmul(self.mv1, tf.transpose(r)) + self.bv), axis=1),
                           lambda: tf.transpose(tf.matmul(self.mv1, tf.transpose(r)) + self.bv))
            return tf.transpose(tf.matmul(self.mv2, tf.transpose(t))) + tmp1
        else:
            return tf.transpose(tf.matmul(self.mv1, tf.transpose(r)) + tf.matmul(self.mv2, tf.transpose(t)) + self.bv)

    def match(self, h, r, t, test_flag=False):
        if self.config.bilinear:
            if test_flag:
                tmp1 = self.gu_bilinear(h, r, test_flag)
                tmp2 = self.gv_bilinear(r, t, test_flag)
                result = tf.cond(tf.shape(tmp1)[1] < tf.shape(tmp2)[1],
                                 lambda: tf.reduce_sum(tf.multiply(tf.expand_dims(tmp1, axis=1), tmp2), -1),
                                 lambda: tf.reduce_sum(tf.multiply(tf.expand_dims(tmp2, axis=1), tmp1), -1))
                return result
            else:
                return tf.reduce_sum(tf.multiply(self.gu_bilinear(h, r, test_flag), self.gv_bilinear(r, t, test_flag)),
                                     1)
        else:
            if test_flag:
                tmp1 = self.gu_linear(h, r, test_flag)
                tmp2 = self.gv_linear(r, t, test_flag)
                result = tf.cond(tf.shape(tmp1)[1] < tf.shape(tmp2)[1],
                                 lambda: tf.reduce_sum(tf.expand_dims(tmp1, axis=1) * tmp2, -1),
                                 lambda: tf.reduce_sum(tf.expand_dims(tmp2, axis=1) * tmp1, -1))
                return result
            else:
                return tf.reduce_sum(self.gu_linear(h, r, test_flag) * self.gv_linear(r, t, test_flag), 1)

    def def_inputs(self):
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
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        with tf.name_scope("weights_and_parameters"):
            self.mu1 = tf.get_variable(name="mu1", shape=[k, k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.mu2 = tf.get_variable(name="mu2", shape=[k, k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.bu = tf.get_variable(name="bu", shape=[k, 1],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.mv1 = tf.get_variable(name="mv1", shape=[k, k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.mv2 = tf.get_variable(name="mv2", shape=[k, k],
                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.bv = tf.get_variable(name="bv", shape=[k, 1],
                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings,
                               self.mu1, self.mu2, self.bu, self.mv1, self.mv2, self.bv]

    def def_loss(self):
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)
        energy_pos = self.match(pos_h_e, pos_r_e, pos_t_e)
        energy_neg = self.match(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(energy_neg + self.config.margin - energy_pos, 0))

    def test_step(self):
        num_entity = self.data_stats.tot_entity

        h_vec, r_vec, t_vec = self.embed(self.test_h, self.test_r, self.test_t)
        energy_h = self.match(tf.nn.l2_normalize(self.ent_embeddings, axis=1), r_vec, t_vec)
        energy_t = self.match(h_vec, r_vec, tf.nn.l2_normalize(self.ent_embeddings, axis=1))

        _, head_rank = tf.nn.top_k(tf.negative(energy_h), k=num_entity)
        _, tail_rank = tf.nn.top_k(tf.negative(energy_t), k=num_entity)

        return head_rank, tail_rank

    def test_batch(self):
        num_entity = self.data_stats.tot_entity

        h_vec, r_vec, t_vec = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)
        energy_h = self.match(tf.nn.l2_normalize(self.ent_embeddings, axis=1), r_vec, t_vec, test_flag=True)
        energy_t = self.match(h_vec, r_vec, tf.nn.l2_normalize(self.ent_embeddings, axis=1), test_flag=True)

        _, head_rank = tf.nn.top_k(tf.negative(energy_h), k=num_entity)
        _, tail_rank = tf.nn.top_k(tf.negative(energy_t), k=num_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        emb_h = tf.nn.embedding_lookup(norm_ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(norm_ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def get_embed(self, h, r, t, sess=None):
        """function to get the embedding value in numpy"""
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess):
        """function to get the projected embedding value in numpy"""
        return self.get_embed(h, r, t, sess)
