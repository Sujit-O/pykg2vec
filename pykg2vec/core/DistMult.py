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


class DistMult(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    EMBEDDING ENTITIES AND RELATIONS FOR LEARNING AND INFERENCE IN KNOWLEDGE BASES
    ------------------Paper Authors---------------------------
    Bishan Yang1˚, Wen-tau Yih2
    , Xiaodong He2
    , Jianfeng Gao2 & Li Deng2
    1Department of Computer Science, Cornell University, Ithaca, NY, 14850, USA
    bishan@cs.cornell.edu
    2Microsoft Research, Redmond, WA 98052, USA
    {scottyih,xiaohe,jfgao,deng}@microsoft.com
    ------------------Summary---------------------------------
    DistMult is a simpler model comparing with RESCAL in that it simplifies
    the weight matrix used in RESCAL to a diagonal matrix. The scoring
    function used DistMult can capture the pairwise interactions between
     the head and the tail entities. However, DistMult has limitation on modeling
     asymmetric relations.
    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.tot_ent = self.data_stats.tot_entity
        self.tot_rel = self.data_stats.tot_relation
        self.model_name = 'Distmult'

    def def_inputs(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.hr_t = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])
        self.rt_h = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])

        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
            k = self.config.hidden_size
            with tf.name_scope("embedding"):
                self.emb_e = tf.get_variable(name="emb_e_real", shape=[self.tot_ent, k],
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                self.emb_rel = tf.get_variable(name="emb_rel_real", shape=[self.tot_rel, k],
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.emb_e, self.emb_rel]

    def def_loss(self):
        h_emb, r_emb, t_emb = self.embed(self.h, self.r, self.t)

        pred_tails = tf.matmul(h_emb * r_emb, tf.transpose(tf.nn.l2_normalize(self.emb_e, axis=1)))
        pred_heads = tf.matmul(t_emb * r_emb, tf.transpose(tf.nn.l2_normalize(self.emb_e, axis=1)))

        pred_tails = tf.nn.sigmoid(pred_tails)
        pred_heads = tf.nn.sigmoid(pred_heads)

        hr_t = self.hr_t * (1.0 - self.config.label_smoothing) + 1.0 / self.data_stats.tot_entity
        rt_h = self.rt_h * (1.0 - self.config.label_smoothing) + 1.0 / self.data_stats.tot_entity

        loss_tails = tf.reduce_mean(tf.keras.backend.binary_crossentropy(hr_t, pred_tails))
        loss_heads = tf.reduce_mean(tf.keras.backend.binary_crossentropy(rt_h, pred_heads))

        self.loss = loss_tails + loss_heads

    def test_batch(self):
        h_emb, r_emb, t_emb = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        pred_tails = tf.matmul(h_emb * r_emb, tf.transpose(tf.nn.l2_normalize(self.emb_e, axis=1)))
        pred_tails = tf.nn.sigmoid(pred_tails)

        pred_heads = tf.matmul(t_emb * r_emb, tf.transpose(tf.nn.l2_normalize(self.emb_e, axis=1)))
        pred_heads = tf.nn.sigmoid(pred_heads)

        _, head_rank = tf.nn.top_k(pred_tails, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(pred_heads, k=self.data_stats.tot_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        norm_emb_e = tf.nn.l2_normalize(self.emb_e, axis=1)
        norm_emb_rel = tf.nn.l2_normalize(self.emb_rel, axis=1)

        h_emb = tf.nn.embedding_lookup(norm_emb_e, h)
        r_emb = tf.nn.embedding_lookup(norm_emb_rel, r)
        t_emb = tf.nn.embedding_lookup(norm_emb_e, t)

        return h_emb, r_emb, t_emb

    def get_embed(self, h, r, t, sess=None):
        """function to get the embedding value in numpy"""
        h, r, t = self.embed(h, r, t)
        return sess.run([h, r, t])

    def get_proj_embed(self, h, r, t, sess):
        """function to get the projected embedding value in numpy"""
        return self.get_embed(h, r, t, sess)

