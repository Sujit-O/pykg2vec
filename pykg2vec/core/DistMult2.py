#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")
import tensorflow as tf
from core.KGMeta import ModelMeta


class DistMult2(ModelMeta):
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

    def __init__(self, config):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.model_name = 'Distmult2'

        # self.def_inputs()
        # self.def_parameters()
        # self.def_loss()

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
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        k = self.config.hidden_size

        self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                              shape=[num_total_ent, k],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.rel_embeddings = tf.get_variable(name="rel_embeddings",
                                              shape=[num_total_rel, k],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings]

    def cal_truth_val(self, h, r, t):
        # dim of h: [m, k, 1]
        #        r: [m, k, k]
        #        t: [m, k, 1]
        return tf.reduce_sum(h * r * t, axis=-1)

    def def_loss(self):
        # norm_ent_embeddings = self.ent_embeddings # tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        # norm_rel_embeddings = self.rel_embeddings # tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        
        pos_h_e = tf.nn.embedding_lookup(norm_ent_embeddings, self.pos_h)
        pos_r_e = tf.nn.embedding_lookup(norm_rel_embeddings, self.pos_r)
        pos_t_e = tf.nn.embedding_lookup(norm_ent_embeddings, self.pos_t)
        neg_h_e = tf.nn.embedding_lookup(norm_ent_embeddings, self.neg_h)
        neg_r_e = tf.nn.embedding_lookup(norm_rel_embeddings, self.neg_r)
        neg_t_e = tf.nn.embedding_lookup(norm_ent_embeddings, self.neg_t)

        pos_score = self.cal_truth_val(pos_h_e, pos_r_e, pos_t_e)
        neg_score = self.cal_truth_val(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(neg_score + self.config.margin - pos_score, 0)) + 0.0001 * (tf.nn.l2_loss(norm_ent_embeddings) + tf.nn.l2_loss(norm_rel_embeddings))

    def test_step(self):
        num_entity = self.data_stats.tot_entity
        # norm_ent_embeddings = self.ent_embeddings # tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        # norm_rel_embeddings = self.rel_embeddings # tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        h_vec = tf.nn.embedding_lookup(norm_ent_embeddings, self.test_h)
        r_vec = tf.nn.embedding_lookup(norm_rel_embeddings, self.test_r)
        t_vec = tf.nn.embedding_lookup(norm_ent_embeddings, self.test_t)

        h_sim = self.cal_truth_val(norm_ent_embeddings, r_vec, t_vec)
        t_sim = self.cal_truth_val(h_vec, r_vec, norm_ent_embeddings)

        _, head_rank = tf.nn.top_k(-h_sim, k=num_entity)
        _, tail_rank = tf.nn.top_k(-t_sim, k=num_entity)

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
