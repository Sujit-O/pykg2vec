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
            self.rel_matrices   = tf.get_variable(name="rel_matrices",
                                                  shape=[num_total_rel, k*k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.ent_embeddings, self.rel_matrices]

    def cal_truth_val(self, h, r, t):
        # dim of h: [m, k, 1]
        #        r: [m, k, k]
        #        t: [m, k, 1]
        return tf.reduce_sum(h*tf.matmul(r,t), [1,2])

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

        with tf.name_scope('lookup_embeddings'):
            h_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_h)
            r_vec = tf.nn.embedding_lookup(self.rel_matrices, self.test_r)
            t_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_t)
 
        with tf.name_scope('reshaping'):
            h_vec = tf.reshape(h_vec, [k, 1])
            r_vec = tf.reshape(r_vec, [k, k])
            t_vec = tf.reshape(t_vec, [k, 1])
       
        h_sim = tf.matmul(self.ent_embeddings, tf.matmul(r_vec, t_vec))
        t_sim = tf.transpose(tf.matmul(tf.matmul(tf.transpose(h_vec), r_vec), tf.transpose(self.ent_embeddings)))

        with tf.name_scope('normalization'):
            self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
            self.rel_matrices = tf.nn.l2_normalize(self.rel_matrices, axis=1)

        with tf.name_scope('lookup_embeddings'):
            norm_h_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_h)
            norm_r_vec = tf.nn.embedding_lookup(self.rel_matrices, self.test_r)
            norm_t_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_t)

        with tf.name_scope('reshaping'):        
            norm_h_vec = tf.reshape(norm_h_vec, [k, 1])
            norm_r_vec = tf.reshape(norm_r_vec, [k, k])
            norm_t_vec = tf.reshape(norm_t_vec, [k, 1])

        norm_h_sim = tf.matmul(self.ent_embeddings, tf.matmul(norm_r_vec, norm_t_vec))
        norm_t_sim = tf.transpose(tf.matmul(tf.matmul(tf.transpose(norm_h_vec), norm_r_vec), tf.transpose(self.ent_embeddings)))

        _, self.head_rank      = tf.nn.top_k(tf.reduce_sum(tf.negative(h_sim), 1), k=num_entity)
        _, self.tail_rank      = tf.nn.top_k(tf.reduce_sum(tf.negative(t_sim), 1), k=num_entity)
        _, self.norm_head_rank = tf.nn.top_k(tf.reduce_sum(tf.negative(norm_h_sim), 1), k=num_entity)
        _, self.norm_tail_rank = tf.nn.top_k(tf.reduce_sum(tf.negative(norm_t_sim), 1), k=num_entity)

        return self.head_rank, self.tail_rank, self.norm_head_rank, self.norm_tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_matrices, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def get_embed(self, h, r, t, sess):
        """function to get the embedding value in numpy"""
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t   

    def get_proj_embed(self, h, r, t, sess):
        """function to get the projectd embedding value in numpy"""
        return self.get_embed(h, r, t, sess)