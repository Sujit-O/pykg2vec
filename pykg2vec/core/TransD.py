#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")
import tensorflow as tf
import numpy as np
from core.KGMeta import ModelMeta
import pickle

class TransD(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    Knowledge Graph Embedding via Dynamic Mapping Matrix
    ------------------Paper Authors---------------------------
    Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao
    National Laboratory of Pattern Recognition (NLPR) 
    Institute of Automation Chinese Academy of Sciences, Beijing, 100190, China 
    {guoliang.ji,shizhu.he,lhxu,kliu,jzhao}@nlpr.ia.ac.cn
    ------------------Summary---------------------------------
    TransD constructs a dynamic mapping matrix for each entity-relation pair by considering the diversity of entities and relations simultaneously. 
    Compared with TransR/CTransR, TransD has fewer parameters and has no matrix vector multiplication.

    Portion of Code Based on https://github.com/thunlp/OpenKE/blob/master/models/TransD.py
    """

    def __init__(self, config):
        self.config = config
        with open(self.config.tmp_data / 'data_stats.pkl', 'rb') as f:
            self.data_stats = pickle.load(f)
        self.model_name = 'TransD'
        
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

            self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                                  shape=[num_total_ent, d],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.ent_mappings   = tf.get_variable(name="ent_mappings",
                                                  shape=[num_total_ent, d],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.rel_embeddings = tf.get_variable(name="rel_embedding",
                                                  shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_mappings   = tf.get_variable(name="rel_mappings",
                                                  shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.ent_mappings, self.rel_mappings]

    def def_loss(self):
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        pos_h_m = tf.nn.embedding_lookup(self.ent_mappings, self.pos_h)
        pos_r_m = tf.nn.embedding_lookup(self.rel_mappings, self.pos_r)
        pos_t_m = tf.nn.embedding_lookup(self.ent_mappings, self.pos_t)

        pos_h_e = tf.nn.l2_normalize(tf.reduce_sum(((tf.expand_dims(pos_r_m, -1) @ tf.expand_dims(pos_h_m, -2)) + tf.eye(k, batch_shape=[tf.shape(pos_h_m)[0]], num_columns=d)) @ tf.expand_dims(pos_h_e, -1) , -1), -1)
        pos_t_e = tf.nn.l2_normalize(tf.reduce_sum(((tf.expand_dims(pos_r_m, -1) @ tf.expand_dims(pos_t_m, -2)) + tf.eye(k, batch_shape=[tf.shape(pos_t_m)[0]], num_columns=d)) @ tf.expand_dims(pos_t_e, -1) , -1), -1)

        neg_h_m = tf.nn.embedding_lookup(self.ent_mappings, self.neg_h)
        neg_r_m = tf.nn.embedding_lookup(self.rel_mappings, self.neg_r)
        neg_t_m = tf.nn.embedding_lookup(self.ent_mappings, self.neg_t)

        neg_h_e = tf.nn.l2_normalize(tf.reduce_sum(((tf.expand_dims(neg_r_m, -1) @ tf.expand_dims(neg_h_m, -2)) + tf.eye(k, batch_shape=[tf.shape(neg_h_m)[0]], num_columns=d)) @ tf.expand_dims(neg_h_e, -1) , -1), -1)
        neg_t_e = tf.nn.l2_normalize(tf.reduce_sum(((tf.expand_dims(neg_r_m, -1) @ tf.expand_dims(neg_t_m, -2)) + tf.eye(k, batch_shape=[tf.shape(neg_t_m)[0]], num_columns=d)) @ tf.expand_dims(neg_t_e, -1) , -1), -1)

        score_pos = self.distance(pos_h_e, pos_r_e, pos_t_e)
        score_neg = self.distance(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(score_pos - score_neg + self.config.margin, 0))

    def mapping(self, h_e, h_m, r_m):
        # ex. projected head => wr * whT * h + h
        return h_e + tf.matmul(r_m, h_m, transpose_b=True)

    def test_step(self):
        num_total_ent = self.data_stats.tot_entity
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        head_vec, rel_vec, tail_vec = self.embed(self.test_h, self.test_r, self.test_t)
        h_m = tf.nn.embedding_lookup(self.ent_mappings, self.test_h)
        r_m = tf.nn.embedding_lookup(self.rel_mappings, self.test_r)
        t_m = tf.nn.embedding_lookup(self.ent_mappings, self.test_t)

        head_vec = tf.nn.l2_normalize(tf.reduce_sum(((tf.expand_dims(r_m, -1) @ tf.expand_dims(h_m, -2)) + tf.eye(k, batch_shape=[tf.shape(h_m)[0]], num_columns=d)) @ tf.expand_dims(head_vec, -1) , -1), -1)
        tail_vec = tf.nn.l2_normalize(tf.reduce_sum(((tf.expand_dims(r_m, -1) @ tf.expand_dims(t_m, -2)) + tf.eye(k, batch_shape=[tf.shape(t_m)[0]], num_columns=d)) @ tf.expand_dims(tail_vec, -1) , -1), -1)
        
        project_ent_embedding = tf.nn.l2_normalize(tf.reduce_sum(((tf.tile(tf.expand_dims(r_m, -1), [tf.shape(self.ent_mappings)[0], 1, 1]) @ tf.expand_dims(self.ent_mappings, -2)) + tf.eye(k, batch_shape=[tf.shape(self.ent_mappings)[0]], num_columns=d)) @ tf.expand_dims(self.ent_embeddings, -1) , -1), -1)
               
        score_head = self.distance(project_ent_embedding, rel_vec, tail_vec)
        score_tail = self.distance(head_vec, rel_vec, project_ent_embedding)
        
        _, self.head_rank = tf.nn.top_k(score_head, k=num_total_ent)
        _, self.tail_rank = tf.nn.top_k(score_tail, k=num_total_ent)

        return self.head_rank, self.tail_rank

    def distance(self, h, r, t):    
        return tf.reduce_sum((h+r-t)**2, axis=1) # L2 norm

    def embed(self, h, r, t):
        """function to get the embedding value"""
        h_e = tf.nn.embedding_lookup(self.ent_embeddings, h)
        r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)
        t_e = tf.nn.embedding_lookup(self.ent_embeddings, t)

        return h_e, r_e, t_e

    def get_embed(self, h, r, t, sess):
        """function to get the embedding value in numpy"""
        pos_h_e, pos_r_e, pos_t_e = self.embed(h, r, t)
        # temp
        pos_h_e, pos_r_e, pos_t_e = tf.squeeze(pos_h_e, 0), tf.squeeze(pos_r_e, 0), tf.squeeze(pos_t_e, 0)
        h, r, t = sess.run([pos_h_e, pos_r_e, pos_t_e])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess=None):
        """function to get the projected embedding value in numpy"""
        return self.get_embed(h, r, t, sess) 