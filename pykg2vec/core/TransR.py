#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from pykg2vec.core.KGMeta import ModelMeta


class TransR(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    Learning Entity and Relation Embeddings for Knowledge Graph Completion
    ------------------Paper Authors---------------------------
    Yankai Lin1, Zhiyuan Liu1∗, Maosong Sun 1,2, Yang Liu3, Xuan Zhu 3
    1 Department of Computer Science and Technology, State Key Lab on Intelligent Technology and Systems,
    National Lab for Information Science and Technology, Tsinghua University, Beijing, China
    2 Jiangsu Collaborative Innovation Center for Language Competence, Jiangsu, China
    3 Samsung R&D Institute of China, Beijing, China
    ------------------Summary---------------------------------
    TranR is a translation based knowledge graph embedding method. Similar to TransE and TransH, it also
    builds entity and relation embeddings by regarding a relation as translation from head entity to tail
    entity. However, compared to them, it builds the entity and relation embeddings in a separate entity
    and relation spaces.

    Portion of Code Based on https://github.com/thunlp/TensorFlow-TransX/blob/master/transR.py
    """

    def __init__(self, config, data_handler, load_entity=None, load_rel=None):
        self.config = config
        self.data_handler = data_handler
        self.model_name = 'TransR'
        
    def def_inputs(self):
        self.pos_h = tf.placeholder(tf.int32, [self.config.batch_size])
        self.pos_t = tf.placeholder(tf.int32, [self.config.batch_size])
        self.pos_r = tf.placeholder(tf.int32, [self.config.batch_size])
        self.neg_h = tf.placeholder(tf.int32, [self.config.batch_size])
        self.neg_t = tf.placeholder(tf.int32, [self.config.batch_size])
        self.neg_r = tf.placeholder(tf.int32, [self.config.batch_size])
        self.test_h = tf.placeholder(tf.int32, [1])
        self.test_t = tf.placeholder(tf.int32, [1])
        self.test_r = tf.placeholder(tf.int32, [1])

    def def_parameters(self):
        num_total_ent = self.data_handler.tot_entity
        num_total_rel = self.data_handler.tot_relation
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        with tf.name_scope("embedding"):

            self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                                  shape=[num_total_ent, d],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding",
                                                  shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            rel_matrix = np.zeros([num_total_rel, d*k], dtype=np.float32)
            
            for i in range(num_total_rel):
                for j in range(k):
                    for z in range(d):
                        if j == z:
                            rel_matrix[i][j * d + z] = 1.0

            self.rel_matrix = tf.Variable(rel_matrix, name="rel_matrix")

            self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.rel_matrix]

    def def_loss(self):
        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        score_pos = self.distance(pos_h_e, pos_r_e, pos_t_e)
        score_neg = self.distance(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(score_pos - score_neg + self.config.margin, 0))

    def test_step(self):
        num_total_ent = self.data_handler.tot_entity

        head_vec, rel_vec, tail_vec = self.embed(self.test_h, self.test_r, self.test_t)
        pos_matrix = self.get_transform_matrix(self.test_r)
 
        project_ent_embedding = self.transform(self.ent_embeddings, tf.transpose(tf.squeeze(pos_matrix, [0])))
        project_ent_embedding = tf.nn.l2_normalize(project_ent_embedding, axis=1)
        
        score_head = self.distance(project_ent_embedding, rel_vec, tail_vec)
        score_tail = self.distance(head_vec, rel_vec, project_ent_embedding)
        
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        norm_head_vec, norm_rel_vec, norm_tail_vec = self.embed(self.test_h, self.test_r, self.test_t) 
        norm_pos_matrix = self.get_transform_matrix(self.test_r)
 
        norm_project_ent_embedding = self.transform(self.ent_embeddings, tf.transpose(tf.squeeze(norm_pos_matrix, [0])))
        norm_project_ent_embedding = tf.nn.l2_normalize(norm_project_ent_embedding, axis=1)

        norm_score_head = self.distance(norm_project_ent_embedding, norm_rel_vec, norm_tail_vec)
        norm_score_tail = self.distance(norm_head_vec, norm_rel_vec, norm_project_ent_embedding)

        _, self.head_rank = tf.nn.top_k(score_head, k=num_total_ent)
        _, self.tail_rank = tf.nn.top_k(score_tail, k=num_total_ent)
        _, self.norm_head_rank = tf.nn.top_k(norm_score_head, k=num_total_ent)
        _, self.norm_tail_rank = tf.nn.top_k(norm_score_tail, k=num_total_ent)

        return self.head_rank, self.tail_rank, self.norm_head_rank, self.norm_tail_rank

    def transform(self, matrix, embeddings):
        return tf.matmul(matrix, embeddings)

    def distance(self, h, r, t):
        if self.config.L1_flag: 
            return tf.reduce_sum(tf.abs(h+r-t), axis=1) # L1 norm 
        else:
            return tf.reduce_sum((h+r-t)**2, axis=1) # L2 norm

    def get_transform_matrix(self, r):
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size
        return tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, r), [-1, k, d])

    def embed(self, h, r, t):
        """function to get the embedding value"""
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, h), [-1, d, 1])
        r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings, r), [-1, k])
        t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings, t), [-1, d, 1])
        matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, r), [-1, k, d])

        transform_h_e = self.transform(matrix, h_e)
        transform_t_e = self.transform(matrix, t_e)
        h_e = tf.nn.l2_normalize(tf.reshape(transform_h_e, [-1, k]), -1)
        r_e = tf.nn.l2_normalize(tf.reshape(r_e, [-1, k]), -1)
        t_e = tf.nn.l2_normalize(tf.reshape(transform_t_e, [-1, k]), -1) 

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