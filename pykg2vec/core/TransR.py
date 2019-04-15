#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
sys.path.append("../")

from core.KGMeta import ModelMeta
from utils.visualization import Visualization

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
        with tf.name_scope("read_inputs"):
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
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        with tf.name_scope('lookup_embeddings'):
            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
            neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)
            pos_matrix = tf.nn.embedding_lookup(self.rel_matrix, self.pos_r)
            neg_matrix = tf.nn.embedding_lookup(self.rel_matrix, self.neg_r)

        with tf.name_scope('reshaping'):
            pos_h_e = tf.reshape(pos_h_e, [-1, d, 1])
            pos_r_e = tf.reshape(pos_r_e, [-1, k])
            pos_t_e = tf.reshape(pos_t_e, [-1, d, 1])
            neg_h_e = tf.reshape(neg_h_e, [-1, d, 1])
            neg_r_e = tf.reshape(neg_r_e, [-1, k])
            neg_t_e = tf.reshape(neg_t_e, [-1, d, 1])            
            pos_matrix = tf.reshape(pos_matrix, [-1, k, d])
            neg_matrix = tf.reshape(neg_matrix, [-1, k, d])

        with tf.name_scope('transformation'):
            transform_pos_h_e = self.transform(pos_matrix, pos_h_e)
            transform_pos_t_e = self.transform(pos_matrix, pos_t_e)
            transform_neg_h_e = self.transform(neg_matrix, neg_h_e)
            transform_neg_t_e = self.transform(neg_matrix, neg_t_e)

            pos_h_e = tf.nn.l2_normalize(tf.reshape(transform_pos_h_e, [-1, k]), 1)
            pos_r_e = tf.nn.l2_normalize(tf.reshape(pos_r_e, [-1, k]), 1)
            pos_t_e = tf.nn.l2_normalize(tf.reshape(transform_pos_t_e, [-1, k]), 1)           
            neg_h_e = tf.nn.l2_normalize(tf.reshape(transform_neg_h_e, [-1, k]), 1)
            neg_r_e = tf.nn.l2_normalize(tf.reshape(neg_r_e, [-1, k]), 1)
            neg_t_e = tf.nn.l2_normalize(tf.reshape(transform_neg_t_e, [-1, k]), 1)

        if self.config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)

        self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.config.margin, 0))
    
    def transform(self, matrix, embeddings):
        return tf.matmul(matrix, embeddings)

    def test_step(self):
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size
        num_total_ent = self.data_handler.tot_entity
        num_total_rel = self.data_handler.tot_relation

        head_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_h)
        rel_vec  = tf.nn.embedding_lookup(self.rel_embeddings, self.test_r)
        tail_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_t)
        pos_matrix = tf.nn.embedding_lookup(self.rel_matrix, self.test_r)

        head_vec = tf.reshape(head_vec, [-1, d, 1])
        rel_vec  = tf.reshape(rel_vec,  [-1, k, 1])
        tail_vec = tf.reshape(tail_vec, [-1, d, 1])
        pos_matrix = tf.reshape(pos_matrix, [-1, k, d])
 
        head_vec = self.transform(pos_matrix, head_vec) 
        tail_vec = self.transform(pos_matrix, tail_vec) 
        
        head_vec = tf.nn.l2_normalize(tf.reshape(head_vec, [-1, k]), 1)
        rel_vec  = tf.nn.l2_normalize(tf.reshape(rel_vec,  [-1, k]), 1)
        tail_vec = tf.nn.l2_normalize(tf.reshape(tail_vec, [-1, k]), 1)

        project_ent_embedding = self.transform(self.ent_embeddings, tf.transpose(tf.squeeze(pos_matrix, [0])))
        project_ent_embedding = tf.nn.l2_normalize(project_ent_embedding, axis=1)
        
        head_score = tf.reduce_sum(tf.abs(project_ent_embedding + rel_vec - tail_vec), axis=1)
        tail_score = tf.reduce_sum(tf.abs(head_vec + rel_vec - project_ent_embedding), axis=1)
        
        norm_embedding_entity = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_embedding_relation = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        norm_head_vec = tf.nn.embedding_lookup(norm_embedding_entity, self.test_h)
        norm_rel_vec  = tf.nn.embedding_lookup(norm_embedding_relation, self.test_r)
        norm_tail_vec = tf.nn.embedding_lookup(norm_embedding_entity, self.test_t)

        norm_head_vec = tf.matmul(norm_head_vec, tf.transpose(tf.squeeze(pos_matrix, [0])))
        norm_tail_vec = tf.matmul(norm_tail_vec, tf.transpose(tf.squeeze(pos_matrix, [0])))

        norm_head_vec = tf.nn.l2_normalize(tf.reshape(norm_head_vec, [-1, k]), 1)
        norm_rel_vec =  tf.nn.l2_normalize(tf.reshape(norm_rel_vec,  [-1, k]), 1)
        norm_tail_vec = tf.nn.l2_normalize(tf.reshape(norm_tail_vec, [-1, k]), 1)
        
        norm_head_score = tf.reduce_sum(tf.abs(project_ent_embedding + norm_rel_vec - norm_tail_vec), axis=1)
        norm_tail_score = tf.reduce_sum(tf.abs(norm_head_vec + norm_rel_vec - project_ent_embedding), axis=1)

        _, self.head_rank = tf.nn.top_k(head_score, k=num_total_ent)
        _, self.tail_rank = tf.nn.top_k(tail_score, k=num_total_ent)
        _, self.norm_head_rank = tf.nn.top_k(norm_head_score, k=num_total_ent)
        _, self.norm_tail_rank = tf.nn.top_k(norm_tail_score, k=num_total_ent)

        return self.head_rank, self.tail_rank, self.norm_head_rank, self.norm_tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                    h),
                             [-1, self.config.ent_hidden_size, 1])
        pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings,
                                                    r),
                             [-1, self.config.rel_hidden_size])
        pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                    t),
                             [-1, self.config.ent_hidden_size, 1])
        return pos_h_e, pos_r_e, pos_t_e

    def predict_embed(self, h, r, t, sess=None):
        """function to get the embedding value in numpy"""
        if not sess:
            raise NotImplementedError('No session found for predicting embedding!')
        pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                    h),
                             [-1, self.config.ent_hidden_size, 1])
        pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings,
                                                    r),
                             [-1, self.config.rel_hidden_size])
        pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                    t),
                             [-1, self.config.ent_hidden_size, 1])
        h, r, t = sess.run([pos_h_e, pos_r_e, pos_t_e])
        return h, r, t
    
    def summary(self):
        """function to print the summary"""
        print("\n----------------SUMMARY----------------")
        # Acquire the max length and add four more spaces
        maxspace = len(max([k for k in self.config.__dict__.keys()])) + 15
        for key, val in self.config.__dict__.items():
            if 'gpu' in key:
                continue
            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            print(key, ":", val)
        print("-----------------------------------------")

    def display(self, sess=None):
        """function to display embedding"""
        if self.config.plot_embedding:
            triples = self.data_handler.validation_triples_ids[:self.config.disp_triple_num]
            viz = Visualization(triples=triples,
                                idx2entity=self.data_handler.idx2entity,
                                idx2relation=self.data_handler.idx2relation)

            viz.get_idx_n_emb(model=self, sess=sess)
            viz.reduce_dim()
            viz.plot_embedding(resultpath=self.config.figures, algos=self.model_name)

        if self.config.plot_training_result:
            viz = Visualization()
            viz.plot_train_result(path=self.config.result,
                                  result=self.config.figures,
                                  algo=['TransE', 'TransR', 'TransH'],
                                  data=['Freebase15k'])

        if self.config.plot_testing_result:
            viz = Visualization()
            viz.plot_test_result(path=self.config.result,
                                 result=self.config.figures,
                                 algo=['TransE', 'TransR', 'TransH'],
                                 data=['Freebase15k'], paramlist=None, hits=self.config.hits)