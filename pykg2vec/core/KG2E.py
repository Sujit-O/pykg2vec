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

class KG2E(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    Learning to Represent Knowledge Graphs with Gaussian Embedding
    ------------------Paper Authors---------------------------
    Shizhu He, Kang Liu, Guoliang Ji and Jun Zhao 
    National Laboratory of Pattern Recognition Institute of Automation, 
    Chinese Academy of Sciences, Beijing, 100190, China 
    {shizhu.he, kliu, guoliang.ji, jzhao}@nlpr.ia.ac.cn
    ------------------Summary---------------------------------

    Portion of the code based on https://github.com/mana-ysh/gaussian-embedding/blob/master/src/models/gaussian_model.py
    """

    def __init__(self, config=None):
        self.config = config
        with open(self.config.tmp_data / 'data_stats.pkl', 'rb') as f:
            self.data_stats = pickle.load(f)

        if self.config.distance_measure == "expected_likelihood":
            self.model_name = 'KG2E_EL'
        else:
            self.model_name = 'KG2E_KL'

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
        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            # the mean for each element in the embedding space. 
            self.ent_embeddings_mu    = tf.get_variable(name="ent_embeddings_mu", shape=[num_total_ent, k], 
                                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))           
            self.rel_embeddings_mu    = tf.get_variable(name="rel_embeddings_mu", shape=[num_total_rel, k], 
                                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))

            # as the paper suggested, sigma is simplified to be the diagonal element in the covariance matrix. 
            self.ent_embeddings_sigma = tf.get_variable(name="ent_embeddings_sigma", shape=[num_total_ent, k], 
                                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            self.rel_embeddings_sigma = tf.get_variable(name="rel_embeddings_sigma", shape=[num_total_rel, k], 
                                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))

            self.ent_embeddings_sigma = tf.maximum(self.config.cmin, tf.minimum(self.config.cmax, (self.ent_embeddings_sigma + 1.0)))
            self.rel_embeddings_sigma = tf.maximum(self.config.cmin, tf.minimum(self.config.cmax, (self.rel_embeddings_sigma + 1.0)))

            self.parameter_list = [self.ent_embeddings_mu, self.ent_embeddings_sigma, 
                                   self.rel_embeddings_mu, self.rel_embeddings_sigma]


    def def_loss(self):
        pos_h_mu, pos_h_sigma, pos_r_mu, pos_r_sigma, pos_t_mu, pos_t_sigma = self.get_embed_guassian(self.pos_h, self.pos_r, self.pos_t)
        neg_h_mu, neg_h_sigma, neg_r_mu, neg_r_sigma, neg_t_mu, neg_t_sigma = self.get_embed_guassian(self.neg_h, self.neg_r, self.neg_t)

        if self.config.distance_measure == "expected_likelihood":
            score_pos = self.cal_score_expected_likelihood(pos_h_mu, pos_h_sigma, pos_r_mu, pos_r_sigma, pos_t_mu, pos_t_sigma)
            score_neg = self.cal_score_expected_likelihood(neg_h_mu, neg_h_sigma, neg_r_mu, neg_r_sigma, neg_t_mu, neg_t_sigma)
        else:
            score_pos = self.cal_score_kl_divergence(pos_h_mu, pos_h_sigma, pos_r_mu, pos_r_sigma, pos_t_mu, pos_t_sigma)
            score_neg = self.cal_score_kl_divergence(neg_h_mu, neg_h_sigma, neg_r_mu, neg_r_sigma, neg_t_mu, neg_t_sigma)

        self.loss = tf.reduce_sum(tf.maximum(score_pos + self.config.margin - score_neg, 0))
        
       
    def cal_score_kl_divergence(self, h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma):
        ''' 
            trace_fac: tr(sigma_r-1 * (sigma_h + sigma_t))
            mul_fac: (mu_h + mu_r - mu_t).T * sigma_r-1 * (mu_h + mu_r - mu_t)
            det_fac: log(det(sigma_r)/det(sigma_h + sigma_t))
        '''
        trace_fac = tf.reduce_sum((h_sigma + t_sigma) / r_sigma, -1)
        mul_fac = tf.reduce_sum((- h_mu + t_mu - r_mu)**2 / r_sigma, -1)
        det_fac = tf.reduce_sum(tf.log(h_sigma + t_sigma) - tf.log(r_sigma), -1)
        
        return trace_fac + mul_fac - det_fac - self.config.hidden_size

    def cal_score_expected_likelihood(self, h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma):
        ''' 
            mul_fac: (mu_h + mu_r - mu_t).T * sigma_r-1 * (mu_h + mu_r - mu_t)
            det_fac: log(det(sigma_r + sigma_h + sigma_t))
        '''    
        mul_fac = tf.reduce_sum((h_mu + r_mu - t_mu)**2 / (h_sigma + r_sigma + t_sigma), -1)
        det_fac = tf.reduce_sum(tf.log(h_sigma + r_sigma + t_sigma), -1)
        
        return mul_fac + det_fac - self.config.hidden_size

    def test_step(self):
        test_h_mu, test_h_sigma, test_r_mu, test_r_sigma, test_t_mu, test_t_sigma = self.get_embed_guassian(self.test_h, self.test_r, self.test_t)

        norm_ent_embeddings_mu    = tf.nn.l2_normalize(self.ent_embeddings_mu,    axis=1)
        norm_ent_embeddings_sigma = tf.nn.l2_normalize(self.ent_embeddings_sigma, axis=1)

        if self.config.distance_measure == "expected_likelihood":
            score_head = self.cal_score_expected_likelihood(norm_ent_embeddings_mu, norm_ent_embeddings_sigma, \
                                                            test_r_mu, test_r_sigma, \
                                                            test_t_mu, test_t_sigma)

            score_tail = self.cal_score_expected_likelihood(test_h_mu, test_h_sigma, \
                                                            test_r_mu, test_r_sigma, \
                                                            norm_ent_embeddings_mu, norm_ent_embeddings_sigma)
        else:
            score_head = self.cal_score_kl_divergence(norm_ent_embeddings_mu, norm_ent_embeddings_sigma, \
                                                      test_r_mu, test_r_sigma, \
                                                      test_t_mu, test_t_sigma)

            score_tail = self.cal_score_kl_divergence(test_h_mu, test_h_sigma, \
                                                      test_r_mu, test_r_sigma, \
                                                      norm_ent_embeddings_mu, norm_ent_embeddings_sigma)
        
        _, head_rank = tf.nn.top_k(score_head, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(score_tail, k=self.data_stats.tot_entity)

        return head_rank, tail_rank

    def test_batch(self):
        test_h_mu, test_h_sigma, test_r_mu, test_r_sigma, test_t_mu, test_t_sigma = self.get_embed_guassian(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        norm_ent_embeddings_mu    = tf.nn.l2_normalize(self.ent_embeddings_mu,    axis=1)
        norm_ent_embeddings_sigma = tf.nn.l2_normalize(self.ent_embeddings_sigma, axis=1)

        if self.config.distance_measure == "expected_likelihood":
            score_head = self.cal_score_expected_likelihood(norm_ent_embeddings_mu, norm_ent_embeddings_sigma, \
                                                            tf.expand_dims(test_r_mu, axis=1), tf.expand_dims(test_r_sigma, axis=1), \
                                                            tf.expand_dims(test_t_mu, axis=1), tf.expand_dims(test_t_sigma, axis=1))

            score_tail = self.cal_score_expected_likelihood(tf.expand_dims(test_h_mu, axis=1), tf.expand_dims(test_h_sigma, axis=1), \
                                                            tf.expand_dims(test_r_mu, axis=1), tf.expand_dims(test_r_sigma, axis=1), \
                                                            norm_ent_embeddings_mu, norm_ent_embeddings_sigma)
        else:
            score_head = self.cal_score_kl_divergence(norm_ent_embeddings_mu, norm_ent_embeddings_sigma, \
                                                      tf.expand_dims(test_r_mu, axis=1), tf.expand_dims(test_r_sigma, axis=1), \
                                                      tf.expand_dims(test_t_mu, axis=1), tf.expand_dims(test_t_sigma, axis=1))

            score_tail = self.cal_score_kl_divergence(tf.expand_dims(test_h_mu, axis=1), tf.expand_dims(test_h_sigma, axis=1), \
                                                      tf.expand_dims(test_r_mu, axis=1), tf.expand_dims(test_r_sigma, axis=1), \
                                                      norm_ent_embeddings_mu, norm_ent_embeddings_sigma)

        _, head_rank = tf.nn.top_k(score_head, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(score_tail, k=self.data_stats.tot_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        emb_h = tf.nn.embedding_lookup(norm_ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(norm_ent_embeddings, t)
        return emb_h, emb_r, emb_t
    
    def get_embed_guassian(self, h, r, t):

        norm_ent_embeddings_mu = tf.nn.l2_normalize(self.ent_embeddings_mu, axis=1)
        norm_rel_embeddings_mu = tf.nn.l2_normalize(self.rel_embeddings_mu, axis=1)

        norm_ent_embeddings_sigma = tf.nn.l2_normalize(self.ent_embeddings_sigma, axis=1)
        norm_rel_embeddings_sigma = tf.nn.l2_normalize(self.rel_embeddings_sigma, axis=1)

        emb_h_mu = tf.nn.embedding_lookup(norm_ent_embeddings_mu, h)
        emb_r_mu = tf.nn.embedding_lookup(norm_rel_embeddings_mu, r)
        emb_t_mu = tf.nn.embedding_lookup(norm_ent_embeddings_mu, t)

        emb_h_sigma = tf.nn.embedding_lookup(norm_ent_embeddings_sigma, h)
        emb_r_sigma = tf.nn.embedding_lookup(norm_rel_embeddings_sigma, r)
        emb_t_sigma = tf.nn.embedding_lookup(norm_ent_embeddings_sigma, t)

        return emb_h_mu, emb_h_sigma, emb_r_mu, emb_r_sigma, emb_t_mu, emb_t_sigma

    def get_embed(self, h, r, t, sess):
        """function to get the embedding value in numpy"""
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess=None):
        """function to get the projected embedding value in numpy"""
        return self.get_embed(h, r, t, sess)