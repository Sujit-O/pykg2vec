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


class TransH(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    Knowledge Graph Embedding by Translating on Hyperplanes
    ------------------Paper Authors---------------------------
    Zhen Wang1,Jianwen Zhang2, Jianlin Feng1, Zheng Chen2
    1Department of Information Science and Technology, Sun Yat-sen University, Guangzhou, China
    2Microsoft Research, Beijing, China
    1{wangzh56@mail2, fengjlin@mail}.sysu.edu.cn
    2{jiazhan, zhengc}@microsoft.com
    ------------------Summary---------------------------------
    TransH models a relation as a hyperplane together with a translation operation on it.
    By doint this, it aims to preserve the mapping properties of relations such as reflexive,
    one-to-many, many-to-one, and many-to-many with almost the same model complexity of TransE.

    Portion of Code Based on https://github.com/thunlp/OpenKE/blob/master/models/TransH.py
     and https://github.com/thunlp/TensorFlow-TransX/blob/master/transH.py
    """

    def __init__(self, config):
        self.config = config
        with open(self.config.tmp_data / 'data_stats.pkl', 'rb') as f:
            self.data_stats = pickle.load(f)
        self.model_name = 'TransH'

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
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.w = tf.get_variable(name="w", shape=[num_total_rel, k],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.w]

    def def_loss(self):
        emb_ph, emb_pr, emb_pt = self.embed(self.pos_h, self.pos_r, self.pos_t)
        emb_nh, emb_nr, emb_nt = self.embed(self.neg_h, self.neg_r, self.neg_t)

        score_pos = self.distance(emb_ph, emb_pr, emb_pt)
        score_neg = self.distance(emb_nh, emb_nr, emb_nt)

        self.loss = tf.reduce_sum(tf.maximum(0., score_pos + self.config.margin - score_neg)) + self.get_reg()

    def get_reg(self):
        norm_ent_embedding = tf.nn.l2_normalize(self.ent_embeddings, axis=-1)
        norm_rel_embedding = tf.nn.l2_normalize(self.rel_embeddings, axis=-1)
        norm_w = tf.nn.l2_normalize(self.w, axis=-1)

        term1 = tf.reduce_sum(tf.maximum(tf.reduce_sum(norm_ent_embedding ** 2, -1) - 1, 0))
        term2 = tf.reduce_sum(tf.maximum(tf.div(tf.reduce_sum(norm_rel_embedding * norm_w, -1) ** 2,
                                                tf.reduce_sum(norm_rel_embedding ** 2, -1)) - 1e-07, 0))

        return self.config.C * (term1 + term2)

    def test_step(self):
        num_entity = self.data_stats.tot_entity

        head_vec, rel_vec, tail_vec = self.embed(self.test_h, self.test_r, self.test_t)
        pos_norm = self.get_proj(self.test_r)

        norm_ent_embedding = tf.nn.l2_normalize(self.ent_embeddings, 1)
        project_ent_embedding = self.projection(norm_ent_embedding, pos_norm)
        score_head = self.distance(project_ent_embedding, rel_vec, tail_vec)
        score_tail = self.distance(head_vec, rel_vec, project_ent_embedding)

        _, head_rank = tf.nn.top_k(score_head, k=num_entity)
        _, tail_rank = tf.nn.top_k(score_tail, k=num_entity)

        return head_rank, tail_rank

    def get_proj(self, r):
        return tf.nn.l2_normalize(tf.nn.embedding_lookup(self.w, r), axis=-1)

    def projection(self, entity, wr):
        return entity - tf.reduce_sum(entity * wr, -1, keepdims=True) * wr

    def distance(self, h, r, t):
        if self.config.L1_flag:
            return tf.reduce_sum(tf.abs(h + r - t), axis=1)  # L1 norm
        else:
            return tf.reduce_sum((h + r - t) ** 2, axis=1)  # L2 norm

    def embed(self, h, r, t):
        """function to get the embedding value"""
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)

        emb_h = tf.nn.l2_normalize(emb_h, axis=-1)
        emb_r = tf.nn.l2_normalize(emb_r, axis=-1)
        emb_t = tf.nn.l2_normalize(emb_t, axis=-1)

        proj_vec = self.get_proj(r)

        return self.projection(emb_h, proj_vec), emb_r, self.projection(emb_t, proj_vec)

    def get_embed(self, h, r, t, sess):
        """function to get the embedding value in numpy"""
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess):
        """function to get the projectd embedding value in numpy"""
        return self.get_embed(h, r, t, sess)
