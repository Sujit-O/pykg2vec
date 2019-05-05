#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("../")
import tensorflow as tf
from core.KGMeta import ModelMeta

import numpy as np
import pickle 

class TransM(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    Transition-based Knowledge Graph Embedding with Relational Mapping Properties
    ------------------Paper Authors---------------------------
    Miao Fan(1,3), Qiang Zhou(1), Emily Chang(2), Thomas Fang Zheng(1,4),
    1. CSLT, Tsinghua National Laboratory for Information Science and Technology
    Department of Computer Science and Technology, Tsinghua University, Beijing, 100084, China.
    2. Emory University, U.S.A.
    3. fanmiao.cslt.thu@gmail.com, 4. fzheng@tsinghua.edu.cn Abstract
    ------------------Summary---------------------------------
    TransM is another line of research that improves TransE by relaxing the overstrict requirement of 
    h+r ==> t. TransM associates each fact (h, r, t) with a weight theta(r) specific to the relation. 
     

    https://github.com/wencolani/TransE.git
    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.train_triples_ids = self.config.read_train_triples_ids()
        self.model_name = 'TransM'

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
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))


            rel_head = {x: [] for x in range(num_total_rel)}
            rel_tail = {x: [] for x in range(num_total_rel)}
            rel_counts = {x: 0 for x in range(num_total_rel)}
            for t in self.train_triples_ids:
                rel_head[t.r].append(t.h)
                rel_tail[t.r].append(t.t)
                rel_counts[t.r] += 1

            theta = [1/np.log(2+rel_counts[x]/(1+len(rel_tail[x])) + rel_counts[x]/(1+len(rel_head[x]))) for x in range(num_total_rel)]
            self.theta = tf.Variable(np.asarray(theta, dtype=np.float32), trainable=False)
            
            self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.theta]

    def def_loss(self):
        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        pos_r_theta = tf.nn.embedding_lookup(self.theta, self.pos_r)
        neg_r_theta = tf.nn.embedding_lookup(self.theta, self.neg_r)

        score_pos = pos_r_theta*self.distance(pos_h_e, pos_r_e, pos_t_e)
        score_neg = neg_r_theta*self.distance(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(score_pos + self.config.margin - score_neg, 0))

    def test_step(self):
        head_vec, rel_vec, tail_vec = self.embed(self.test_h, self.test_r, self.test_t)
        r_theta = tf.nn.embedding_lookup(self.theta, self.test_r)

        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        score_head = r_theta*self.distance(norm_ent_embeddings, rel_vec, tail_vec)
        score_tail = r_theta*self.distance(head_vec, rel_vec, norm_ent_embeddings)

        _, head_rank = tf.nn.top_k(score_head, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(score_tail, k=self.data_stats.tot_entity)

        return head_rank, tail_rank

    def test_batch(self):
        head_vec, rel_vec, tail_vec = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        score_head = self.distance_batch(norm_ent_embeddings, tf.expand_dims(rel_vec,axis=1), tf.expand_dims(tail_vec,axis=1))
        score_tail = self.distance_batch(tf.expand_dims(head_vec,axis=1), tf.expand_dims(rel_vec,axis=1), norm_ent_embeddings)

        _, head_rank = tf.nn.top_k(score_head, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(score_tail, k=self.data_stats.tot_entity)

        return head_rank, tail_rank

    def distance(self, h, r, t):
        if self.config.L1_flag:
            return tf.reduce_sum(tf.abs(h + r - t), axis=1)  # L1 norm
        else:
            return tf.reduce_sum((h + r - t) ** 2, axis=1)  # L2 norm

    def distance_batch(self, h, r, t):
        if self.config.L1_flag:
            return tf.reduce_sum(tf.abs(h + r - t), axis=2)  # L1 norm
        else:
            return tf.reduce_sum((h + r - t) ** 2, axis=2)  # L2 norm

    def embed(self, h, r, t):
        """function to get the embedding value"""
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        emb_h = tf.nn.embedding_lookup(norm_ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(norm_ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def get_embed(self, h, r, t, sess):
        """function to get the embedding value in numpy"""
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess=None):
        """function to get the projected embedding value in numpy"""
        return self.get_embed(h, r, t, sess)

# if __name__ == "__main__":
#     import tensorflow as tf
#     from config.config import TransMConfig
#     from config.global_config import GeneratorConfig
#     from utils.dataprep import DataPrep, DataInput, DataStats
#     from utils.generator import Generator
#     sess = tf.Session()

#     config = TransMConfig()

#     data_handler = DataPrep(name_dataset="Freebase15k", sampling="uniform", algo='TransM')

#     model = TransM(config, data_handler)
#     import pdb
#     pdb.set_trace()

#     gen = iter(Generator(config=GeneratorConfig(data='train', algo='TransM')))
    
#     data = list(next(gen))

#     ph = data[0]
#     pr = data[1]
#     pt = data[2]
#     nh = data[3]
#     nr = data[4]
#     nt = data[5]

#     import pdb
#     pdb.set_trace()

#     sess.run(tf.global_variables_initializer())

#     sess.run(model.embed(ph, pr, pt))
    
#     feed_dict = {
#         model.pos_h:ph, 
#         model.pos_r:pr, 
#         model.pos_t:pt, 
#         model.neg_h:nh, 
#         model.neg_r:nr, 
#         model.neg_t:nt
#     }

#     sess.run(model.loss, feed_dict=feed_dict)

#     # feed_dict = {
#     #     model.test_h: np.reshape(ph, [1, ]),
#     #     model.test_r: np.reshape(pr, [1, ]),
#     #     model.test_t: np.reshape(pt, [1, ])
#     # }