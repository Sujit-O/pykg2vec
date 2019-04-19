#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")
import tensorflow as tf
from core.KGMeta import ModelMeta

class TransE(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    Translating Embeddings for Modeling Multi-relational Data
    ------------------Paper Authors---------------------------
    Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran
    Universite de Technologie de Compiegne – CNRS
    Heudiasyc UMR 7253
    Compiegne, France
    {bordesan, nusunier, agarciad}@utc.fr
    Jason Weston, Oksana Yakhnenko
    Google
    111 8th avenue
    New York, NY, USA
    {jweston, oksana}@google.com
    ------------------Summary---------------------------------
    TransE is an energy based model which represents the
    relationships as translations in the embedding space. Which
    means that if (h,l,t) holds then the embedding of the tail
    't' should be close to the embedding of head entity 'h'
    plus some vector that depends on the relationship 'l'.
    Both entities and relations are vectors in the same space.
    |        ......>.
    |      .     .
    |    .    .
    |  .  .
    |_________________
    Portion of Code Based on https://github.com/thunlp/OpenKE/blob/master/models/TransE.py
     and https://github.com/wencolani/TransE.git
    """
    def __init__(self, config=None, data_handler=None):
        self.config = config
        self.data_handler = data_handler
        self.model_name = 'TransE'

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
        num_total_ent = self.data_handler.tot_entity
        num_total_rel = self.data_handler.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            
            self.parameter_list = [self.ent_embeddings, self.rel_embeddings]    
    
    def def_loss(self):
        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        score_pos = self.distance(pos_h_e, pos_r_e, pos_t_e)
        score_neg = self.distance(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(score_pos + self.config.margin - score_neg, 0))

    def test_step(self):       
        head_vec, rel_vec, tail_vec = self.embed(self.test_h, self.test_r, self.test_t)     
        
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1) 
        score_head = self.distance(norm_ent_embeddings, rel_vec, tail_vec)
        score_tail = self.distance(head_vec, rel_vec, norm_ent_embeddings)    

        _, self.head_rank      = tf.nn.top_k(score_head, k=self.data_handler.tot_entity)
        _, self.tail_rank      = tf.nn.top_k(score_tail, k=self.data_handler.tot_entity)

        return self.head_rank, self.tail_rank

    def distance(self, h, r, t):
        if self.config.L1_flag: 
            return tf.reduce_sum(tf.abs(h+r-t), axis=1) # L1 norm 
        else:
            return tf.reduce_sum((h+r-t)**2, axis=1) # L2 norm
            
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