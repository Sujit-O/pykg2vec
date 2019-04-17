#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("../")
import tensorflow as tf
from core.KGMeta import ModelMeta


class NTN(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    A Semantic Matching Energy Function for Learning with Multi-relational Data
    ------------------Paper Authors---------------------------
    Antoine Bordes
    Universite de Technologie de Compiegne – CNRS
    Heudiasyc UMR 7253
    Compiegne, France
    antoine.bordes@utc.fr
    Jason Weston
    Google
    111 8th avenue
    New York, NY, USA
    jweston@google.com
    Xaiver Glorot, Joshua Bengio
    Universite de Montreal
    Monteal, QC, Canada
    {glorotxa, bengioy}@iro.umontreal.ca
    ------------------Summary---------------------------------
    Semantic Matching Energy (SME) is an algorithm for embedding multi-relational data into vector spaces. 
    SME conducts semantic matching using neural network architectures. Given a fact (h, r, t), it first projects 
    entities and relations to their embeddings in the input layer. Later the relation r is combined with both h and t
    to get gu(h, r) and gv(r, t) in its hidden layer. The score is determined by calculating the matching score of gu and gv.

    There are two versions of SME: a linear version(SMELinear) as well as bilinear(SMEBilinear) version which differ in how the hidden layer is defined.

    Portion of Code Based on https://github.com/glorotxa/SME/blob/master/model.py
    """
    def __init__(self, config=None, data_handler=None):
        self.config = config
        self.data_handler = data_handler
        self.model_name = 'NTN'
    
    def gu(self, h, r):
        return tf.transpose(tf.matmul(self.mu1, tf.transpose(h)) + tf.matmul(self.mu2, tf.transpose(r)) + self.bu)

    def gv(self, r, t):
        return tf.transpose(tf.matmul(self.mv1, tf.transpose(r)) + tf.matmul(self.mv2, tf.transpose(t)) + self.bv)

    def match(self, h, r, t):
        return tf.reduce_sum(tf.multiply(self.gu(h, r), self.gv(r, t)), 1)

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
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, d],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        
        with tf.name_scope("weights_and_parameters"):
            self.mr1 = tf.get_variable(name="mr1", shape=[d, k], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.mr2 = tf.get_variable(name="mr2", shape=[d, k], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.br  = tf.get_variable(name="br",  shape=[k, 1], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.mr  = tf.get_variable(name="mr",  shape=[k, d, d], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, \
                               self.mr1, self.mr2, self.br, self.mr]
    

    def def_loss(self):
        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        
        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        energy_pos = tf.reduce_sum(pos_r_e*self.layer(pos_h_e, pos_t_e), -1)
        energy_neg = tf.reduce_sum(neg_r_e*self.layer(neg_h_e, neg_t_e), -1)
        
        self.loss = tf.reduce_sum(tf.maximum(energy_neg + self.config.margin - energy_pos, 0))

    def layer(self, h, t, expand=None):
        # h => [m, d]
        # self.mr1 => [d, k]
        if expand == "t":
            t = tf.tile(t, [tf.shape(h)[0],1])
        elif expand == "h":
            h = tf.tile(h, [tf.shape(t)[0],1])

        mr1h = tf.matmul(h, self.mr1)

        # t_e => [m, d]
        # self.mr2 => [d, k]
        mr2t =  tf.matmul(t, self.mr2)

        # br = [k]
        br = tf.squeeze(self.br, -1)

        # [m, k, 1, d]
        expanded_h = tf.tile(tf.expand_dims(tf.expand_dims(h, 1),1), [1, self.config.rel_hidden_size, 1, 1])
        # [m, k, d, d]
        expanded_mr = tf.tile(tf.expand_dims(self.mr, 0), [tf.shape(h)[0], 1, 1, 1])
        # [m, k, d, 1]
        expanded_t = tf.tile(tf.expand_dims(tf.expand_dims(t, 1),3), [1, self.config.rel_hidden_size, 1, 1])

        # [m, k]
        htmrt = tf.squeeze(tf.matmul(tf.matmul(expanded_h, expanded_mr), expanded_t), [2,3])

        return tf.tanh(mr1h+mr2t+br+htmrt)

    def test_step(self):
        num_entity = self.data_handler.tot_entity
        # import pdb
        # pdb.set_trace()
        h_vec, r_vec, t_vec = self.embed(self.test_h, self.test_r, self.test_t)
        energy_h = tf.reduce_sum(r_vec*self.layer(self.ent_embeddings, t_vec, expand='t'), -1)
        energy_t = tf.reduce_sum(r_vec*self.layer(h_vec, self.ent_embeddings, expand='h'), -1)

        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        norm_h_vec, norm_r_vec, norm_t_vec = self.embed(self.test_h, self.test_r, self.test_t)
        norm_energy_h = tf.reduce_sum(norm_r_vec*self.layer(self.ent_embeddings, norm_t_vec, expand='t'), -1)
        norm_energy_t = tf.reduce_sum(norm_r_vec*self.layer(norm_h_vec, self.ent_embeddings, expand='h'), -1)

        _, self.head_rank      = tf.nn.top_k(tf.negative(energy_h), k=num_entity)
        _, self.tail_rank      = tf.nn.top_k(tf.negative(energy_t), k=num_entity)
        _, self.norm_head_rank = tf.nn.top_k(tf.negative(norm_energy_h), k=num_entity)
        _, self.norm_tail_rank = tf.nn.top_k(tf.negative(norm_energy_t), k=num_entity)

        return self.head_rank, self.tail_rank, self.norm_head_rank, self.norm_tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def get_embed(self, h, r, t, sess=None):
        """function to get the embedding value in numpy"""
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t 

    def get_proj_embed(self, h, r, t, sess):
        """function to get the projected embedding value in numpy"""
        return self.get_embed(h, r, t, sess)