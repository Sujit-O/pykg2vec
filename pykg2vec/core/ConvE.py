#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("../")
import tensorflow as tf
from core.KGMeta import ModelMeta


class ConvE(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    Convolutional 2D Knowledge Graph Embeddings
    ------------------Paper Authors---------------------------
    Tim Dettmers∗
    Università della Svizzera italiana
    tim.dettmers@gmail.com
    Pasquale Minervini, Pontus Stenetorp, Sebastian Riedel
    University College London
    {p.minervini,p.stenetorp,s.riedel}@cs.ucl.ac.uk
    ------------------Summary---------------------------------
    ConvE is a multi-layer convolutional network model for link prediction,
    it is a embedding model which is highly parameter efficient.
    """

    def __init__(self, config=None, data_handler=None):
        self.config = config
        self.data_handler = data_handler
        self.model_name = 'ConvE'

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
        with tf.name_scope("activation bias"):
            self.b = tf.get_variable(name="bias", shape=[self.config.batch_size, num_total_ent],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        with tf.name_scope("transformation matrix"):
            self.W = tf.get_variable(name="Wmatrix", shape=[k, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings,self.b, self.W]

    def def_loss(self):
        # ent_emb_norm = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        # rel_emb_norm = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
        pos_r_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_r)
        pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)

        neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
        neg_r_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_r)
        neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)

        # prepare h,r
        stacked_inputs_h = tf.concat([pos_h_e, neg_h_e], 0)
        stacked_inputs_r = tf.concat([pos_r_e, neg_r_e], 0)
        stacked_inputs_h = tf.reshape(stacked_inputs_h, [self.config.batch_size, 10, -1, 1])
        stacked_inputs_r = tf.reshape(stacked_inputs_r, [self.config.batch_size, 10, -1, 1])
        stacked_inputs = tf.concat([stacked_inputs_h, stacked_inputs_r], 1)

        # prepare t
        stacked_inputs_t = tf.concat([pos_t_e, neg_t_e], 0)
        y = tf.concat([tf.ones(self.config.batch_size,), tf.zeros(self.config.batch_size,)],0)
        e2_multi = tf.scalar_mul((1.0 - self.config.label_smoothing), y)+(1.0 / self.config.hidden_size)
        pred_val = self.layer(stacked_inputs,stacked_inputs_t)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=e2_multi, logits=pred_val)

    def layer(self, st_inp,st_inp_t, train= True):
        # batch normalization in the first axis
        x = tf.layers.batch_normalization(st_inp, axis=0)
        # input dropout
        x = tf.layers.dropout(x, rate=self.config.input_dropout)
        # 2d convolution layer, output channel =32, kernel size = 3,3
        x = tf.layers.conv2d(x, 32, [3, 3], strides=(1, 1), padding='valid', activation=None)
        # batch normalization across feature dimension
        x = tf.layers.batch_normalization(x, axis=1)
        # first non-linear activation
        x = tf.nn.relu(x)
        # feature dropout
        x = tf.layers.dropout(x, rate=self.config.featre_map_dropout)
        # reshape the tensor to get the batch size
        x = tf.reshape(x, [self.config.batch_size,-1])
        # pass the feature through fully connected layer, output size = batch size, hidden size
        x = tf.layers.dense(x,units=self.config.hidden_size)
        # dropout in the hidden layer
        x = tf.layers.dropout(x, rate=self.config.hidden_dropout)
        # batch normalization across feature dimension
        x = tf.layers.batch_normalization(x, axis=1)
        # second non-linear activation
        x = tf.nn.relu(x)
        # matrix multiplication to project the tensor into k dimension

        x = tf.matmul(x, self.W)
        # add a bias value
        x = tf.add(x, self.b)
        # multiplication with
        if train:
            x = tf.reduce_sum(tf.matmul(x, tf.transpose(st_inp_t)),1)
        else:
            x = tf.reduce_sum(tf.matmul(x, tf.transpose(self.ent_embeddings)),1)
        # sigmoid activation

        pred = tf.nn.sigmoid(x)

        return pred

    def test_step(self):
        num_entity = self.data_handler.tot_entity
        # import pdb
        # pdb.set_trace()
        h_vec, r_vec, t_vec = self.embed(self.test_h, self.test_r, self.test_t)
        energy_h = tf.reduce_sum(r_vec * self.layer(self.ent_embeddings, t_vec, expand='t'), -1)
        energy_t = tf.reduce_sum(r_vec * self.layer(h_vec, self.ent_embeddings, expand='h'), -1)

        self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        norm_h_vec, norm_r_vec, norm_t_vec = self.embed(self.test_h, self.test_r, self.test_t)
        norm_energy_h = tf.reduce_sum(norm_r_vec * self.layer(self.ent_embeddings, norm_t_vec, expand='t'), -1)
        norm_energy_t = tf.reduce_sum(norm_r_vec * self.layer(norm_h_vec, self.ent_embeddings, expand='h'), -1)

        _, self.head_rank = tf.nn.top_k(tf.negative(energy_h), k=num_entity)
        _, self.tail_rank = tf.nn.top_k(tf.negative(energy_t), k=num_entity)
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


if __name__=='__main__':
    import tensorflow as tf
    tf.enable_eager_execution()
    batch = 128
    embed_dim = 100
    tot_entity = 147000
    pos_h_e = tf.random_normal([batch // 2, embed_dim])
    print('pos_r_e:', pos_h_e)
    pos_r_e = tf.random_normal([batch // 2, embed_dim])
    print('pos_r_e:', pos_r_e)
    pos_t_e = tf.random_normal([batch // 2, embed_dim])
    print('pos_t_e:', pos_t_e)
    neg_h_e = tf.random_normal([batch // 2, embed_dim])
    print('neg_h_e:', neg_h_e)
    neg_r_e = tf.random_normal([batch // 2, embed_dim])
    print('neg_r_e:', neg_r_e)
    neg_t_e = tf.random_normal([batch // 2, embed_dim])
    print('neg_t_e:', neg_t_e)
    stacked_inputs_pos = tf.concat([pos_h_e, neg_h_e], 0)
    stacked_inputs_neg = tf.concat([pos_r_e, neg_r_e], 0)
    stacked_inputs_pos = tf.reshape(stacked_inputs_pos, [-1, 10, 20, 1])
    stacked_inputs_neg = tf.reshape(stacked_inputs_neg, [-1, 10, 20, 1])
    stacked_inputs = tf.concat([stacked_inputs_pos, stacked_inputs_neg], 1)
    print('stacked_inputs:', stacked_inputs)
    x = tf.layers.batch_normalization(x, axis=0)
    print("x_batch normalize:", x)
    x = tf.layers.dropout(x, rate=0.2)
    print("x_dropped out:", x)
    x = tf.layers.conv2d(x, 32, [3, 3], strides=(1, 1), padding='valid', activation=None)
    print("x_conv2d:", x)
    x = tf.layers.batch_normalization(x, axis=1)
    print("x_batch normalize:", x)
    x = tf.nn.relu(x)
    print("x_relu activation:", x)
    x = tf.layers.dropout(x, rate=0.2)
    print("x_dropped out:", x)
    x = tf.reshape(x, [batch, -1])
    print("x_reshaped:", x)
    x = tf.layers.dense(x, units=embed_dim)
    print("x_dense:", x)
    x = tf.layers.dropout(x, rate=0.3)
    print("x_droppedout:", x)
    x = tf.layers.batch_normalization(x, axis=1)
    print("x_batch normalize:", x)
    x = tf.nn.relu(x)
    print("x_relu activation:", x)
    W = tf.get_variable(name="ent_embedding", shape=[tot_entity, embed_dim],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    print("ent_embedding:", W)
    x = tf.matmul(x, tf.transpose(W))
    print("x_mul with ent_embeeding:", x)
    b = tf.get_variable(name="b", shape=[batch, tot_entity],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    print("bias:", b)
    x = tf.add(x, b)
    print("x_added with bias:", x)
    pred = tf.nn.sigmoid(x)
    print("prediction:", pred)