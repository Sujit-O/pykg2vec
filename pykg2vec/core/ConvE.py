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

    def __init__(self, config=None):
        self.config = config
        with open(self.config.tmp_data / 'data_stats.pkl', 'rb') as f:
            self.data_stats = pickle.load(f)
        self.model_name = 'ConvE'
        self.dense_last_dim = {50: 2592, 100: 5184, 200: 10368}
        if self.config.hidden_size not in self.dense_last_dim:
            raise NotImplementedError("The hidden dimension is not supported!")
        self.last_dim = self.dense_last_dim[self.config.hidden_size]

        self.def_inputs()
        self.def_parameters()
        self.def_layer()
        self.def_loss()

    def def_inputs(self):
        self.e1 = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.int32, [None])
        self.e2_multi1 = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])

        self.test_e1 = tf.placeholder(tf.int32, [None])
        self.test_e2 = tf.placeholder(tf.int32, [None])
        self.test_r = tf.placeholder(tf.int32, [None])
        self.test_r_rev = tf.placeholder(tf.int32, [None])
        self.test_e2_multi1 = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])
        self.test_e2_multi2 = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])

    def def_parameters(self):
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        with tf.name_scope("activation_bias"):
            self.b = tf.get_variable(name="bias", shape=[self.config.batch_size, num_total_ent],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.b]

    def def_layer(self):
        self.bn0 = tf.keras.layers.BatchNormalization(trainable=True)
        self.inp_drop = tf.keras.layers.Dropout(rate=self.config.input_dropout)
        self.conv2d_1 = tf.keras.layers.Conv2D(32, [3, 3], strides=(1, 1), padding='valid', activation=None,
                                               use_bias=True)
        self.bn1 = tf.keras.layers.BatchNormalization(trainable=True)
        self.feat_drop = tf.keras.layers.Dropout(rate=self.config.feature_map_dropout)
        self.fc1 = tf.keras.layers.Dense(units=self.config.hidden_size)
        self.hidden_drop = tf.keras.layers.Dropout(rate=self.config.hidden_dropout)
        self.bn2 = tf.keras.layers.BatchNormalization(trainable=True)

    def forward(self, st_inp):
        # batch normalization in the first axis
        x = self.bn0(st_inp)
        # input dropout
        x = self.inp_drop(x)
        # 2d convolution layer, output channel =32, kernel size = 3,3
        x = self.conv2d_1(x)
        # batch normalization across feature dimension
        x = self.bn1(x)
        # first non-linear activation
        x = tf.nn.relu(x)
        # feature dropout
        x = self.feat_drop(x)
        # reshape the tensor to get the batch size
        '''10368 with k=200,5184 with k=100, 2592 with k=50'''
        x = tf.reshape(x, [self.config.batch_size, self.last_dim])
        # pass the feature through fully connected layer, output size = batch size, hidden size
        x = self.fc1(x)
        # dropout in the hidden layer
        x = self.hidden_drop(x)
        # batch normalization across feature dimension
        x = self.bn2(x)
        # second non-linear activation
        x = tf.nn.relu(x)
        # project and get inner product with the tail triple
        x = tf.matmul(x, tf.transpose(tf.nn.l2_normalize(self.ent_embeddings, axis=1)))
        # add a bias value
        x = tf.add(x, self.b)
        # sigmoid activation
        return tf.nn.sigmoid(x)

    def def_loss(self):
        ent_emb_norm = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        rel_emb_norm = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        e1 = tf.nn.embedding_lookup(ent_emb_norm, self.e1)
        r = tf.nn.embedding_lookup(rel_emb_norm, self.r)

        stacked_e = tf.reshape(e1, [-1, 10, 20, 1])
        stacked_r = tf.reshape(r, [-1, 10, 20, 1])

        stacked_er = tf.concat([stacked_e, stacked_r], 1)

        e2_multi1 = self.e2_multi1 * (1.0 - self.config.label_smoothing) + 1.0 / self.data_stats.tot_entity
        # e2_multi1 = tf.reshape(e2_multi1, [self.config.batch_size, self.data_stats.tot_entity])
        pred = self.forward(stacked_er)

        loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(e2_multi1, pred))

        reg_losses = tf.nn.l2_loss(self.ent_embeddings) + tf.nn.l2_loss(self.rel_embeddings)

        self.loss = loss + self.config.lmbda * reg_losses


    def test_batch(self):
        ent_emb_norm = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        rel_emb_norm = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        e1 = tf.nn.embedding_lookup(ent_emb_norm, self.test_e1)
        e2 = tf.nn.embedding_lookup(ent_emb_norm, self.test_e2)

        r = tf.nn.embedding_lookup(rel_emb_norm, self.test_r)
        r_rev = tf.nn.embedding_lookup(rel_emb_norm, self.test_r_rev)

        stacked_head_e = tf.reshape(e1, [-1, 10, 20, 1])
        stacked_head_r = tf.reshape(r, [-1, 10, 20, 1])

        stacked_tail_e = tf.reshape(e2, [-1, 10, 20, 1])
        stacked_tail_r = tf.reshape(r_rev, [-1, 10, 20, 1])

        stacked_hr = tf.concat([stacked_head_e, stacked_head_r], 1)
        stacked_tr = tf.concat([stacked_tail_e, stacked_tail_r], 1)

        # e2_multi1 = tf.scalar_mul((1.0 - self.config.label_smoothing),
        #                           self.test_e2_multi1) + (1.0 / self.data_stats.tot_entity)
        # e2_multi2 = tf.scalar_mul((1.0 - self.config.label_smoothing),
        #                           self.test_e2_multi2) + (1.0 / self.data_stats.tot_entity)

        pred4head = self.forward(stacked_hr)
        pred4tail = self.forward(stacked_tr)
        #
        # head_vec = tf.keras.backend.binary_crossentropy(e2_multi1, pred4head)
        # tail_vec = tf.keras.backend.binary_crossentropy(e2_multi2, pred4tail)

        _, head_rank = tf.nn.top_k(-pred4head, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(-pred4tail, k=self.data_stats.tot_entity)

        return head_rank, tail_rank

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


if __name__ == '__main__':
    # Unit Test Script with tensorflow Eager Execution
    import tensorflow as tf

    tf.enable_eager_execution()
    batch = 128
    embed_dim = 100
    tot_entity = 147000
    train = True
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
    stacked_inputs_e = tf.concat([pos_h_e, neg_h_e], 0)
    stacked_inputs_r = tf.concat([pos_r_e, neg_r_e], 0)
    stacked_inputs_e = tf.reshape(stacked_inputs_e, [batch, 10, -1, 1])
    stacked_inputs_r = tf.reshape(stacked_inputs_r, [batch, 10, -1, 1])
    stacked_inputs = tf.concat([stacked_inputs_e, stacked_inputs_r], 1)
    stacked_inputs_t = tf.concat([pos_t_e, neg_t_e], 0)
    print('stacked_inputs:', stacked_inputs)
    x = tf.layers.batch_normalization(stacked_inputs, axis=0)
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
    W = tf.get_variable(name="ent_embedding", shape=[embed_dim, embed_dim],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    print("ent_embedding:", W)
    x = tf.matmul(x, tf.transpose(W))
    print("x_mul with ent_embeeding:", x)
    b = tf.get_variable(name="b", shape=[batch, tot_entity],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    print("bias:", b)
    x = tf.add(x, b)
    print("x_added with bias:", x)
    ent_embeddings = tf.get_variable(name="ent_embedding", shape=[tot_entity, embed_dim],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    if train:
        x = tf.reduce_sum(tf.matmul(x, tf.transpose(stacked_inputs_t)), 1)
    else:
        x = tf.reduce_sum(tf.matmul(x, tf.transpose(ent_embeddings)), 1)
    pred = tf.nn.sigmoid(x)
    print("prediction:", pred)

    import tensorflow as tf

    tf.enable_eager_execution()
    batch = 128
    embed_dim = 100
    tot_entity = 147000
    train = False
    input_dropout = 0.2
    input_dropout = 0.2
    hidden_dropout = 0.3
    feature_map_dropout = 0.2
    ent_embeddings = tf.get_variable(name="ent_embedding", shape=[tot_entity, embed_dim],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    W = tf.get_variable(name="ent_embedding", shape=[embed_dim, embed_dim],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name="b", shape=[batch, embed_dim],
                        initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    print("ent_embedding:", W)
    pos_h_e = tf.random_normal([1, embed_dim])
    print('pos_r_e:', pos_h_e)
    pos_r_e = tf.random_normal([1, embed_dim])
    print('pos_r_e:', pos_r_e)
    pos_t_e = tf.random_normal([1, embed_dim])
    stacked_inputs_h = tf.reshape(pos_h_e, [1, 10, -1, 1])
    stacked_inputs_r = tf.reshape(pos_r_e, [1, 10, -1, 1])
    stacked_inputs_hr = tf.concat([stacked_inputs_h, stacked_inputs_r], 1)
    x = tf.layers.batch_normalization(stacked_inputs_hr, axis=0)
    x = tf.layers.dropout(x, rate=input_dropout)
    x = tf.layers.conv2d(x, 32, [3, 3], strides=(1, 1), padding='valid', activation=None)
    x = tf.layers.batch_normalization(x, axis=1)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=feature_map_dropout)
    if train:
        x = tf.reshape(x, [batch_size, -1])
    else:
        x = tf.reshape(x, [1, -1])

    x = tf.layers.dense(x, units=embed_dim)
    x = tf.layers.dropout(x, rate=hidden_dropout)
    x = tf.layers.batch_normalization(x, axis=1)
    x = tf.nn.relu(x)
    x = tf.matmul(x, W)
    if train:
        x = tf.add(x, b)
    else:
        x = tf.add(x, tf.slice(b, [0, 0], [1, embed_dim]))

    if train:
        x = tf.reduce_sum(tf.matmul(x, tf.transpose(st_inp_t)), 1)
        x = tf.reduce_sum(tf.matmul(x, tf.transpose(ent_emb_norm)), 1)
    else:
        ent_emb_norm = tf.nn.l2_normalize(ent_embeddings, axis=1)
        x = tf.matmul(x, tf.transpose(ent_emb_norm))
    pred = tf.nn.sigmoid(x)
