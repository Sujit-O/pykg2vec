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
        self.dense_last_dim = {50: 2592, 100: 5184, 200: 10368}
        if self.config.hidden_size not in self.dense_last_dim:
            raise NotImplementedError("The hidden dimension is not supported!")
        self.last_dim = self.dense_last_dim[self.config.hidden_size]

    # def def_inputs(self):
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])
        self.test_h = tf.placeholder(tf.int32, [1])
        self.test_t = tf.placeholder(tf.int32, [1])
        self.test_r = tf.placeholder(tf.int32, [1])

    # def def_parameters(self):
        num_total_ent = self.data_handler.tot_entity
        num_total_rel = self.data_handler.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        with tf.name_scope("activation_bias"):
            self.b = tf.get_variable(name="bias", shape=[self.config.batch_size, num_total_ent],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        # self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.b]
        self.loss = None

    def def_loss(self):
        ent_emb_norm = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        rel_emb_norm = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        pos_h_e = tf.nn.embedding_lookup(ent_emb_norm, self.pos_h)
        pos_r_e = tf.nn.embedding_lookup(rel_emb_norm, self.pos_r)
        pos_t_e = tf.nn.embedding_lookup(ent_emb_norm, self.pos_t)

        neg_h_e = tf.nn.embedding_lookup(ent_emb_norm, self.neg_h)
        neg_r_e = tf.nn.embedding_lookup(rel_emb_norm, self.neg_r)
        neg_t_e = tf.nn.embedding_lookup(ent_emb_norm, self.neg_t)

        # prepare h,r
        stacked_inputs_h = tf.concat([pos_h_e, neg_h_e], 0)
        stacked_inputs_r = tf.concat([pos_r_e, neg_r_e], 0)
        stacked_inputs_t = tf.concat([pos_t_e, neg_t_e], 0)

        stacked_inputs_h = tf.reshape(stacked_inputs_h, [-1, 10, 20, 1])
        stacked_inputs_r = tf.reshape(stacked_inputs_r, [-1, 10, 20, 1])
        stacked_inputs_t = tf.reshape(stacked_inputs_t, [-1, 10, 20, 1])

        stacked_inputs_hr = tf.concat([stacked_inputs_h, stacked_inputs_r], 1)
        stacked_inputs_rt = tf.concat([stacked_inputs_r, stacked_inputs_t], 1)

        y_hr_t = tf.concat([tf.ones(self.config.batch_size // 2, ), tf.zeros(self.config.batch_size // 2, )], 0)
        e2_multi_hr_t = tf.scalar_mul((1.0 - self.config.label_smoothing), y_hr_t) + (1.0 / self.config.hidden_size)

        pred_h_rt = self.layer_hr_t(stacked_inputs_rt)

        self.loss =  tf.reduce_mean(tf.keras.backend.binary_crossentropy(e2_multi_hr_t, pred_hr_t))

    def layer(self, st_inp):
        # batch normalization in the first axis
        x = tf.keras.layers.BatchNormalization()(st_inp, training=train)
        # input dropout
        x = tf.keras.layers.Dropout(rate=self.config.input_dropout)(x)
        # 2d convolution layer, output channel =32, kernel size = 3,3
        x = tf.keras.layers.Conv2D(32, [3, 3], strides=(1, 1), padding='valid', activation=None)(x)
        # batch normalization across feature dimension
        x = tf.keras.layers.BatchNormalization()(x, training=train)
        # first non-linear activation
        x = tf.nn.relu(x)
        # feature dropout
        x = tf.keras.layers.Dropout(rate=self.config.feature_map_dropout)(x)
        # reshape the tensor to get the batch size
        '''10368 with k=200,5184 with k=100, 2592 with k=50'''
        x = tf.reshape(x, [self.config.batch_size, self.last_dim])
        # pass the feature through fully connected layer, output size = batch size, hidden size
        x = tf.keras.layers.Dense(units=self.config.hidden_size)(x)
        # dropout in the hidden layer
        x = tf.keras.layers.Dropout(rate=self.config.hidden_dropout)(x)
        # batch normalization across feature dimension
        x = tf.keras.layers.BatchNormalization()(x, training=train)
        # second non-linear activation
        x = tf.nn.relu(x)
        # project and get inner product with the tail triple
        x = tf.matmul(x, tf.transpose(tf.nn.l2_normalize(self.ent_embeddings, axis=1)))
        # add a bias value
        x = tf.add(x, self.b)
        # sigmoid activation
        return tf.nn.sigmoid(x)

    def test_step(self):
        num_entity = self.data_handler.tot_entity
        embed_dim = self.config.hidden_size
        # import pdb
        # pdb.set_trace()
        ent_emb_norm = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        rel_emb_norm = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        pos_h_e = tf.nn.embedding_lookup(ent_emb_norm, self.pos_h)
        pos_r_e = tf.nn.embedding_lookup(rel_emb_norm, self.pos_r)
        pos_t_e = tf.nn.embedding_lookup(ent_emb_norm, self.pos_t)

        # prepare h,r
        stacked_inputs_h = tf.reshape(pos_h_e, [-1, 10, 20, 1])
        stacked_inputs_r = tf.reshape(pos_r_e, [-1, 10, 20, 1])
        stacked_inputs_hr = tf.concat([stacked_inputs_h, stacked_inputs_r], 1)

        pred_val_head = self.layer(stacked_inputs_hr, train=False)
        _, head_rank = tf.nn.top_k(pred_val_head, k=num_entity)

        pred_val_tail = []
        for h_i in range(num_entity):
            pos_h_e = tf.nn.embedding_lookup(ent_emb_norm, self.pos_h)
            pos_r_e = tf.nn.embedding_lookup(rel_emb_norm, self.pos_r)

            # prepare h,r
            stacked_inputs_h = tf.reshape(pos_h_e, [1, 10, -1, 1])
            stacked_inputs_r = tf.reshape(pos_r_e, [1, 10, -1, 1])
            stacked_inputs_hr = tf.concat([stacked_inputs_h, stacked_inputs_r], 1)
            pred_val_tail.append(self.layer(stacked_inputs_hr, pos_t_e, train=False, tail=True))

        _, tail_rank = tf.nn.top_k(pred_val_tail, k=num_entity)

        return head_rank, tail_rank, None, None

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
