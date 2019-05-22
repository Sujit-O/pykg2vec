#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("../")
import tensorflow as tf
from core.KGMeta import ModelMeta

class Complex(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    ------------------Paper Authors---------------------------
    ------------------Summary---------------------------------
    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.tot_ent = self.data_stats.tot_entity
        self.tot_rel = self.data_stats.tot_relation
        self.model_name = 'Complex'

    def def_inputs(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.hr_t = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])
        self.rt_h = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])

        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
        k = self.config.hidden_size
        with tf.name_scope("embedding"):
            self.emb_e_real = tf.get_variable(name="emb_e_real", shape=[self.tot_ent, k],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.emb_e_img = tf.get_variable(name="emb_e_img", shape=[self.tot_ent, k],
                                             initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.emb_rel_real = tf.get_variable(name="emb_rel_real", shape=[self.tot_rel, k],
                                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.emb_rel_img = tf.get_variable(name="emb_rel_img", shape=[self.tot_rel, k],
                                               initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.emb_e_real, self.emb_e_img, self.emb_rel_real, self.emb_rel_img]

    def def_loss(self):
        h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img = self.embed(self.h, self.r, self.t)

        h_emb_real, r_emb_real, t_emb_real = self.layer(h_emb_real, r_emb_real, t_emb_real)
        h_emb_img, r_emb_img, t_emb_img = self.layer(h_emb_img, r_emb_img, t_emb_img)

        h_emb_real = tf.squeeze(h_emb_real)
        r_emb_real = tf.squeeze(r_emb_real)
        t_emb_real = tf.squeeze(t_emb_real)
        h_emb_img = tf.squeeze(h_emb_img)
        r_emb_img = tf.squeeze(r_emb_img)
        t_emb_img = tf.squeeze(t_emb_img)

        realrealreal = tf.matmul(h_emb_real * r_emb_real,
                                 tf.transpose(tf.nn.l2_normalize(self.emb_e_real, axis=1)))
        realimgimg = tf.matmul(h_emb_real * r_emb_img,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_img, axis=1)))
        imgrealimg = tf.matmul(h_emb_img * r_emb_real,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_img, axis=1)))
        imgimgreal = tf.matmul(h_emb_img * r_emb_img,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_real, axis=1)))

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred_heads = tf.nn.sigmoid(pred)

        realrealreal = tf.matmul(t_emb_real * r_emb_real,
                                 tf.transpose(tf.nn.l2_normalize(self.emb_e_real, axis=1)))
        realimgimg = tf.matmul(t_emb_real * r_emb_img,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_img, axis=1)))
        imgrealimg = tf.matmul(t_emb_img * r_emb_real,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_img, axis=1)))
        imgimgreal = tf.matmul(t_emb_img * r_emb_img,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_real, axis=1)))

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred_tails = tf.nn.sigmoid(pred)

        hr_t = self.hr_t * (1.0 - self.config.label_smoothing) + 1.0 / self.data_stats.tot_entity
        rt_h = self.rt_h * (1.0 - self.config.label_smoothing) + 1.0 / self.data_stats.tot_entity

        loss_tails = tf.reduce_mean(tf.keras.backend.binary_crossentropy(hr_t, pred_tails))
        loss_heads = tf.reduce_mean(tf.keras.backend.binary_crossentropy(rt_h, pred_heads))

        # reg_losses = tf.nn.l2_loss(self.E) + tf.nn.l2_loss(self.R) + tf.nn.l2_loss(self.W)

        self.loss = loss_heads + loss_tails #+ self.config.lmbda * reg_losses

    def def_layer(self):
        self.inp_drop = tf.keras.layers.Dropout(rate=self.config.input_dropout)

    def layer(self, h, r, t):
        h = tf.squeeze(h)
        r = tf.squeeze(r)
        t = tf.squeeze(t)

        h = self.inp_drop(h)
        r = self.inp_drop(r)
        t = self.inp_drop(t)

        return h,r,t

    def test_batch(self):
        h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img = self.embed(self.test_h_batch,
                                                                                         self.test_r_batch,
                                                                                         self.test_t_batch)

        h_emb_real, r_emb_real, t_emb_real = self.layer(h_emb_real, r_emb_real, t_emb_real)
        h_emb_img, r_emb_img, t_emb_img = self.layer(h_emb_img, r_emb_img, t_emb_img)

        h_emb_real = tf.squeeze(h_emb_real)
        r_emb_real = tf.squeeze(r_emb_real)
        t_emb_real = tf.squeeze(t_emb_real)
        h_emb_img = tf.squeeze(h_emb_img)
        r_emb_img = tf.squeeze(r_emb_img)
        t_emb_img = tf.squeeze(t_emb_img)

        realrealreal = tf.matmul(h_emb_real * r_emb_real,
                                 tf.transpose(tf.nn.l2_normalize(self.emb_e_real, axis=1)))
        realimgimg = tf.matmul(h_emb_real * r_emb_img,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_img, axis=1)))
        imgrealimg = tf.matmul(h_emb_img * r_emb_real,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_img, axis=1)))
        imgimgreal = tf.matmul(h_emb_img * r_emb_img,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_real, axis=1)))

        pred_tails = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred_tails = tf.nn.sigmoid(pred_tails)

        realrealreal = tf.matmul(t_emb_real * r_emb_real,
                                 tf.transpose(tf.nn.l2_normalize(self.emb_e_real, axis=1)))
        realimgimg = tf.matmul(t_emb_real * r_emb_img,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_img, axis=1)))
        imgrealimg = tf.matmul(t_emb_img * r_emb_real,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_img, axis=1)))
        imgimgreal = tf.matmul(t_emb_img * r_emb_img,
                               tf.transpose(tf.nn.l2_normalize(self.emb_e_real, axis=1)))

        pred_heads = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred_heads = tf.nn.sigmoid(pred_heads)

        _, head_rank = tf.nn.top_k(pred_tails, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(pred_heads, k=self.data_stats.tot_entity)

        return head_rank, tail_rank

    def embed(self, h,r,t):
        """function to get the embedding value"""
        norm_emb_e_real = tf.nn.l2_normalize(self.emb_e_real, axis=1)
        norm_emb_e_img = tf.nn.l2_normalize(self.emb_e_img, axis=1)
        norm_emb_rel_real = tf.nn.l2_normalize(self.emb_rel_real, axis=1)
        norm_emb_rel_img = tf.nn.l2_normalize(self.emb_rel_img, axis=1)

        h_emb_real = tf.nn.embedding_lookup(norm_emb_e_real, h)
        t_emb_real = tf.nn.embedding_lookup(norm_emb_e_real, t)

        h_emb_img = tf.nn.embedding_lookup(norm_emb_e_img, h)
        t_emb_img = tf.nn.embedding_lookup(norm_emb_e_img, t)

        r_emb_real = tf.nn.embedding_lookup(norm_emb_rel_real, r)
        r_emb_img = tf.nn.embedding_lookup(norm_emb_rel_img, r)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img 

    def get_embed(self, e, r, sess=None):
        """function to get the embedding value in numpy"""
        emb_e_real, rel_emb_real, emb_e_img, rel_emb_img = self.embed(h,r, t)
        emb_e_real, rel_emb_real, emb_e_img, rel_emb_img = sess.run([emb_e_real, rel_emb_real, emb_e_img, rel_emb_img])
        return emb_e_real, rel_emb_real, emb_e_img, rel_emb_img

    def get_proj_embed(self, e, r, sess):
        """function to get the projected embedding value in numpy"""
        return self.get_embed(e, r, sess)


if __name__ == '__main__':
    # Unit Test Script with tensorflow Eager Execution
    import tensorflow as tf
    import numpy as np

    tf.enable_eager_execution()
    batch = 128
    k = 100
    tot_ent = 14700
    tot_rel = 2600
    train = True
    e1 = np.random.randint(0, tot_ent, size=(batch, 1))
    print('pos_r_e:', e1)
    r = np.random.randint(0, tot_rel, size=(batch, 1))
    print('pos_r_e:', r)
    e2 = np.random.randint(0, tot_ent, size=(batch, 1))
    print('pos_t_e:', e2)
    r_rev = np.random.randint(0, tot_rel, size=(batch, 1))
    print('pos_r_e:', r_rev)

    emb_e_real = tf.get_variable(name="emb_e_real", shape=[tot_ent, k],
                                 initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    emb_e_img = tf.get_variable(name="emb_e_img", shape=[tot_ent, k],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    emb_rel_real = tf.get_variable(name="emb_rel_real", shape=[tot_rel, k],
                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    emb_rel_img = tf.get_variable(name="emb_rel_img", shape=[tot_rel, k],
                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    norm_emb_e_real = tf.nn.l2_normalize(emb_e_real, axis=1)
    norm_emb_e_img = tf.nn.l2_normalize(emb_e_img, axis=1)
    norm_emb_rel_real = tf.nn.l2_normalize(emb_rel_real, axis=1)
    norm_emb_rel_img = tf.nn.l2_normalize(emb_rel_img, axis=1)

    emb_e1_real = tf.nn.embedding_lookup(norm_emb_e_real, e1)
    rel_emb_real = tf.nn.embedding_lookup(norm_emb_rel_real, r)
    emb_e1_img = tf.nn.embedding_lookup(norm_emb_e_img, e1)
    rel_emb_img = tf.nn.embedding_lookup(norm_emb_rel_img, r)

    e1_embedded_real = tf.keras.layers.Dropout(rate=0.2)(emb_e1_real)
    rel_embedded_real = tf.keras.layers.Dropout(rate=0.2)(rel_emb_real)
    e1_embedded_img = tf.keras.layers.Dropout(rate=0.2)(emb_e1_img)
    rel_embedded_img = tf.keras.layers.Dropout(rate=0.2)(rel_emb_img)

    e1_embedded_real = tf.squeeze(e1_embedded_real)
    rel_embedded_real = tf.squeeze(rel_embedded_real)
    e1_embedded_img = tf.squeeze(e1_embedded_img)
    rel_embedded_img = tf.squeeze(rel_embedded_img)

    import pdb

    pdb.set_trace()
    print(e1_embedded_real.shape, rel_embedded_real.shape, emb_e_real.shape)
    realrealreal = tf.matmul(e1_embedded_real * rel_embedded_real, tf.transpose(tf.nn.l2_normalize(emb_e_real, axis=1)))
    realimgimg = tf.matmul(e1_embedded_real * rel_embedded_img, tf.transpose(tf.nn.l2_normalize(emb_e_img, axis=1)))
    imgrealimg = tf.matmul(e1_embedded_img * rel_embedded_real, tf.transpose(tf.nn.l2_normalize(emb_e_img, axis=1)))
    imgimgreal = tf.matmul(e1_embedded_img * rel_embedded_img,  tf.transpose(tf.nn.l2_normalize(emb_e_real, axis=1)))

    pred = realrealreal + realimgimg + imgrealimg - imgimgreal
    pred = tf.nn.sigmoid(pred)
