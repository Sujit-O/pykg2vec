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


class DistMult(ModelMeta):
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
        k = self.config.hidden_size
        with tf.name_scope("embedding"):
            self.emb_e = tf.get_variable(name="emb_e_real", shape=[self.tot_ent, k],
                                         initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.emb_rel = tf.get_variable(name="emb_rel_real", shape=[self.tot_rel, k],
                                           initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.emb_e, self.emb_rel]

    def def_loss(self):
        e1_emb, rel_emb = self.embed(self.e1, self.r)

        e1_emb, rel_emb = self.layer(e1_emb, rel_emb)

        e1_emb = tf.squeeze(e1_emb)
        rel_emb = tf.squeeze(rel_emb)

        pred = tf.matmul(e1_emb * rel_emb, tf.transpose(tf.nn.l2_normalize(self.emb_e, axis=1)))
        pred = tf.nn.sigmoid(pred)

        e2_multi1 = self.e2_multi1 * (1.0 - self.config.label_smoothing) + 1.0 / self.data_stats.tot_entity

        loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(e2_multi1, pred))

        reg_losses = tf.nn.l2_loss(self.emb_e) + tf.nn.l2_loss(self.emb_rel)

        self.loss = loss + self.config.lmbda * reg_losses

    def def_layer(self):
        self.inp_drop = tf.keras.layers.Dropout(rate=self.config.input_dropout)

    def layer(self, e, rel):
        e = tf.squeeze(e)
        rel = tf.squeeze(rel)
        e = self.inp_drop(e)
        rel = self.inp_drop(rel)
        return e, rel

    def test_batch(self):
        e1_emb, rel_emb = self.embed(self.test_e1, self.test_r)
        e2_emb, r_rev_emb = self.embed(self.test_e2, self.test_r_rev)

        e1_emb, rel_emb = self.layer(e1_emb, rel_emb)
        e2_emb, r_rev_emb = self.layer(e2_emb, r_rev_emb)

        e1_emb = tf.squeeze(e1_emb)
        e2_emb = tf.squeeze(e2_emb)
        rel_emb = tf.squeeze(rel_emb)
        r_rev_emb = tf.squeeze(r_rev_emb)

        hr_pred = tf.matmul(e1_emb * rel_emb, tf.transpose(tf.nn.l2_normalize(self.emb_e, axis=1)))
        hr_pred = tf.nn.sigmoid(hr_pred)

        tr_pred = tf.matmul(e2_emb * r_rev_emb, tf.transpose(tf.nn.l2_normalize(self.emb_e, axis=1)))
        tr_pred = tf.nn.sigmoid(tr_pred)

        _, head_rank = tf.nn.top_k(-hr_pred, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(-tr_pred, k=self.data_stats.tot_entity)

        return head_rank, tail_rank

    def embed(self, e1, r):
        """function to get the embedding value"""
        norm_emb_e = tf.nn.l2_normalize(self.emb_e, axis=1)
        norm_emb_rel = tf.nn.l2_normalize(self.emb_rel, axis=1)

        emb_e1 = tf.nn.embedding_lookup(norm_emb_e, e1)
        rel_emb = tf.nn.embedding_lookup(norm_emb_rel, r)

        return emb_e1, rel_emb

    def get_embed(self, e, r, sess=None):
        """function to get the embedding value in numpy"""
        emb_e, rel_emb = self.embed(e, r)
        emb_e, rel_emb = sess.run([emb_e, rel_emb])
        return emb_e, rel_emb

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
    imgimgreal = tf.matmul(e1_embedded_img * rel_embedded_img, tf.transpose(tf.nn.l2_normalize(emb_e_real, axis=1)))

    pred = realrealreal + realimgimg + imgrealimg - imgimgreal
    pred = tf.nn.sigmoid(pred)
