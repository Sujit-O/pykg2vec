#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class RotatE(ModelMeta):
    """ `Rotate-Knowledge graph embedding by relation rotation in complex space`_

        RotatE models the entities and the relations in the complex vector space.
        The translational relation in RotatE is defined as the element-wise 2D
        rotation in which the head entity h will be rotated to the tail entity t by
        multiplying the unit-length relation r in complex number form.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.RotatE import RotatE
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = RotatE()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _Rotate-Knowledge graph embedding by relation rotation in complex space:
            https://openreview.net/pdf?id=HkgEQnRqYQ
    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.model_name = 'RotatE'

    def def_inputs(self):
        """Defines the inputs to the model.

           Attributes:
              pos_h (Tensor): Positive Head entities ids.
              pos_r (Tensor): Positive Relation ids of the triple.
              pos_t (Tensor): Positive Tail entity ids of the triple.
              neg_h (Tensor): Negative Head entities ids.
              neg_r (Tensor): Negative Relation ids of the triple.
              neg_t (Tensor): Negative Tail entity ids of the triple.
              test_h_batch (Tensor): Batch of head ids for testing.
              test_r_batch (Tensor): Batch of relation ids for testing
              test_t_batch (Tensor): Batch of tail ids for testing.
        """
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings_real (Tensor Variable): Lookup variable containing real values of the entities.
               ent_embeddings_imag (Tensor Variable): Lookup variable containing imaginary values of the entities.
               rel_embeddings_real (Tensor Variable): Lookup variable containing real values of the relations.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation

        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings_real = tf.get_variable(name="ent_embeddings_real", shape=[num_total_ent, k],
                                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.ent_embeddings_imag = tf.get_variable(name="ent_embeddings_imag", shape=[num_total_ent, k],
                                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings_real = tf.get_variable(name="rel_embeddings_real", shape=[num_total_rel, k],
                                                       initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.ent_embeddings_real, self.ent_embeddings_imag, self.rel_embeddings_real]

    def comp_mul_and_min(self, hr, hi, rr, ri, tr, ti):
        """Calculates training score for loss function.

            Args:
                hi(Tensor): Imaginary part of the head embedding.
                hr(Tensor): Real part of the head embedding.
                ri(Tensor): Imaginary part of the tail embedding.
                rr(Tensor): Real part of the tail embedding.
                ti(Tensor): Imaginary part of the relation embedding.
                tr(Tensor): Real part of the relation embedding.

            Returns:
                Tensors: Returns a tensor
        """
        score_r = hr * rr - hi * ri - tr
        score_i = hr * ri + hi * rr - ti
        return tf.reduce_sum(tf.sqrt(score_r ** 2 + score_i ** 2), -1)

    def comp_mul_and_min_4_test(self, hr, hi, rr, ri, tr, ti):
        """Calculates test score for loss function.

            Args:
                hi(Tensor): Imaginary part of the head embedding.
                hr(Tensor): Real part of the head embedding.
                ri(Tensor): Imaginary part of the tail embedding.
                rr(Tensor): Real part of the tail embedding.
                ti(Tensor): Imaginary part of the relation embedding.
                tr(Tensor): Real part of the relation embedding.

            Returns:
                Tensors: Returns a tensor
        """

        rr = tf.expand_dims(rr, axis=1)
        ri = tf.expand_dims(ri, axis=1)

        score_r = tf.cond(tf.shape(hr)[0] < tf.shape(tr)[0],
                          lambda: tf.expand_dims(hr, axis=1)*rr-tf.expand_dims(hi, axis=1)*ri-tr,
                          lambda: hr*rr-hi*ri-tf.expand_dims(tr, axis=1))

        score_i = tf.cond(tf.shape(hr)[0] < tf.shape(tr)[0],
                          lambda: tf.expand_dims(hr, axis=1) * ri + tf.expand_dims(hi, axis=1) * rr - ti,
                          lambda:  hr * ri + hi * rr - tf.expand_dims(ti, axis=1))

        return tf.reduce_sum(tf.sqrt(score_r ** 2 + score_i ** 2), -1)

    def def_loss(self):
        """Defines the layers of the algorithm."""
        (pos_h_e_r, pos_h_e_i), (pos_r_e_r, pos_r_e_i), (pos_t_e_r, pos_t_e_i) \
            = self.embed(self.pos_h, self.pos_r, self.pos_t)

        (neg_h_e_r, neg_h_e_i), (neg_r_e_r, neg_r_e_i), (neg_t_e_r, neg_t_e_i) \
            = self.embed(self.neg_h, self.neg_r, self.neg_t)

        pos_score = self.comp_mul_and_min(pos_h_e_r, pos_h_e_i, pos_r_e_r, pos_r_e_i, pos_t_e_r, pos_t_e_i)
        neg_score = self.comp_mul_and_min(neg_h_e_r, neg_h_e_i, neg_r_e_r, neg_r_e_i, neg_t_e_r, neg_t_e_i)

        self.loss = tf.reduce_sum(tf.maximum(pos_score + self.config.margin - neg_score, 0))

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        num_entity = self.data_stats.tot_entity

        (h_vec_r, h_vec_i), (r_vec_r, r_vec_i), (t_vec_r, t_vec_i) \
            = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        head_pos_score = self.comp_mul_and_min_4_test(self.ent_embeddings_real, self.ent_embeddings_imag,
                                                      r_vec_r, r_vec_i, t_vec_r, t_vec_i)

        tail_pos_score = self.comp_mul_and_min_4_test(h_vec_r, h_vec_i, r_vec_r, r_vec_i,
                                                      self.ent_embeddings_real, self.ent_embeddings_imag)

        _, head_rank = tf.nn.top_k(head_pos_score, k=num_entity)
        _, tail_rank = tf.nn.top_k(tail_pos_score, k=num_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns real and imaginary values of head, relation and tail embedding.
        """
        pi = 3.14159265358979323846
        h_e_r = tf.nn.embedding_lookup(self.ent_embeddings_real, h)
        h_e_i = tf.nn.embedding_lookup(self.ent_embeddings_imag, h)
        r_e_r = tf.nn.embedding_lookup(self.rel_embeddings_real, r)
        t_e_r = tf.nn.embedding_lookup(self.ent_embeddings_real, t)
        t_e_i = tf.nn.embedding_lookup(self.ent_embeddings_imag, t)
        r_e_r = r_e_r / pi
        r_e_i = tf.sin(r_e_r)
        r_e_r = tf.cos(r_e_r)
        return (h_e_r, h_e_i), (r_e_r, r_e_i), (t_e_r, t_e_i)

    def get_embed(self, h, r, t, sess=None):
        """Function to get the embedding value in numpy.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns real and imaginary values of head, relation and tail embedding.
        """
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess):
        """Function to get the projected embedding value in numpy.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

        """
        return self.get_embed(h, r, t, sess)
