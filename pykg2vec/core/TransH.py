#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class TransH(ModelMeta):
    """ `Knowledge Graph Embedding by Translating on Hyperplanes`_

        TransH models a relation as a hyperplane together with a translation operation on it.
        By doing this, it aims to preserve the mapping properties of relations such as reflexive,
        one-to-many, many-to-one, and many-to-many with almost the same model complexity of TransE.

         Args:
            config (object): Model configuration parameters.

         Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.

         Examples:
            >>> from pykg2vec.core.TransH import TransH
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransH()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

         Portion of the code based on OpenKE_ and thunlp_.
         .. _OpenKE:
             https://github.com/thunlp/OpenKE/blob/master/models/TransH.py

         .. _thunlp:
             https://github.com/thunlp/TensorFlow-TransX/blob/master/transH.py

         .. _Knowledge Graph Embedding by Translating on Hyperplanes:
             https://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf
    """

    def __init__(self, config):
        self.config = config
        self.model_name = 'TransH'

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
        self.test_t_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings (Tensor Variable): Lookup variable containing  embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing  embedding of the relations.
               w   (Tensor Variable): Weight matrix for transformation of entity embeddings.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
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
        """Defines the loss function for the algorithm."""
        emb_ph, emb_pr, emb_pt = self.embed(self.pos_h, self.pos_r, self.pos_t)
        emb_nh, emb_nr, emb_nt = self.embed(self.neg_h, self.neg_r, self.neg_t)

        score_pos = self.distance(emb_ph, emb_pr, emb_pt)
        score_neg = self.distance(emb_nh, emb_nr, emb_nt)

        self.loss = tf.reduce_sum(tf.maximum(0., score_pos + self.config.margin - score_neg)) + self.get_reg()

    def get_reg(self):
        """Performs regularization."""
        norm_ent_embedding = tf.nn.l2_normalize(self.ent_embeddings, axis=-1)
        norm_rel_embedding = tf.nn.l2_normalize(self.rel_embeddings, axis=-1)
        norm_w = tf.nn.l2_normalize(self.w, axis=-1)

        term1 = tf.reduce_sum(tf.maximum(tf.reduce_sum(norm_ent_embedding ** 2, -1) - 1, 0))
        term2 = tf.reduce_sum(tf.maximum(tf.div(tf.reduce_sum(norm_rel_embedding * norm_w, -1) ** 2,
                                                tf.reduce_sum(norm_rel_embedding ** 2, -1)) - 1e-07, 0))

        return self.config.C * (term1 + term2)

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

         Returns:
             Tensors: Returns ranks of head and tail.
        """
        num_entity = self.config.kg_meta.tot_entity

        head_vec, rel_vec, tail_vec = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)
        pos_norm = self.get_proj(self.test_r_batch)

        norm_ent_embedding = tf.nn.l2_normalize(self.ent_embeddings, 1)
        project_ent_embedding = self.projection(norm_ent_embedding, tf.expand_dims(pos_norm, axis=1))
        score_head = self.distance(project_ent_embedding,
                                   tf.expand_dims(rel_vec, axis=1),
                                   tf.expand_dims(tail_vec, axis=1), axis=2)
        score_tail = self.distance(tf.expand_dims(head_vec, axis=1),
                                   tf.expand_dims(rel_vec, axis=1),
                                   project_ent_embedding, axis=2)

        _, head_rank = tf.nn.top_k(score_head, k=num_entity)
        _, tail_rank = tf.nn.top_k(score_tail, k=num_entity)

        return head_rank, tail_rank

    def get_proj(self, r):
        """Calculates the projection of r"""
        return tf.nn.l2_normalize(tf.nn.embedding_lookup(self.w, r), axis=-1)

    def projection(self, entity, wr):
        """Calculates the projection of entities"""
        return entity - tf.reduce_sum(entity * wr, -1, keepdims=True) * wr

    def distance(self, h, r, t, axis=1):
        """Function to calculate distance measure in embedding space.

           Args:
              h (Tensor): Head entities ids.
              r (Tensor): Relation ids of the triple.
              t (Tensor): Tail entity ids of the triple.
              axis (int): Determines the axis for reduction

            Returns:
               Tensors: Returns the distance measure.
        """
        if self.config.L1_flag:
            return tf.reduce_sum(tf.abs(h + r - t), axis=axis)  # L1 norm
        else:
            return tf.reduce_sum((h + r - t) ** 2, axis=1)  # L2 norm

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)

        emb_h = tf.nn.l2_normalize(emb_h, axis=-1)
        emb_r = tf.nn.l2_normalize(emb_r, axis=-1)
        emb_t = tf.nn.l2_normalize(emb_t, axis=-1)

        proj_vec = self.get_proj(r)

        return self.projection(emb_h, proj_vec), emb_r, self.projection(emb_t, proj_vec)

    def get_embed(self, h, r, t, sess):
        """Function to get the embedding value in numpy.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess):
        """"Function to get the projected embedding value in numpy.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        return self.get_embed(h, r, t, sess)
