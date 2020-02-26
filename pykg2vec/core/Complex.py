#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta, InferenceMeta


class Complex(ModelMeta, InferenceMeta):
    """`Complex Embeddings for Simple Link Prediction`_.

    ComplEx is an enhanced version of DistMult in that it uses complex-valued embeddings
    to represent both entities and relations. Using the complex-valued embedding allows
    the defined scoring function in ComplEx to differentiate that facts with assymmetric relations.
    
    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
        tot_ent (int): Total unique entites in the knowledge graph.
        tot_rel (int): Total unique relation in the knowledge graph.
        model (str): Name of the model.
    
    Examples:
        >>> from pykg2vec.core.Complex import Complex
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = Complex()
        >>> trainer = Trainer(model=model, debug=False)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf
    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.tot_ent = self.data_stats.tot_entity
        self.tot_rel = self.data_stats.tot_relation
        self.model_name = 'Complex'

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
        self.pos_r = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
        """Defines the model parameters.
           
           Attributes:
               k (Tensor): Size of the latent dimesnion for entities and relations.
               emb_e_real (Tensor Variable): Lookup variable containing real values of the entities.
               emb_e_img (Tensor Variable): Lookup variable containing imaginary values of the entities.
               emb_rel_real (Tensor Variable): Lookup variable containing real values of the relations.
               emb_rel_img (Tensor Variable): Lookup variable containing imaginary values of the relations.
               parameter_list  (list): List of Tensor parameters. 
        """

        k = self.config.hidden_size
        with tf.name_scope("embedding"):
            self.ent_embeddings_real = tf.get_variable(name="emb_e_real", shape=[self.tot_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.ent_embeddings_img  = tf.get_variable(name="emb_e_img", shape=[self.tot_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings_real = tf.get_variable(name="emb_rel_real", shape=[self.tot_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings_img  = tf.get_variable(name="emb_rel_img", shape=[self.tot_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.ent_embeddings_real, self.ent_embeddings_img, 
                                   self.rel_embeddings_real, self.rel_embeddings_img]

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        pos_h_e_real, pos_h_e_img, pos_r_e_real, pos_r_e_img, pos_t_e_real, pos_t_e_img = \
            self.embed(self.pos_h, self.pos_r, self.pos_t)

        neg_h_e_real, neg_h_e_img, neg_r_e_real, neg_r_e_img, neg_t_e_real, neg_t_e_img = \
            self.embed(self.neg_h, self.neg_r, self.neg_t)

        score_pos = self.distance(pos_h_e_real, pos_h_e_img, pos_r_e_real, pos_r_e_img, pos_t_e_real, pos_t_e_img)
        score_neg = self.distance(neg_h_e_real, neg_h_e_img, neg_r_e_real, neg_r_e_img, neg_t_e_real, neg_t_e_img)

        regul_term = tf.reduce_sum(pos_h_e_real**2 + pos_h_e_img**2 + pos_r_e_real**2 + pos_r_e_img**2 + pos_t_e_real**2 + pos_t_e_img**2) + \
                     tf.reduce_sum(neg_h_e_real**2 + neg_h_e_img**2 + neg_r_e_real**2 + neg_r_e_img**2 + neg_t_e_real**2 + neg_t_e_img**2) 
        loss_term  = tf.reduce_sum(tf.concat([tf.nn.softplus(-1*score_pos), tf.nn.softplus(score_neg)], axis=0))        
        
        self.loss = loss_term + self.config.lmbda*regul_term

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img = \
            self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        score_head = self.distance(self.ent_embeddings_real, self.ent_embeddings_img, 
                                   tf.expand_dims(r_emb_real, axis=1), tf.expand_dims(r_emb_img, axis=1),
                                   tf.expand_dims(t_emb_real, axis=1), tf.expand_dims(t_emb_img, axis=1))
        score_tail = self.distance(tf.expand_dims(h_emb_real, axis=1), tf.expand_dims(h_emb_img, axis=1),
                                   tf.expand_dims(r_emb_real, axis=1), tf.expand_dims(r_emb_img, axis=1),
                                   self.ent_embeddings_real, self.ent_embeddings_img)

        _, head_rank = tf.nn.top_k(-score_head, k=self.config.kg_meta.tot_entity)
        _, tail_rank = tf.nn.top_k(-score_tail, k=self.config.kg_meta.tot_entity)

        return head_rank, tail_rank

    def distance(self, h_real, h_img, r_real, r_img, t_real, t_img):
        return tf.reduce_sum(h_real * t_real * r_real + h_img * t_img * r_real + h_real * t_img * r_img - h_img * t_real * r_img, axis=-1, keep_dims = False)

    # Override
    def dissimilarity(self, h, r, t):
        """Function to calculate dissimilarity measure in embedding space.

        Args:
            h (Tensor): Head entities ids.
            r (Tensor): Relation ids of the triple.
            t (Tensor): Tail entity ids of the triple.

        Returns:
            Tensors: Returns the dissimilarity measure.
        """
        if self.config.L1_flag:
            return tf.reduce_sum(tf.abs(h + r - t), axis=1)  # L1 norm
        else:
            return tf.reduce_sum((h + r - t) ** 2, axis=1)  # L2 norm

    def embed(self, h, r, t):
        """Function to get the embedding value.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns real and imaginary values of head, relation and tail embedding.
        """
        h_emb_real = tf.nn.embedding_lookup(self.ent_embeddings_real, h)
        h_emb_img  = tf.nn.embedding_lookup(self.ent_embeddings_img,  h)

        r_emb_real = tf.nn.embedding_lookup(self.rel_embeddings_real, r)
        r_emb_img  = tf.nn.embedding_lookup(self.rel_embeddings_img,  r)

        t_emb_real = tf.nn.embedding_lookup(self.ent_embeddings_real, t)
        t_emb_img  = tf.nn.embedding_lookup(self.ent_embeddings_img,  t)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img

    def get_embed(self, h, r, t, sess=None):
        """Function to get the embedding value in numpy.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns real and imaginary values of head, relation and tail embedding.
        """
        h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img = self.embed(h, r, t)
        h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img = sess.run(
            [h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img])
        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img

    def get_proj_embed(self, h, r, t, sess):
        """Function to get the projected embedding value in numpy.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

        """
        return self.get_embed(h, r, t, sess)
