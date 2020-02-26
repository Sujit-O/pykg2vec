#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta, InferenceMeta


class DistMult(ModelMeta, InferenceMeta):
    """`EMBEDDING ENTITIES AND RELATIONS FOR LEARNING AND INFERENCE IN KNOWLEDGE BASES`_

        DistMult is a simpler model comparing with RESCAL in that it simplifies
        the weight matrix used in RESCAL to a diagonal matrix. The scoring
        function used DistMult can capture the pairwise interactions between
        the head and the tail entities. However, DistMult has limitation on modeling
        asymmetric relations.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            tot_ent (int): Total unique entites in the knowledge graph.
            tot_rel (int): Total unique relation in the knowledge graph.
            model (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.Complex import DistMult
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = DistMult()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _EMBEDDING ENTITIES AND RELATIONS FOR LEARNING AND INFERENCE IN KNOWLEDGE BASES:
            https://arxiv.org/pdf/1412.6575.pdf

    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.tot_ent = self.data_stats.tot_entity
        self.tot_rel = self.data_stats.tot_relation
        self.model_name = 'DistMult'

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

    def dissimilarity(self, h, r, t, axis=-1):
        """Function to calculate dissimilarity measure in embedding space.

        Args:
            h (Tensor): Head entities ids.
            r (Tensor): Relation ids of the triple.
            t (Tensor): Tail entity ids of the triple.
            axis (int): Determines the axis for reduction

        Returns:
            Tensors: Returns the dissimilarity measure.
        """
        return tf.reduce_sum(h*r*t, axis=axis, keep_dims=False)

    def def_parameters(self):
        """Defines the model parameters.
           
           Attributes:
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings(Tensor Variable): Lookup variable containing embedding of entities.
               rel_embeddings (Tensor Variable): Lookup variable containing embedding of relations.
               parameter_list  (list): List of Tensor parameters. 
        """
        k = self.config.hidden_size
        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[self.tot_ent, k],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[self.tot_rel, k],
                                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings]

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        score_pos = self.dissimilarity(pos_h_e, pos_r_e, pos_t_e)
        score_neg = self.dissimilarity(neg_h_e, neg_r_e, neg_t_e)

        regul_term = tf.reduce_sum(pos_r_e**2) + tf.reduce_mean(neg_r_e**2)

        self.loss = tf.reduce_sum(tf.maximum(score_neg - score_pos + 1, 0)) + self.config.lmbda*regul_term

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        h_emb, r_emb, t_emb = self.embed(self.test_h_batch, self.test_r_batch, self.test_t_batch)

        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)

        score_head = self.dissimilarity(norm_ent_embeddings, 
                                        tf.expand_dims(r_emb, axis=1), 
                                        tf.expand_dims(t_emb, axis=1))
        score_tail = self.dissimilarity(tf.expand_dims(h_emb, axis=1), 
                                        tf.expand_dims(r_emb, axis=1), 
                                        norm_ent_embeddings)

        _, head_rank = tf.nn.top_k(tf.negative(score_head), k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(tf.negative(score_tail), k=self.data_stats.tot_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
        """Function to get the embedding value.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        # norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        h_emb = tf.nn.embedding_lookup(norm_ent_embeddings, h)
        r_emb = tf.nn.embedding_lookup(self.rel_embeddings, r)
        t_emb = tf.nn.embedding_lookup(norm_ent_embeddings, t)

        return h_emb, r_emb, t_emb

    def get_embed(self, h, r, t, sess=None):
        """Function to get the embedding value in numpy.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        h, r, t = self.embed(h, r, t)
        return sess.run([h, r, t])

    def get_proj_embed(self, h, r, t, sess):
        """Function to get the projected embedding value in numpy.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        return self.get_embed(h, r, t, sess)

