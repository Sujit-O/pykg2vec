#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class Rescal(ModelMeta):
    """`A Three-Way Model for Collective Learning on Multi-Relational Data`_

        RESCAL is a tensor factorization approach to knowledge representation learning,
        which is able to perform collective learning via the latent components of the factorization.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.core.Rescal import Rescal
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = Rescal()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on mnick_ and `OpenKE_Rescal`_.

         .. _mnick: https://github.com/mnick/rescal.py/blob/master/rescal/rescal.py

         .. _OpenKE_Rescal: https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py

         .. _A Three-Way Model for Collective Learning on Multi-Relational Data : http://www.icml-2011.org/papers/438_icmlpaper.pdf
    """

    def __init__(self, config):
        super(Rescal, self).__init__()
        self.config = config
        self.model_name = 'Rescal'

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
               rel_matrices  (Tensor Variable): Transformation matrices for entities into relation space.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        emb_initializer = tf.initializers.glorot_normal()

        # A: per each entity, store its embedding representation.
        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embedding")
        # M: per each relation, store a matrix that models the interactions between entity embeddings.
        self.rel_matrices = tf.Variable(emb_initializer(shape=(num_total_rel, k * k)), name="rel_matrices")

        self.parameter_list = [self.ent_embeddings, self.rel_matrices]

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        k = self.config.hidden_size
        emb_h = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), h)
        emb_r = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.rel_matrices, axis=1), r)
        emb_t = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=1), t)
        emb_h = tf.reshape(emb_h, [-1, k, 1])
        emb_r = tf.reshape(emb_r, [-1, k, k])
        emb_t = tf.reshape(emb_t, [-1, k, 1])

        return emb_h, emb_r, emb_t

    def match(self, h, r, t):
        """Function to calculate truth value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns Tensors.
        """
        # dim of h: [m, k, 1]
        #        r: [m, k, k]
        #        t: [m, k, 1]
        return tf.reduce_sum(h * tf.matmul(r, t), [1, 2])

    def get_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """Defines the loss function for the algorithm."""
        pos_h_e, pos_r_e, pos_t_e = self.embed(pos_h, pos_r, pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(neg_h, neg_r, neg_t)
        pos_score = self.match(pos_h_e, pos_r_e, pos_t_e)
        neg_score = self.match(neg_h_e, neg_r_e, neg_t_e)

        loss = tf.reduce_sum(tf.maximum(neg_score + self.config.margin - pos_score, 0))
        return loss

    def predict(self, h, r, t, topk=-1):
        """Function that performs prediction for TransE. 
           shape of h can be either [num_tot_entity] or [1]. 
           shape of t can be either [num_tot_entity] or [1].

          Returns:
              Tensors: Returns ranks of head and tail.
        """
        h_e, r_e, t_e = self.embed(h, r, t)
        score = -self.match(h_e, r_e, t_e)
        _, rank = tf.nn.top_k(score, k=topk)

        return rank

    def test_batch(self, h_batch, r_batch, t_batch):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        k = self.config.hidden_size

        h_vec, r_vec, t_vec = self.embed(h_batch, r_batch, t_batch)

        h_sim = tf.tensordot(tf.squeeze(tf.matmul(r_vec, t_vec), axis=-1), self.ent_embeddings, axes=((-1), (-1)))
        t_sim = tf.squeeze(tf.tensordot(tf.matmul(tf.reshape(h_vec, [-1, 1, k]), r_vec),
                                        self.ent_embeddings, axes=((-1), (-1))), axis=1)

        _, head_rank = tf.nn.top_k(tf.negative(h_sim), k=self.config.kg_meta.tot_entity)
        _, tail_rank = tf.nn.top_k(tf.negative(t_sim), k=self.config.kg_meta.tot_entity)

        return head_rank, tail_rank