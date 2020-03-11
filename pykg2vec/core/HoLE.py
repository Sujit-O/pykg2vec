#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class HoLE(ModelMeta):
    """`Holographic Embeddings of Knowledge Graphs`_.

    HoLE employs the circular correlation to create composition correlations. It
    is able to represent and capture the interactions betweek entities and relations
    while being efficient to compute, easier to train and scalable to large dataset.

    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
    
    Examples:
        >>> from pykg2vec.core.HoLE import HoLE
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = HoLE()
        >>> trainer = Trainer(model=model, debug=False)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Holographic Embeddings of Knowledge Graphs:
        https://arxiv.org/pdf/1510.04935.pdf

    """

    def __init__(self, config=None):
        super(HoLE, self).__init__()
        self.config = config
        self.model_name = 'HoLE'

    def cir_corre(self, a, b):
        """Function performs circular correlation.

            Args:
                a (Tensor): Input Tensor.
                b (Tensor): Input Tensor.

            Returns:
                Tensor: Output Tensor after performing circular correlation.

        """
        a = tf.cast(a, tf.complex64)
        b = tf.cast(b, tf.complex64)
        return tf.math.real(tf.signal.ifft(tf.math.conj(tf.signal.fft(a)) * tf.signal.fft(b)))

    def dissimilarity(self, head, tail, rel, axis=1):
        """Function calculates the dissimilarity.

            Args:
                head (Tensor): Embedding of the head entity.
                tail (Tensor): Embedding of the tail entity.
                rel (Tensor): Embedding of the relations.
                axis (Int): Axis across which the sum reduced before activation.

            Returns:
                Tensor: Output after activation of the Tensor.

        """
        r = tf.nn.l2_normalize(rel, 1)
        e = self.cir_corre(head, tail)
        return -tf.sigmoid(tf.reduce_sum(r * e, axis=axis))

    def def_parameters(self):
        """Defines the model parameters.
           
           Attributes:
               num_total_ent (int): Total number of entities. 
               num_total_rel (int): Total number of relations. 
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing embedding of the relations.
               b  (Tensor Variable): Variable storing the bias values.
               parameter_list  (list): List of Tensor parameters.
        """ 
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        emb_initializer = tf.initializers.glorot_normal()

        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embedding")

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings]

    def get_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """Defines the loss function for the algorithm."""
        pos_h_e, pos_r_e, pos_t_e = self.embed(pos_h, pos_r, pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(neg_h, neg_r, neg_t)

        score_pos = self.dissimilarity(pos_h_e, pos_r_e, pos_t_e)
        score_neg = self.dissimilarity(neg_h_e, neg_r_e, neg_t_e)

        loss = tf.reduce_sum(tf.maximum(score_pos + self.config.margin - score_neg, 0))

        return loss

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
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        emb_h = tf.nn.embedding_lookup(norm_ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(norm_ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def predict(self, h, r, t, topk=-1):
        """Function that performs prediction for TransE. 
           shape of h can be either [num_tot_entity] or [1]. 
           shape of t can be either [num_tot_entity] or [1].

          Returns:
              Tensors: Returns ranks of head and tail.
        """
        h_e, r_e, t_e = self.embed(h, r, t)
        score = self.dissimilarity(h_e, r_e, t_e)
        _, rank = tf.nn.top_k(score, k=topk)

        return rank