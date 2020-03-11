#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class DistMult(ModelMeta):
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
        super(DistMult, self).__init__()
        self.config = config
        self.model_name = 'DistMult'

    def def_parameters(self):
        """Defines the model parameters.
           
           Attributes:
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings(Tensor Variable): Lookup variable containing embedding of entities.
               rel_embeddings (Tensor Variable): Lookup variable containing embedding of relations.
               parameter_list  (list): List of Tensor parameters. 
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size
        
        emb_initializer = tf.initializers.glorot_normal()

        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embedding")
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings]

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
        return tf.reduce_sum(h*r*t, axis=axis, keepdims=False)

    
    def get_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """Defines the loss function for the algorithm."""
        pos_h_e, pos_r_e, pos_t_e = self.embed(pos_h, pos_r, pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(neg_h, neg_r, neg_t)

        score_pos = self.dissimilarity(pos_h_e, pos_r_e, pos_t_e)
        score_neg = self.dissimilarity(neg_h_e, neg_r_e, neg_t_e)

        regul_term = tf.reduce_mean(pos_r_e**2) + tf.reduce_mean(neg_r_e**2)

        loss = tf.reduce_sum(tf.maximum(score_neg - score_pos + 1, 0)) + self.config.lmbda*regul_term

        return loss

    def predict(self, h, r, t, topk=-1):
        """Function that performs prediction for TransE. 
           shape of h can be either [num_tot_entity] or [1]. 
           shape of t can be either [num_tot_entity] or [1].

          Returns:
              Tensors: Returns ranks of head and tail.
        """
        h_e, r_e, t_e = self.embed(h, r, t)
        score = self.dissimilarity(h_e, r_e, t_e)
        _, rank = tf.nn.top_k(-score, k=topk)

        return rank