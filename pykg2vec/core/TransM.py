#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta

import numpy as np


class TransM(ModelMeta):
    """ `Transition-based Knowledge Graph Embedding with Relational Mapping Properties`_

        TransM is another line of research that improves TransE by relaxing the overstrict requirement of
        h+r ==> t. TransM associates each fact (h, r, t) with a weight theta(r) specific to the relation.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.core.TransM import TransM
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransM()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _Transition-based Knowledge Graph Embedding with Relational Mapping Properties:
            https://pdfs.semanticscholar.org/0ddd/f37145689e5f2899f8081d9971882e6ff1e9.pdf
    """

    def __init__(self, config=None):
        super(TransM, self).__init__()
        self.config = config
        self.model_name = 'TransM'

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               k (Tensor): Size of the latent dimesnion for entities and relations.

               ent_embeddings (Tensor Variable): Lookup variable containing  embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing  embedding of the relations.

               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        emb_initializer = tf.initializers.glorot_normal()
        
        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embedding")

        rel_head = {x: [] for x in range(num_total_rel)}
        rel_tail = {x: [] for x in range(num_total_rel)}
        rel_counts = {x: 0 for x in range(num_total_rel)}
        train_triples_ids = self.config.knowledge_graph.read_cache_data('triplets_train')
        for t in train_triples_ids:
            rel_head[t.r].append(t.h)
            rel_tail[t.r].append(t.t)
            rel_counts[t.r] += 1

        theta = [1/np.log(2+rel_counts[x]/(1+len(rel_tail[x])) + rel_counts[x]/(1+len(rel_head[x]))) for x in range(num_total_rel)]
        self.theta = tf.Variable(np.asarray(theta, dtype=np.float32), trainable=False)
        
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.theta]

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

        return emb_h, emb_r, emb_t

    def dissimilarity(self, h, r, t, axis=-1):
        """Function to calculate distance measure in embedding space.

        Args:
            h (Tensor): shape [b, k] Head entities in a batch. 
            r (Tensor): shape [b, k] Relation entities in a batch.
            t (Tensor): shape [b, k] Tail entities in a batch.
            axis (int): Determines the axis for reduction

        Returns:
            Tensor: shape [b] the aggregated distance measure.
        """
        norm_h = tf.nn.l2_normalize(h, axis=axis)
        norm_r = tf.nn.l2_normalize(r, axis=axis)
        norm_t = tf.nn.l2_normalize(t, axis=axis)
        
        dissimilarity = norm_h + norm_r - norm_t 

        if self.config.L1_flag:
            dissimilarity = tf.math.abs(dissimilarity) # L1 norm 
        else:
            dissimilarity = tf.math.square(dissimilarity) # L2 norm
        
        return tf.reduce_sum(dissimilarity, axis=axis)

    def get_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """Defines the loss function for the algorithm."""
        pos_h_e, pos_r_e, pos_t_e = self.embed(pos_h, pos_r, pos_t)
        neg_h_e, neg_r_e, neg_t_e = self.embed(neg_h, neg_r, neg_t)

        pos_r_theta = tf.nn.embedding_lookup(self.theta, pos_r)
        neg_r_theta = tf.nn.embedding_lookup(self.theta, neg_r)

        pos_score = pos_r_theta*self.dissimilarity(pos_h_e, pos_r_e, pos_t_e)
        neg_score = neg_r_theta*self.dissimilarity(neg_h_e, neg_r_e, neg_t_e)

        loss = self.pairwise_margin_loss(pos_score, neg_score)

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
        _, rank = tf.nn.top_k(score, k=topk)

        return rank