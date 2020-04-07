#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy
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
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _Transition-based Knowledge Graph Embedding with Relational Mapping Properties:
            https://pdfs.semanticscholar.org/0ddd/f37145689e5f2899f8081d9971882e6ff1e9.pdf
    """

    def __init__(self, config):
        super(TransM, self).__init__()
        self.config = config
        self.model_name = 'TransM'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

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

    def forward(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids.
               t (Tensor): Tail entity ids.

            Returns:
                Tensors: the scores of evaluationReturns head, relation and tail embedding Tensors.
        """
        h_e, r_e, t_e = self.embed(h, r, t)

        norm_h_e = tf.nn.l2_normalize(h_e, -1)
        norm_r_e = tf.nn.l2_normalize(r_e, -1)
        norm_t_e = tf.nn.l2_normalize(t_e, -1)
        
        r_theta = tf.nn.embedding_lookup(self.theta, r)

        if self.config.L1_flag:
            return r_theta*tf.reduce_sum(tf.math.abs(norm_h_e + norm_r_e - norm_t_e), -1) # L1 norm 
        else:
            return r_theta*tf.reduce_sum(tf.math.square(norm_h_e + norm_r_e - norm_t_e), -1) # L2 norm

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