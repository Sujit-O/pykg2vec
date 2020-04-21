#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

class TransR(ModelMeta):
    """ `Learning Entity and Relation Embeddings for Knowledge Graph Completion`_

        TranR is a translation based knowledge graph embedding method. Similar to TransE and TransH, it also
        builds entity and relation embeddings by regarding a relation as translation from head entity to tail
        entity. However, compared to them, it builds the entity and relation embeddings in a separate entity
        and relation spaces.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.core.TransR import TransR
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransR()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on `thunlp_transR`_.

         .. _thunlp_transR:
             https://github.com/thunlp/TensorFlow-TransX/blob/master/transR.py

        .. _Learning Entity and Relation Embeddings for Knowledge Graph Completion:
            http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf
    """

    def __init__(self, config):
        super(TransR, self).__init__()
        self.config = config
        self.model_name = 'TransR'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    def def_parameters(self):
        """Defines the model parameters.

          Attributes:
              num_total_ent (int): Total number of entities.
              num_total_rel (int): Total number of relations.
              k (Tensor): Size of the latent dimesnion for entities and relations.
              ent_embeddings (Tensor Variable): Lookup variable containing  embedding of the entities.
              rel_embeddings  (Tensor Variable): Lookup variable containing  embedding of the relations.
              rel_matrix   (Tensor Variable): Weight matrix for transformation of entity embeddings.
              parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.ent_hidden_size
        d = self.config.rel_hidden_size

        emb_initializer = tf.initializers.glorot_normal()

        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, d)), name="rel_embedding")
        self.rel_matrix     = tf.Variable(emb_initializer(shape=(num_total_rel, k, d)), name="rel_matrix")
        
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.rel_matrix]

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        h_e = tf.nn.embedding_lookup(self.ent_embeddings, h)
        r_e = tf.nn.embedding_lookup(self.rel_embeddings, r)
        t_e = tf.nn.embedding_lookup(self.ent_embeddings, t)
        
        h_e = tf.nn.l2_normalize(h_e, axis=-1)
        r_e = tf.nn.l2_normalize(r_e, axis=-1)
        t_e = tf.nn.l2_normalize(t_e, axis=-1)

        h_e = tf.expand_dims(h_e, axis=1)
        t_e = tf.expand_dims(t_e, axis=1)
        # [b, 1, k]

        matrix = tf.nn.embedding_lookup(self.rel_matrix, r)
        # [b, k, d]

        transform_h_e = tf.matmul(h_e, matrix)
        transform_t_e = tf.matmul(t_e, matrix)
        # [b, 1, d] = [b, 1, k] * [b, k, d]

        h_e = tf.squeeze(transform_h_e, axis=1)
        t_e = tf.squeeze(transform_t_e, axis=1)
        # [b, d]
        return h_e, r_e, t_e

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

        if self.config.L1_flag:
            return tf.reduce_sum(tf.math.abs(norm_h_e + norm_r_e - norm_t_e), -1) # L1 norm 
        else:
            return tf.reduce_sum(tf.math.square(norm_h_e + norm_r_e - norm_t_e), -1) # L2 norm