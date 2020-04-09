#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

class TuckER(ModelMeta):
    """ `TuckER-Tensor Factorization for Knowledge Graph Completion`_

        TuckER is a Tensor-factorization-based embedding technique based on
        the Tucker decomposition of a third-order binary tensor of triplets. Although
        being fully expressive, the number of parameters used in Tucker only grows linearly
        with respect to embedding dimension as the number of entities or relations in a
        knowledge graph increases.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.TuckER import TuckER
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TuckER()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _TuckER-Tensor Factorization for Knowledge Graph Completion:
            https://arxiv.org/pdf/1901.09590.pdf

    """

    def __init__(self, config=None):
        super(TuckER, self).__init__()
        self.config = config
        self.model_name = 'TuckER'
        self.training_strategy = TrainingStrategy.PROJECTION_BASED
        
        # raise NotImplementedError("TransG is yet finished in pykg2vec.")

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               d2 (Tensor): Size of the latent dimension for relations.
               d1 (Tensor): Size of the latent dimension for entities .
               parameter_list  (list): List of Tensor parameters.
               E (Tensor Variable): Lookup variable containing  embedding of the entities.
               R  (Tensor Variable): Lookup variable containing  embedding of the relations.
               W (Tensor Varible): Transformation matrix.
        """

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        self.d1 = self.config.ent_hidden_size
        self.d2 = self.config.rel_hidden_size
        
        emb_initializer = tf.initializers.glorot_normal()

        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, self.d1)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, self.d2)), name="rel_embedding")
        self.W = tf.Variable(emb_initializer(shape=(self.d2, self.d1, self.d1)), name="W")

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.W]
        self.inp_drop = tf.keras.layers.Dropout(rate=self.config.input_dropout)
        self.hidden_dropout1 = tf.keras.layers.Dropout(rate=self.config.hidden_dropout1)
        self.hidden_dropout2 = tf.keras.layers.Dropout(rate=self.config.hidden_dropout2)

    def forward(self, e1, r, direction=None):
        """Implementation of the layer.

            Args:
                e1(Tensor): entities id.
                r(Tensor): Relation id.

            Returns:
                Tensors: Returns the activation values.
        """
        e1 = tf.nn.embedding_lookup(self.ent_embeddings, e1)
        e1 = tf.nn.l2_normalize(e1, axis=1)
        e1 = self.inp_drop(e1)
        e1 = tf.reshape(e1, [-1, 1, self.d1])

        rel = tf.nn.embedding_lookup(self.rel_embeddings, r)
        W_mat = tf.matmul(rel, tf.reshape(self.W, [self.d2, -1]))
        W_mat = tf.reshape(W_mat, [-1, self.d1, self.d1])
        W_mat = self.hidden_dropout1(W_mat)

        x = tf.matmul(e1, W_mat)
        x = tf.reshape(x, [-1, self.d1])
        x = tf.nn.l2_normalize(x, axis=1)
        x = self.hidden_dropout2(x)
        x = tf.matmul(x, self.ent_embeddings, transpose_b=True)
        return tf.nn.sigmoid(x)

    def predict_tail_rank(self, e, r, topk=-1):
        _, rank = tf.nn.top_k(-self.forward(e, r), k=topk)
        return rank

    def predict_head_rank(self, e, r, topk=-1):
        _, rank = tf.nn.top_k(-self.forward(e, r), k=topk)
        return rank