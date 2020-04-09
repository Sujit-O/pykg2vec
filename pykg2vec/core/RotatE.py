#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

class RotatE(ModelMeta):
    """ `Rotate-Knowledge graph embedding by relation rotation in complex space`_

        RotatE models the entities and the relations in the complex vector space.
        The translational relation in RotatE is defined as the element-wise 2D
        rotation in which the head entity h will be rotated to the tail entity t by
        multiplying the unit-length relation r in complex number form.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.RotatE import RotatE
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = RotatE()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _Rotate-Knowledge graph embedding by relation rotation in complex space:
            https://openreview.net/pdf?id=HkgEQnRqYQ
    """

    def __init__(self, config):
        super(RotatE, self).__init__()
        self.config = config
        self.model_name = 'RotatE'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings_real (Tensor Variable): Lookup variable containing real values of the entities.
               ent_embeddings_imag (Tensor Variable): Lookup variable containing imaginary values of the entities.
               rel_embeddings_real (Tensor Variable): Lookup variable containing real values of the relations.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation

        k = self.config.hidden_size
        # emb_initializer = tf.initializers.glorot_normal()
        self.embedding_range = (self.config.margin + 2.0) / k 
        emb_initializer = tf.random_uniform_initializer(minval=-self.embedding_range, maxval=self.embedding_range)
        
        self.ent_embeddings       = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embeddings_real")
        self.ent_embeddings_imag  = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embeddings_imag") 
        self.rel_embeddings       = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embeddings_real")

        self.parameter_list = [self.ent_embeddings, self.ent_embeddings_imag, self.rel_embeddings]

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns real and imaginary values of head, relation and tail embedding.
        """
        pi = 3.14159265358979323846
        h_e_r = tf.nn.embedding_lookup(self.ent_embeddings, h)
        h_e_i = tf.nn.embedding_lookup(self.ent_embeddings_imag, h)
        r_e_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        t_e_r = tf.nn.embedding_lookup(self.ent_embeddings, t)
        t_e_i = tf.nn.embedding_lookup(self.ent_embeddings_imag, t)
        r_e_r = r_e_r / (self.embedding_range / pi)
        r_e_i = tf.sin(r_e_r)
        r_e_r = tf.cos(r_e_r)
        return h_e_r, h_e_i, r_e_r, r_e_i, t_e_r, t_e_i
   
    def forward(self, h, r, t):
        h_e_r, h_e_i, r_e_r, r_e_i, t_e_r, t_e_i = self.embed(h, r, t)
        score_r = h_e_r * r_e_r - h_e_i * r_e_i - t_e_r
        score_i = h_e_r * r_e_i + h_e_i * r_e_r - t_e_i
        return -(self.config.margin - tf.reduce_sum(score_r**2 + score_i**2, axis=-1))