#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

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
            >>> trainer = Trainer(model=model)
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
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

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
        emb_h = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=-1), h)
        emb_r = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.rel_matrices, axis=-1), r)
        emb_t = tf.nn.embedding_lookup(tf.nn.l2_normalize(self.ent_embeddings, axis=-1), t)
        emb_h = tf.reshape(emb_h, [-1, k, 1])
        emb_r = tf.reshape(emb_r, [-1, k, k])
        emb_t = tf.reshape(emb_t, [-1, k, 1])

        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        # dim of h: [m, k, 1]
        #        r: [m, k, k]
        #        t: [m, k, 1]
        return -tf.reduce_sum(h_e * tf.matmul(r_e, t_e), [1, 2])