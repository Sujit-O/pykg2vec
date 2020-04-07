#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

class TransH(ModelMeta):
    """ `Knowledge Graph Embedding by Translating on Hyperplanes`_

        TransH models a relation as a hyperplane together with a translation operation on it.
        By doing this, it aims to preserve the mapping properties of relations such as reflexive,
        one-to-many, many-to-one, and many-to-many with almost the same model complexity of TransE.

         Args:
            config (object): Model configuration parameters.

         Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.

         Examples:
            >>> from pykg2vec.core.TransH import TransH
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransH()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

         Portion of the code based on `OpenKE_TransH`_ and `thunlp_TransH`_.

         .. _OpenKE_TransH:
             https://github.com/thunlp/OpenKE/blob/master/models/TransH.py

         .. _thunlp_TransH:
             https://github.com/thunlp/TensorFlow-TransX/blob/master/transH.py

         .. _Knowledge Graph Embedding by Translating on Hyperplanes:
             https://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf
    """

    def __init__(self, config):
        super(TransH, self).__init__()
        self.config = config
        self.model_name = 'TransH'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings (Tensor Variable): Lookup variable containing  embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing  embedding of the relations.
               w   (Tensor Variable): Weight matrix for transformation of entity embeddings.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        initializer = tf.initializers.glorot_normal()

        self.ent_embeddings = tf.Variable(initializer(shape=(num_total_ent, self.config.hidden_size)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(initializer(shape=(num_total_rel, self.config.hidden_size)), name="rel_embedding")
        self.w              = tf.Variable(initializer(shape=(num_total_rel, self.config.hidden_size)), name="w")
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.w]

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)

        norm_h_e = tf.nn.l2_normalize(h_e, -1)
        norm_r_e = tf.nn.l2_normalize(r_e, -1)
        norm_t_e = tf.nn.l2_normalize(t_e, -1)

        if self.config.L1_flag:
            return tf.reduce_sum(tf.math.abs(norm_h_e + norm_r_e - norm_t_e), -1) # L1 norm 
        else:
            return tf.reduce_sum(tf.math.square(norm_h_e + norm_r_e - norm_t_e), -1) # L2 norm
    
    def projection(self, emb_e, proj_vec):
        """Calculates the projection of entities"""
        proj_vec = tf.nn.l2_normalize(proj_vec, axis=-1)

        # [b, k], [b, k]
        return emb_e - tf.reduce_sum(emb_e * proj_vec, -1, keepdims=True) * proj_vec

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h =    tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r =    tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t =    tf.nn.embedding_lookup(self.ent_embeddings, t)
        proj_vec = tf.nn.embedding_lookup(self.w, r)

        emb_h = self.projection(emb_h, proj_vec)
        emb_t = self.projection(emb_t, proj_vec)

        return emb_h, emb_r, emb_t