#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

class Complex(ModelMeta):
    """`Complex Embeddings for Simple Link Prediction`_.

    ComplEx is an enhanced version of DistMult in that it uses complex-valued embeddings
    to represent both entities and relations. Using the complex-valued embedding allows
    the defined scoring function in ComplEx to differentiate that facts with assymmetric relations.
    
    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
        model (str): Name of the model.
    
    Examples:
        >>> from pykg2vec.core.Complex import Complex
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = Complex()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf
    """

    def __init__(self, config):
        super(Complex, self).__init__()
        self.config = config
        self.model_name = 'Complex'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

    def def_parameters(self):
        """Defines the model parameters.
           
           Attributes:
               k (Tensor): Size of the latent dimesnion for entities and relations.
               emb_e_real (Tensor Variable): Lookup variable containing real values of the entities.
               emb_e_img (Tensor Variable): Lookup variable containing imaginary values of the entities.
               emb_rel_real (Tensor Variable): Lookup variable containing real values of the relations.
               emb_rel_img (Tensor Variable): Lookup variable containing imaginary values of the relations.
               parameter_list  (list): List of Tensor parameters. 
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size
        
        emb_initializer = tf.initializers.glorot_normal()

        self.ent_embeddings_real = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="emb_e_real")
        self.ent_embeddings_img  = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="emb_e_img") 
        self.rel_embeddings_real = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="emb_rel_real")
        self.rel_embeddings_img  = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="emb_rel_img")

        self.parameter_list = [self.ent_embeddings_real, self.ent_embeddings_img, self.rel_embeddings_real, self.rel_embeddings_img]

    def embed(self, h, r, t):
        """Function to get the embedding value.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns real and imaginary values of head, relation and tail embedding.
        """
        h_emb_real = tf.nn.embedding_lookup(self.ent_embeddings_real, h)
        h_emb_img  = tf.nn.embedding_lookup(self.ent_embeddings_img,  h)

        r_emb_real = tf.nn.embedding_lookup(self.rel_embeddings_real, r)
        r_emb_img  = tf.nn.embedding_lookup(self.rel_embeddings_img,  r)

        t_emb_real = tf.nn.embedding_lookup(self.ent_embeddings_real, t)
        t_emb_img  = tf.nn.embedding_lookup(self.ent_embeddings_img,  t)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img

    def dissimilarity(self, h_real, h_img, r_real, r_img, t_real, t_img):
        return tf.reduce_sum(h_real * t_real * r_real + h_img * t_img * r_real + h_real * t_img * r_img - h_img * t_real * r_img, axis=-1, keepdims = False)

    def get_loss(self, h, r, t, y):
        """Defines the loss function for the algorithm."""
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)

        score = self.dissimilarity(h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img)

        regul_term = tf.nn.l2_loss(h_e_real) + tf.nn.l2_loss(h_e_img) + tf.nn.l2_loss(r_e_real) + tf.nn.l2_loss(r_e_img) + tf.nn.l2_loss(t_e_real) + tf.nn.l2_loss(t_e_img)
        loss = tf.reduce_sum(tf.nn.softplus(-score*y)) + self.config.lmbda*regul_term

        return loss

    def predict(self, h, r, t, topk=-1):
        """Function that performs prediction for TransE. 
           shape of h can be either [num_tot_entity] or [1]. 
           shape of t can be either [num_tot_entity] or [1].

          Returns:
              Tensors: Returns ranks of head and tail.
        """
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)
        score = self.dissimilarity(h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img)
        _, rank = tf.nn.top_k(-score, k=topk)

        return rank