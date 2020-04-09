#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

class ConvKB(ModelMeta):
    """`A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network`_

    ConvKB, each triple (head entity, relation, tail entity) is represented as a 3-
    column matrix where each column vector represents a triple element

    Args:
        config (object): Model configuration parameters.
    
    Attributes:
        config (object): Model configuration.
        data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
        model (str): Name of the model.
        last_dim (int): The size of the last dimesion, depends on hidden size.

    
    Examples:
        >>> from pykg2vec.core.ConvKB import ConvKB
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = ConvKB()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()
    
    Portion of the code based on Niubohan_ and BookmanHan_.
    .. _daiquocnguyen:
        https://github.com/daiquocnguyen/ConvKB

    .. _A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network:
        https://www.aclweb.org/anthology/N18-2053

    """

    def __init__(self, config):
        super(ConvKB, self).__init__()
        self.config = config
        self.model_name = 'ConvKB'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

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

        self.conv_list = [tf.keras.layers.Conv2D(self.config.num_filters, (3, filter_size), padding = 'valid', 
            use_bias=True, strides=(1,1), activation=tf.keras.layers.ReLU()) for filter_size in self.config.filter_sizes]
        
        self.fc1 = tf.keras.layers.Dense(1, use_bias=True,
            kernel_regularizer=tf.keras.regularizers.l2(l=self.config.lmbda),
            bias_regularizer=tf.keras.regularizers.l2(l=self.config.lmbda))
    
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

    def forward(self, h, r, t):
        h_emb, r_emb, t_emb = self.embed(h, r, t)
        first_dimen = h_emb.get_shape().as_list()[0]

        stacked_h = tf.expand_dims(h_emb, 1)
        stacked_r = tf.expand_dims(r_emb, 1)
        stacked_t = tf.expand_dims(t_emb, 1)

        stacked_hrt = tf.concat([stacked_h, stacked_r, stacked_t], 1)
        stacked_hrt = tf.expand_dims(stacked_hrt, -1) # [b, 3, k, 1]

        stacked_hrt = [self.conv_list[i](stacked_hrt) for i in range(len(self.config.filter_sizes))]
        stacked_hrt = tf.keras.layers.concatenate(stacked_hrt, axis=2)
        stacked_hrt = tf.reshape(stacked_hrt, [first_dimen, -1])
        preds = self.fc1(stacked_hrt)
        preds = tf.squeeze(preds, -1)
        return preds