#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

class ConvE(ModelMeta):
    """`Convolutional 2D Knowledge Graph Embeddings`_

    ConvE is a multi-layer convolutional network model for link prediction,
    it is a embedding model which is highly parameter efficient.

    Args:
        config (object): Model configuration parameters.
    
    Attributes:
        config (object): Model configuration.
        data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
        model (str): Name of the model.
        last_dim (int): The size of the last dimesion, depends on hidden size.

    
    Examples:
        >>> from pykg2vec.core.Complex import ConvE
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = ConvE()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Convolutional 2D Knowledge Graph Embeddings:
        https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17366/15884
    """

    def __init__(self, config):
        super(ConvE, self).__init__()
        self.config = config
        self.model_name = 'ConvE'
        self.training_strategy = TrainingStrategy.PROJECTION_BASED

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
        self.b = tf.Variable(emb_initializer(shape=(1, num_total_ent)), name="bias")
        self.bn0 = tf.keras.layers.BatchNormalization(axis=-1)
        self.inp_drop = tf.keras.layers.Dropout(rate=self.config.input_dropout)
        self.conv2d_1 = tf.keras.layers.Conv2D(self.config.num_filters, [3, 3], strides=(1, 1), padding='valid', use_bias=True)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.feat_drop = tf.keras.layers.SpatialDropout2D(self.config.feature_map_dropout)
        self.fc1 = tf.keras.layers.Dense(units=self.config.hidden_size)
        self.hidden_drop = tf.keras.layers.Dropout(rate=self.config.hidden_dropout)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)

        self.b2 = tf.Variable(emb_initializer(shape=(1, num_total_ent)), name="bias2")
        self.bn02 = tf.keras.layers.BatchNormalization(axis=-1)
        self.inp_drop2 = tf.keras.layers.Dropout(rate=self.config.input_dropout)
        self.conv2d_12 = tf.keras.layers.Conv2D(self.config.num_filters, [3, 3], strides=(1, 1), padding='valid', use_bias=True)
        self.bn12 = tf.keras.layers.BatchNormalization(axis=-1)
        self.feat_drop2 = tf.keras.layers.SpatialDropout2D(self.config.feature_map_dropout)
        self.fc12 = tf.keras.layers.Dense(units=self.config.hidden_size)
        self.hidden_drop2 = tf.keras.layers.Dropout(rate=self.config.hidden_dropout)
        self.bn22 = tf.keras.layers.BatchNormalization(axis=-1)
        self.parameter_list2 = [self.ent_embeddings, self.rel_embeddings, self.b]

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

    def embed2(self, e, r):
        emb_e = tf.nn.embedding_lookup(self.ent_embeddings, e)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        return emb_e, emb_r

    def inner_forward1(self, st_inp, first_dimension_size):
        """Implements the forward pass layers of the algorithm."""
        # batch normalization in the first axis
        x = self.bn0(st_inp)
        # input dropout
        x = self.inp_drop(x)
        # 2d convolution layer, output channel =32, kernel size = 3,3
        x = self.conv2d_1(x)
        # batch normalization across feature dimension
        x = self.bn1(x)
        # first non-linear activation
        x = tf.nn.relu(x)
        # feature dropout
        x = self.feat_drop(x)
        # reshape the tensor to get the batch size
        '''10368 with k=200,5184 with k=100, 2592 with k=50'''
        x = tf.reshape(x, [first_dimension_size, -1])
        # pass the feature through fully connected layer, output size = batch size, hidden size
        x = self.fc1(x)
        # dropout in the hidden layer
        x = self.hidden_drop(x)
        # batch normalization across feature dimension
        x = self.bn2(x)
        # second non-linear activation
        x = tf.nn.relu(x)
        # project and get inner product with the tail triple
        x = tf.matmul(x, self.ent_embeddings, transpose_b=True) # [b, k] * [k, tot_ent]
        # add a bias value
        x = tf.add(x, self.b)
        # sigmoid activation
        return tf.nn.sigmoid(x)

    def inner_forward2(self, st_inp, first_dimension_size):
        """Implements the forward pass layers of the algorithm."""
        # batch normalization in the first axis
        x = self.bn02(st_inp)
        # input dropout
        x = self.inp_drop2(x)
        # 2d convolution layer, output channel =32, kernel size = 3,3
        x = self.conv2d_12(x)
        # batch normalization across feature dimension
        x = self.bn12(x)
        # first non-linear activation
        x = tf.nn.relu(x)
        # feature dropout
        x = self.feat_drop2(x)
        # reshape the tensor to get the batch size
        '''10368 with k=200,5184 with k=100, 2592 with k=50'''
        x = tf.reshape(x, [first_dimension_size, -1])
        # pass the feature through fully connected layer, output size = batch size, hidden size
        x = self.fc12(x)
        # dropout in the hidden layer
        x = self.hidden_drop2(x)
        # batch normalization across feature dimension
        x = self.bn22(x)
        # second non-linear activation
        x = tf.nn.relu(x)
        # project and get inner product with the tail triple
        x = tf.matmul(x, self.ent_embeddings, transpose_b=True) # [b, k] * [k, tot_ent]
        # add a bias value
        x = tf.add(x, self.b2)
        # sigmoid activation
        return tf.nn.sigmoid(x)

    def forward(self, e, r, direction="tail"):
        e_emb, r_emb, = self.embed2(e, r)
        first_dimension_size = e.get_shape().as_list()[0]

        stacked_e  = tf.reshape(e_emb, [-1, 10, 20, 1])
        stacked_r  = tf.reshape(r_emb, [-1, 10, 20, 1])
        stacked_er = tf.concat([stacked_e, stacked_r], 1)

        if direction == "tail":
            preds = self.inner_forward1(stacked_er, first_dimension_size)
        else:
            preds = self.inner_forward2(stacked_er, first_dimension_size)
        
        return preds

    def predict_tail_rank(self, e, r, topk=-1):
        _, rank = tf.nn.top_k(-self.forward(e, r, direction="tail"), k=topk)
        return rank

    def predict_head_rank(self, e, r, topk=-1):
        _, rank = tf.nn.top_k(-self.forward(e, r, direction="head"), k=topk)
        return rank