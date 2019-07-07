#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


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
        >>> trainer = Trainer(model=model, debug=False)
        >>> trainer.build_model()
        >>> trainer.train_model()
    
    Portion of the code based on Niubohan_ and BookmanHan_.
    .. _daiquocnguyen:
        https://github.com/daiquocnguyen/ConvKB

    .. _A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network:
        https://www.aclweb.org/anthology/N18-2053

    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta
        self.model_name = 'ConvKB'
        self.dense_last_dim = {50: 2592, 100: 5184, 200: 10368}
        if self.config.hidden_size not in self.dense_last_dim:
            raise NotImplementedError("The hidden dimension is not supported!")
        self.last_dim = self.dense_last_dim[self.config.hidden_size]

    def def_inputs(self):
        """Defines the inputs to the model.
           
           Attributes:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               hr_t (Tensor): Tail tensor list for (h,r) pair.
               rt_h (Tensor): Head tensor list for (r,t) pair.
               test_h_batch (Tensor): Batch of head ids for testing.
               test_r_batch (Tensor): Batch of relation ids for testing
               test_t_batch (Tensor): Batch of tail ids for testing.
        """
        self.h = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.hr_t = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])
        self.rt_h = tf.placeholder(tf.float32, [None, self.data_stats.tot_entity])

        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])

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
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.rel_embeddings = tf.get_variable(name="rel_embedding", shape=[num_total_rel, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        with tf.name_scope("activation_bias"):
            self.b = tf.get_variable(name="bias", shape=[1, num_total_ent],
                                     initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.b]

    def def_layer(self):
        k = self.config.hidden_size
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                if self.config.useConstantInit == False:
                    filter_shape = [sequence_length, filter_size, 1, self.config.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, seed=1234), name="W")
                else:
                    init1 = tf.constant([[[[0.1]]], [[[0.1]]], [[[-0.1]]]])
                    weight_init = tf.tile(init1, [1, filter_size, 1, self.config.num_filters])
                    W = tf.get_variable(name="W3", initializer=weight_init)

                b = tf.Variable(tf.constant(0.0, shape=[self.config.num_filters]), name="b")

        # Add dropout
        with tf.name_scope("dropout"):        
            self.hidden_drop = tf.keras.layers.Dropout(rate=self.config.hidden_dropout)        
        
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[k, self.config.num_classes],
                initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            b = tf.Variable(tf.constant(0.0, shape=[self.config.num_classes]), name="b")

    def forward(self, st_inp):
        k = self.config.hidden_size
        """Create a convolution + maxpool layer for each filter size."""
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                conv = tf.nn.conv2d(
                    st_inp,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled_outputs.append(h)

        # Combine all the pooled features
        self.h_pool = tf.concat(pooled_outputs, 2)
        total_dims = (k * len(self.config.filter_sizes) - sum(self.config.filter_sizes) + len(self.config.filter_sizes)) * self.config.num_filters
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_dims])
        
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = self.hidden_drop(self.h_pool_flat) 

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.nn.sigmoid(self.scores)

        return self.scores   
        

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        ent_emb_norm = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        rel_emb_norm = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        h_emb = tf.nn.embedding_lookup(ent_emb_norm, self.h)
        r_emb = tf.nn.embedding_lookup(rel_emb_norm, self.r)
        t_emb = tf.nn.embedding_lookup(ent_emb_norm, self.t)

        stacked_h = tf.reshape(h_emb, [-1, 10, 20, 1])
        stacked_r = tf.reshape(r_emb, [-1, 10, 20, 1])
        stacked_t = tf.reshape(t_emb, [-1, 10, 20, 1])

        stacked_hrt = tf.concat([stacked_h, stacked_r], 1)

        # TODO make two different forward layers for head and tail
        scores = self.forward(stacked_hrt)

        with tf.name_scope("loss"):
            losses = tf.nn.softplus(scores * self.hr_t)
            self.loss = tf.reduce_mean(losses) + self.config.lmbda * self.l2_loss

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        ent_emb_norm = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        rel_emb_norm = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        h_emb = tf.nn.embedding_lookup(ent_emb_norm, self.test_h_batch)
        r_emb = tf.nn.embedding_lookup(rel_emb_norm, self.test_r_batch)
        t_emb = tf.nn.embedding_lookup(ent_emb_norm, self.test_t_batch)

        
        stacked_h = tf.reshape(h_emb, [-1, 10, 20, 1])
        stacked_r = tf.reshape(r_emb, [-1, 10, 20, 1])
        stacked_t = tf.reshape(t_emb, [-1, 10, 20, 1])

        stacked_hrt = tf.concat([stacked_h, stacked_r], 1)
        stacked_tr = tf.concat([stacked_t, stacked_r], 1)

        # TODO make two different forward layers for head and tail
        pred_tails = self.forward(stacked_hr)
        pred_heads = self.forward(stacked_tr)

        _, head_rank = tf.nn.top_k(-pred_heads, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(-pred_tails, k=self.data_stats.tot_entity)

        return head_rank, tail_rank

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

    def get_embed(self, h, r, t, sess=None):
        """Function to get the embedding value in numpy.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h, emb_r, emb_t = self.embed(h, r, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

    def get_proj_embed(self, h, r, t, sess):
        """Function to get the projected embedding value in numpy.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        return self.get_embed(h, r, t, sess)

