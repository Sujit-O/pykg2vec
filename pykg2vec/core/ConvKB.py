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
        """Defines the layers of the algorithm."""
        self.conv_list = [tf.keras.layers.Conv2D(self.config.num_filters, 
            (self.config.sequence_length, filter_size), 
            padding = 'valid', 
            use_bias= True, 
            strides = (1,1),
            activation = tf.keras.layers.ReLU()) for filter_size in self.config.filter_sizes]
        self.drop = tf.keras.layers.Dropout(rate=self.config.hidden_dropout)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(self.config.hidden_size,
            activation=tf.keras.layers.ReLU(),
            use_bias=True,
            kernel_regularizer = tf.keras.regularizers.l2(l=0.01),
            bias_regularizer = tf.keras.regularizers.l2(l=0.01))


    def forward(self, x, batch):
        k = self.config.hidden_size
        #pass the data from all the convolution layers
        x = [self.conv_list[i](x) for i in range(len(self.config.filter_sizes))]
        #concatenate the result from all the layers
        x = tf.keras.layers.concatenate(x,axis=2)
        #get the total dimension
        total_dims = (k*len(self.config.filter_sizes)-sum(self.config.filter_sizes)+len(self.config.filter_sizes)) * self.config.num_filters
        #reshape the result
        #TODO: fixe the final dimension calculation equation
        x = tf.reshape(x, [-1,128250])
        #perform the dropout
        x = self.drop(x)
        #pass it through the fully connected layer
        x= self.fc1(x)
        #pass the feature through fully connected laye
        x = tf.matmul(x, tf.transpose(tf.nn.l2_normalize(self.ent_embeddings, axis=1)))
        #add a bias value
        x = tf.add(x, self.b)
        # sigmoid activation
        return tf.nn.sigmoid(x)
        

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        ent_emb_norm = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        rel_emb_norm = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        h_emb = tf.nn.embedding_lookup(ent_emb_norm, self.h)
        r_emb = tf.nn.embedding_lookup(rel_emb_norm, self.r)
        t_emb = tf.nn.embedding_lookup(ent_emb_norm, self.t)

        hr_t = self.hr_t #* (1.0 - self.config.label_smoothing) + 1.0 / self.data_stats.tot_entity
        rt_h = self.rt_h #* (1.0 - self.config.label_smoothing) + 1.0 / self.data_stats.tot_entity

        stacked_h = tf.reshape(h_emb, [-1, 10, 20, 1])
        stacked_r = tf.reshape(r_emb, [-1, 10, 20, 1])
        stacked_t = tf.reshape(t_emb, [-1, 10, 20, 1])

        stacked_hr = tf.concat([stacked_h, stacked_r], 1)
        stacked_tr = tf.concat([stacked_t, stacked_r], 1)

        # TODO make two different forward layers for head and tail
        pred_tails = self.forward(stacked_hr, self.config.batch_size)
        pred_heads = self.forward(stacked_tr, self.config.batch_size)

        loss_tail_pred = tf.reduce_mean(tf.keras.backend.binary_crossentropy(hr_t, pred_tails))
        loss_head_pred = tf.reduce_mean(tf.keras.backend.binary_crossentropy(rt_h, pred_heads))

        # reg_losses = tf.nn.l2_loss(h_emb) + tf.nn.l2_loss(r_emb) + tf.nn.l2_loss(t_emb)

        self.loss = loss_tail_pred + loss_head_pred#+ self.config.lmbda * reg_losses

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

        stacked_hr = tf.concat([stacked_h, stacked_r], 1)
        stacked_tr = tf.concat([stacked_t, stacked_r], 1)

        # TODO make two different forward layers for head and tail
        pred_tails = self.forward(stacked_hr, self.config.batch_size_testing)
        pred_heads = self.forward(stacked_tr, self.config.batch_size_testing)

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

