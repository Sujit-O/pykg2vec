#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from pykg2vec.core.KGMeta import ModelMeta


class TransG(ModelMeta):
    """ `TransG-A Generative Model for Knowledge Graph Embedding`_

        Generative based embedding that utilized clustering to find the
        relevant entity cluster for the given relation.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.TransG import TransG
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransG()
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on Niubohan_ and BookmanHan_.
        
        .. _Niubohan:
            https://github.com/Niubohan/TransG

        .. _BookmanHan:
            https://github.com/BookmanHan/Embedding

        .. _TransG-A Generative Model for Knowledge Graph Embedding:
            https://www.aclweb.org/anthology/P16-1219
    """

    def __init__(self, config=None):
        self.config = config
        self.model_name = 'TransG'

    def def_inputs(self):
        """Defines the inputs to the model.

           Attributes:
              pos_h (Tensor): Positive Head entities ids.
              pos_r (Tensor): Positive Relation ids of the triple.
              pos_t (Tensor): Positive Tail entity ids of the triple.
              neg_h (Tensor): Negative Head entities ids.
              neg_r (Tensor): Negative Relation ids of the triple.
              neg_t (Tensor): Negative Tail entity ids of the triple.
              test_h_batch (Tensor): Batch of head ids for testing.
              test_r_batch (Tensor): Batch of relation ids for testing
              test_t_batch (Tensor): Batch of tail ids for testing.
        """
        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        self.test_h_batch = tf.placeholder(tf.int32, [None])
        self.test_t_batch = tf.placeholder(tf.int32, [None])
        self.test_r_batch = tf.placeholder(tf.int32, [None])

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings (Tensor Variable): Lookup variable containing  embedding of the entities.
               rel_clusters  (Tensor Variable): Relation clusters for the entities.
               rel_clusters  (Tensor Variable): Weights cluster.
               c (int): Number of initial clusters.
               CRP (float): Chinese Restaurant Process factor
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size
        self.c = self.config.ncluster
        self.CRP = self.config.CRP_factor / self.config.kg_meta.tot_train_triples * num_total_rel

        with tf.name_scope("embedding"):
            # each entity is represented as a normal distribution. 
            self.ent_embeddings = tf.get_variable(name="ent_embedding", shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            # TODO: use the variance. 
            self.ent_embeddings_var = tf.get_variable(name="ent_embedding_var", shape=[num_total_ent],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.rel_clusters = tf.get_variable(name="rel_clusters", shape=[num_total_rel, self.config.ncluster, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.weights_clusters = tf.get_variable(name="weights_clusters", shape=[num_total_rel, self.config.ncluster],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            
            # clusters = [[1/self.config.ncluster for x in range(self.config.ncluster)] for y in range(num_total_rel)]
            # self.weights_clusters = tf.Variable(np.asarray(clusters, dtype=np.float32), trainable=False)
            # import pdb; pdb.set_trace()
            self.parameter_list = [self.ent_embeddings, self.rel_clusters, self.weights_clusters, self.ent_embeddings_var]

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_clusters = tf.nn.l2_normalize(self.rel_clusters, axis=2)
        
        pos_h_e = tf.nn.embedding_lookup(norm_ent_embeddings, self.pos_h)
        pos_t_e = tf.nn.embedding_lookup(norm_ent_embeddings, self.pos_t)
        pos_r_c_e = tf.nn.embedding_lookup(norm_rel_clusters, self.pos_r)
        pos_r_w_e = tf.nn.embedding_lookup(self.weights_clusters, self.pos_r)

        pos_h_e = tf.expand_dims(pos_h_e, axis=1) # ?, 1, 50
        pos_t_e = tf.expand_dims(pos_t_e, axis=1) # ?, 1, 50

        pos_exp_e = (pos_h_e - pos_t_e) + pos_r_c_e # ?, 4, 50
        pos_exp_e = tf.math.exp(tf.negative(tf.reduce_sum(pos_exp_e ** 2, axis=2)))
        pos_score = tf.reduce_sum(pos_exp_e * pos_r_w_e, axis=1)

        neg_h_e = tf.nn.embedding_lookup(norm_ent_embeddings, self.neg_h)
        neg_t_e = tf.nn.embedding_lookup(norm_ent_embeddings, self.neg_t)
        neg_r_c_e = tf.nn.embedding_lookup(norm_rel_clusters, self.neg_r)
        neg_r_w_e = tf.nn.embedding_lookup(self.weights_clusters, self.neg_r)

        neg_h_e = tf.expand_dims(neg_h_e, axis=1)
        neg_t_e = tf.expand_dims(neg_t_e, axis=1)

        neg_exp_e = (neg_h_e - neg_t_e) + neg_r_c_e
        neg_exp_e = tf.math.exp(tf.negative(tf.reduce_sum(neg_exp_e ** 2, axis=2)))
        neg_score = tf.reduce_sum(neg_exp_e * neg_r_w_e, axis=1)
        
        # regularizers 
        self.loss = 0.001 * tf.reduce_sum(norm_ent_embeddings**2) + tf.reduce_sum(norm_rel_clusters**2)
        self.loss = tf.reduce_sum(tf.maximum(- pos_score + self.config.margin + neg_score, 0))
        # import pdb; pdb.set_trace()
        # tf.math.exp

        # neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
        # neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)

        # pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
        # neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        # loss=0
        # c = lambda i: tf.less(i, self.config.batch_size)
        # b = lambda i,loss: [i+1, loss+self.train_step(self.pos_h[i],
        #     self.pos_r[i], self.pos_t[i],self.neg_h[i],self.neg_r[i],self.neg_t[i])]
        # tf.while_loop(c, b, [0,loss])
        # self.loss = loss

    # def prob_triples(self, h,r,t):
    #     """Function to that calculates probability fot he triple.

    #        Args:
    #           h (Tensor): Head entities ids.
    #           r (Tensor): Relation ids of the triple.
    #           t (Tensor): Tail entity ids of the triple.

    #         Returns:
    #            Tensors: Returns the probablity.
    #     """
    #     mixed_prob = 1e-100
    #     error_c = self.ent_embeddings[h] + self.rel_clusters[r][0:self.c] - self.ent_embeddings[t]
    #     mixed_prob = tf.math.abs(self.weights_clusters[r][0:self.c]) * tf.math.exp(-tf.reduce_sum(tf.math.abs(error_c),[0,2]))
    #     mixed_prob = tf.math.reduce_max(mixed_prob)
    #     return mixed_prob

    # def training_prob_triples(self, h,r,t):
    #     """Function to that calculates training probability fot he triple.

    #        Args:
    #           h (Tensor): Head entities ids.
    #           r (Tensor): Relation ids of the triple.
    #           t (Tensor): Tail entity ids of the triple.

    #         Returns:
    #            Tensors: Returns the probablity.
    #     """
    #     mixed_prob = 1e-100
    #     error_c = self.ent_embeddings[h] + self.rel_clusters[r][0:self.c] - self.ent_embeddings[t]
    #     mixed_prob = tf.math.abs(self.weights_clusters[r][0:self.c]) * tf.math.exp(-tf.reduce_sum(tf.math.abs(error_c),[0,2]))
    #     mixed_prob = tf.reduce_sum(mixed_prob)
    #     return mixed_prob

    # def train_cluster_once(self, ph, pr, pt, nh, nr, nt, c, prob_true, prob_false,factor):
    #     """Function to train the cluster once.

    #        Todo:
    #           * Define all the arguments.
    #     """
    #     prob_local_true = tf.math.exp(-tf.reduce_sum(tf.math.abs(self.ent_embeddings[ph] + self.rel_clusters[pr][c] -
    #                                               self.ent_embeddings[pt])))
    #     prob_local_false = tf.math.exp(-tf.reduce_sum(tf.math.abs(self.ent_embeddings[nh] + self.rel_clusters[nr][c] -
    #                                               self.ent_embeddings[nt])))
    #     self.weights_clusters[pr][0, c] += factor / prob_true * prob_local_true * tf.sign(self.weights_clusters[pr][0, c])
    #     self.weights_clusters[nr][0, c] -= factor / prob_false * prob_local_false * tf.sign(self.weights_clusters[nr][0, c])
        
    #     change = factor * prob_local_true / prob_true * tf.math.abs(self.weights_clusters[pr][0, c])
    #     change_f = factor * prob_local_false / prob_false * tf.math.abs(self.weights_clusters[nr][0, c])

    #     self.ent_embeddings[ph] -= change * tf.sign(self.ent_embeddings[ph] + self.rel_clusters[pr][c] - self.ent_embeddings[pt])
    #     self.ent_embeddings[pt] += change * tf.sign(self.ent_embeddings[ph] + self.rel_clusters[pr][c] - self.ent_embeddings[pt])
    #     self.rel_clusters[pr][c] -= change * tf.sign(self.ent_embeddings[ph] + self.rel_clusters[pr][c] -self.ent_embeddings[pt])
    #     self.ent_embeddings[nh] += change_f * tf.sign(self.ent_embeddings[nh] + self.rel_clusters[nr][c] -self.ent_embeddings[nt])
    #     self.ent_embeddings[nt] -= change_f * tf.sign(self.ent_embeddings[nh] + self.rel_clusters[nr][c] -self.ent_embeddings[nt])
    #     self.rel_clusters[nr][c] += change_f * tf.sign(self.ent_embeddings[nh] + self.rel_clusters[nr][c] - self.ent_embeddings[nt])

    #     self.rel_clusters[pr][c]=tf.cond(tf.less(1.0, tf.norm(self.rel_clusters[pr][c])), 
    #         lambda: tf.nn.l2_normalize(self.rel_clusters[pr][c]), lambda: self.rel_clusters[pr][c])

    #     self.rel_clusters[nr][c]=tf.cond(tf.less(1.0, tf.norm(self.rel_clusters[nr][c])), 
    #         lambda: tf.nn.l2_normalize(self.rel_clusters[nr][c]), lambda: self.rel_clusters[nr][c])

    # def f1(self,ph,pr,pt,nh,nr,nt, prob_true, prob_false, cur_epoch):
    #     """Function to calculate score.

    #        Todo:
    #           * Define all the arguments.
    #     """
        
    #     for i in range(self.c):
    #         self.train_cluster_once(ph,pr,pt,nh,nr,nt, i, prob_true, prob_false, self.alpha)

    #     prob_new_component = self.CRP * tf.math.exp(-tf.reduce_sum(tf.abs(self.ent_embeddings[ph] - self.ent_embeddings[pt])))
        
    #     if random.random() < prob_new_component / (prob_new_component + prob_true) \
    #     and self.c < 20 and cur_epoch >= self.config.step_before:
    #         component = self.c
    #         self.weights_clusters[pr][0,component] = self.CRP
    #         self.c+=1
        
    #     self.ent_embeddings[ph]=tf.cond(tf.less(1.0, tf.norm(self.ent_embeddings[ph])), 
    #         lambda: tf.nn.l2_normalize(self.ent_embeddings[ph]), lambda: self.ent_embeddings[ph])
    #     self.ent_embeddings[pt]=tf.cond(tf.less(1.0, tf.norm(self.ent_embeddings[pt])), 
    #         lambda: tf.nn.l2_normalize(self.ent_embeddings[pt]), lambda: self.ent_embeddings[pt])
    #     self.ent_embeddings[nh]=tf.cond(tf.less(1.0, tf.norm(self.ent_embeddings[nh])), 
    #         lambda: tf.nn.l2_normalize(self.ent_embeddings[nh]), lambda: self.ent_embeddings[nh])
    #     self.ent_embeddings[nt]=tf.cond(tf.less(1.0, tf.norm(self.ent_embeddings[nt])), 
    #         lambda: tf.nn.l2_normalize(self.ent_embeddings[nt]), lambda: self.ent_embeddings[nt])
    #     if self.config.weight_norm:
    #         self.weights_clusters[pr] = tf.nn.l2_normalize(self.weights_clusters[pr])  

    #     return prob_false / prob_true  

    # def train_step(self,ph,pr,pt,nh,nr,nt):
    #     """Trains the clusters for one step"""
    #     prob_true = self.training_prob_triples(ph, pr, pt)
    #     prob_false = self.training_prob_triples(nh, nr, nt)
        
    #     loss = tf.cond(tf.less(tf.math.exp(self.config.training_threshold), prob_true / prob_false), 
    #         lambda: f1(ph,pr,pt,nh,nr,nt, prob_true, prob_false, cur_epoch), lambda: prob_false / prob_true)

    #     return loss


    def test_batch(self):
        """Function that performs batch testing for the algorithm.

          Returns:
              Tensors: Returns ranks of head and tail.
        """
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_clusters = tf.nn.l2_normalize(self.rel_clusters, axis=2)
        
        head_vec = tf.nn.embedding_lookup(norm_ent_embeddings, self.test_h_batch)
        tail_vec = tf.nn.embedding_lookup(norm_ent_embeddings, self.test_t_batch) 

        rel_c_vec = tf.nn.embedding_lookup(norm_rel_clusters, self.test_r_batch) # ?, 4, 50
        rel_r_w_vec = tf.nn.embedding_lookup(self.weights_clusters, self.test_r_batch) # ?, 4

        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        
        all_ent_embeddings = tf.expand_dims(norm_ent_embeddings, axis=1) # 14951, 1, 50
        all_ent_embeddings = tf.expand_dims(all_ent_embeddings, axis=0) # 1, 14951, 1, 50
        
        head_vec = tf.expand_dims(head_vec, axis=1) 
        head_vec = tf.expand_dims(head_vec, axis=1) # ?, 1, 1, 50
        tail_vec = tf.expand_dims(tail_vec, axis=1) 
        tail_vec = tf.expand_dims(tail_vec, axis=1) # ?, 1, 1, 50
        rel_c_vec = tf.expand_dims(rel_c_vec, axis=1) # ?, 1, 4, 50
        rel_r_w_vec = tf.expand_dims(rel_r_w_vec, axis=1) # ?, 1, 4

        head_exp_e = (all_ent_embeddings - tail_vec) + rel_c_vec # ?, 14951, 4, 50
        head_exp_e = tf.math.exp(tf.negative(tf.reduce_sum(head_exp_e ** 2, axis=-1))) # ?, 14951, 4
        head_score = tf.reduce_sum(head_exp_e*rel_r_w_vec, axis=-1)

        tail_exp_e = (head_vec - all_ent_embeddings) + rel_c_vec # ?, 14951, 4, 50
        tail_exp_e = tf.math.exp(tf.negative(tf.reduce_sum(tail_exp_e ** 2, axis=-1))) # ?, 14951, 4
        tail_score = tf.reduce_sum(tail_exp_e*rel_r_w_vec, axis=-1)

        _, head_rank = tf.nn.top_k(-head_score, k=self.config.kg_meta.tot_entity)
        _, tail_rank = tf.nn.top_k(-tail_score, k=self.config.kg_meta.tot_entity)

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
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        emb_h = tf.nn.embedding_lookup(norm_ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(norm_ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def get_embed(self, h, r, t, sess):
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

    def get_proj_embed(self, h, r, t, sess=None):
        """"Function to get the projected embedding value in numpy.

            Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.
               sess (object): Tensorflow Session object.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        return self.get_embed(h, r, t, sess)
