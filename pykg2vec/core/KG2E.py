#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta


class KG2E(ModelMeta):
    """`Learning to Represent Knowledge Graphs with Gaussian Embedding`_

     Instead of assumming entities and relations as determinstic points in the
     embedding vector spaces, KG2E models both entities and relations (h, r and t)
     using random variables derived from multivariate Gaussian distribution.
     KG2E then evaluates a fact using translational relation by evaluating the
     distance between two distributions, r and t-h. KG2E provides two distance
     measures (KL-divergence and estimated likelihood).

     Args:
        config (object): Model configuration parameters.

     Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

     Examples:
        >>> from pykg2vec.core.KG2E import KG2E
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = KG2E()
        >>> trainer = Trainer(model=model, debug=False)
        >>> trainer.build_model()
        >>> trainer.train_model()

     Portion of the code based on `this Source`_.
     .. _this Source:
         https://github.com/mana-ysh/gaussian-embedding/blob/master/src/models/gaussian_model.py

     .. _Learning to Represent Knowledge Graphs with Gaussian Embedding:
         https://pdfs.semanticscholar.org/0ddd/f37145689e5f2899f8081d9971882e6ff1e9.pdf
    """

    def __init__(self, config=None):
        self.config = config
        self.data_stats = self.config.kg_meta

        if self.config.distance_measure == "expected_likelihood":
            self.model_name = 'KG2E_EL'
        else:
            self.model_name = 'KG2E_KL'

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
               ent_embeddings_mu  (Tensor Variable): Lookup variable containing mean of embedding of the entities.
               rel_embeddings_mu  (Tensor Variable): Lookup variable containing mean embedding of the relations.

               ent_embeddings_sigma  (Tensor Variable): Lookup variable containing variance of embedding of the entities.
               rel_embeddings_sigma  (Tensor Variable): Lookup variable containing variance embedding of the relations.

               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.data_stats.tot_entity
        num_total_rel = self.data_stats.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            # the mean for each element in the embedding space. 
            self.ent_embeddings_mu = tf.get_variable(name="ent_embeddings_mu", shape=[num_total_ent, k],
                                                     initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            self.rel_embeddings_mu = tf.get_variable(name="rel_embeddings_mu", shape=[num_total_rel, k],
                                                     initializer=tf.contrib.layers.xavier_initializer(uniform=True))

            # as the paper suggested, sigma is simplified to be the diagonal element in the covariance matrix. 
            self.ent_embeddings_sigma = tf.get_variable(name="ent_embeddings_sigma", shape=[num_total_ent, k],
                                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            self.rel_embeddings_sigma = tf.get_variable(name="rel_embeddings_sigma", shape=[num_total_rel, k],
                                                        initializer=tf.contrib.layers.xavier_initializer(uniform=True))
            self.parameter_list = [self.ent_embeddings_mu, self.ent_embeddings_sigma,
                                   self.rel_embeddings_mu, self.rel_embeddings_sigma]

            self.ent_embeddings_sigma = tf.maximum(self.config.cmin,
                                                   tf.minimum(self.config.cmax, (self.ent_embeddings_sigma + 1.0)))
            self.rel_embeddings_sigma = tf.maximum(self.config.cmin,
                                                   tf.minimum(self.config.cmax, (self.rel_embeddings_sigma + 1.0)))

    def def_loss(self):
        """Defines the loss function for the algorithm."""
        pos_h_mu, pos_h_sigma, pos_r_mu, pos_r_sigma, pos_t_mu, pos_t_sigma = self.get_embed_guassian(self.pos_h,
                                                                                                      self.pos_r,
                                                                                                      self.pos_t)
        neg_h_mu, neg_h_sigma, neg_r_mu, neg_r_sigma, neg_t_mu, neg_t_sigma = self.get_embed_guassian(self.neg_h,
                                                                                                      self.neg_r,
                                                                                                      self.neg_t)

        if self.config.distance_measure == "expected_likelihood":
            score_pos = self.cal_score_expected_likelihood(pos_h_mu, pos_h_sigma, pos_r_mu, pos_r_sigma, pos_t_mu,
                                                           pos_t_sigma)
            score_neg = self.cal_score_expected_likelihood(neg_h_mu, neg_h_sigma, neg_r_mu, neg_r_sigma, neg_t_mu,
                                                           neg_t_sigma)
        else:
            score_pos = self.cal_score_kl_divergence(pos_h_mu, pos_h_sigma, pos_r_mu, pos_r_sigma, pos_t_mu,
                                                     pos_t_sigma)
            score_neg = self.cal_score_kl_divergence(neg_h_mu, neg_h_sigma, neg_r_mu, neg_r_sigma, neg_t_mu,
                                                     neg_t_sigma)

        self.loss = tf.reduce_sum(tf.maximum(score_pos + self.config.margin - score_neg, 0))

    def cal_score_kl_divergence(self, h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma):
        """ It calculates the kl_divergence as a score.

            trace_fac: tr(sigma_r-1 * (sigma_h + sigma_t))
            mul_fac: (mu_h + mu_r - mu_t).T * sigma_r-1 * (mu_h + mu_r - mu_t)
            det_fac: log(det(sigma_r)/det(sigma_h + sigma_t))

            Args:
                 h_mu (Tensor): Mean of the embedding value of the head.
                 h_sigma(Tensor): Variance of the embedding value of the head.
                 r_mu(Tensor): Mean of the embedding value of the relation.
                 r_sigma(Tensor): Variance of the embedding value of the relation.
                 t_mu(Tensor): Mean of the embedding value of the tail.
                 t_sigma(Tensor): Variance of the embedding value of the tail.

            Returns:
                Tensor: Score after calculating the KL_Divergence.

        """
        trace_fac = tf.reduce_sum((h_sigma + t_sigma) / r_sigma, -1)
        mul_fac = tf.reduce_sum((- h_mu + t_mu - r_mu) ** 2 / r_sigma, -1)
        det_fac = tf.reduce_sum(tf.log(h_sigma + t_sigma) - tf.log(r_sigma), -1)

        return trace_fac + mul_fac - det_fac - self.config.hidden_size

    def cal_score_expected_likelihood(self, h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma):
        """ It calculates the expected likelihood as a score.

            mul_fac: (mu_h + mu_r - mu_t).T * sigma_r-1 * (mu_h + mu_r - mu_t)
            det_fac: log(det(sigma_r + sigma_h + sigma_t))

            Args:
                 h_mu (Tensor): Mean of the embedding value of the head.
                 h_sigma(Tensor): Variance of the embedding value of the head.
                 r_mu(Tensor): Mean of the embedding value of the relation.
                 r_sigma(Tensor): Variance of the embedding value of the relation.
                 t_mu(Tensor): Mean of the embedding value of the tail.
                 t_sigma(Tensor): Variance of the embedding value of the tail.

            Returns:
                Tensor: Score after calculating the expected likelihood.
        """
        mul_fac = tf.reduce_sum((h_mu + r_mu - t_mu) ** 2 / (h_sigma + r_sigma + t_sigma), -1)
        det_fac = tf.reduce_sum(tf.log(h_sigma + r_sigma + t_sigma), -1)

        return mul_fac + det_fac - self.config.hidden_size

    def test_batch(self):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        test_h_mu, test_h_sigma, test_r_mu, test_r_sigma, test_t_mu, test_t_sigma = self.get_embed_guassian(
            self.test_h_batch,
            self.test_r_batch,
            self.test_t_batch)
        test_h_mu = tf.expand_dims(test_h_mu, axis=1)
        test_h_sigma = tf.expand_dims(test_h_sigma, axis=1)
        test_r_mu = tf.expand_dims(test_r_mu, axis=1)
        test_r_sigma = tf.expand_dims(test_r_sigma, axis=1)
        test_t_mu = tf.expand_dims(test_t_mu, axis=1)
        test_t_sigma = tf.expand_dims(test_t_sigma, axis=1)

        norm_ent_embeddings_mu = tf.nn.l2_normalize(self.ent_embeddings_mu, axis=1)
        norm_ent_embeddings_sigma = tf.nn.l2_normalize(self.ent_embeddings_sigma, axis=1)

        if self.config.distance_measure == "expected_likelihood":
            score_head = self.cal_score_expected_likelihood(norm_ent_embeddings_mu, norm_ent_embeddings_sigma, \
                                                            test_r_mu, test_r_sigma, \
                                                            test_t_mu, test_t_sigma)

            score_tail = self.cal_score_expected_likelihood(test_h_mu, test_h_sigma, \
                                                            test_r_mu, test_r_sigma, \
                                                            norm_ent_embeddings_mu, norm_ent_embeddings_sigma)
        else:
            score_head = self.cal_score_kl_divergence(norm_ent_embeddings_mu, norm_ent_embeddings_sigma, \
                                                      test_r_mu, test_r_sigma, \
                                                      test_t_mu, test_t_sigma)

            score_tail = self.cal_score_kl_divergence(test_h_mu, test_h_sigma, \
                                                      test_r_mu, test_r_sigma, \
                                                      norm_ent_embeddings_mu, norm_ent_embeddings_sigma)

        _, head_rank = tf.nn.top_k(score_head, k=self.data_stats.tot_entity)
        _, tail_rank = tf.nn.top_k(score_tail, k=self.data_stats.tot_entity)

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

    def get_embed_guassian(self, h, r, t):
        """Function to get the embedding.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        norm_ent_embeddings_mu = tf.nn.l2_normalize(self.ent_embeddings_mu, axis=1)
        norm_rel_embeddings_mu = tf.nn.l2_normalize(self.rel_embeddings_mu, axis=1)

        norm_ent_embeddings_sigma = tf.nn.l2_normalize(self.ent_embeddings_sigma, axis=1)
        norm_rel_embeddings_sigma = tf.nn.l2_normalize(self.rel_embeddings_sigma, axis=1)

        emb_h_mu = tf.nn.embedding_lookup(norm_ent_embeddings_mu, h)
        emb_r_mu = tf.nn.embedding_lookup(norm_rel_embeddings_mu, r)
        emb_t_mu = tf.nn.embedding_lookup(norm_ent_embeddings_mu, t)

        emb_h_sigma = tf.nn.embedding_lookup(norm_ent_embeddings_sigma, h)
        emb_r_sigma = tf.nn.embedding_lookup(norm_rel_embeddings_sigma, r)
        emb_t_sigma = tf.nn.embedding_lookup(norm_ent_embeddings_sigma, t)

        return emb_h_mu, emb_h_sigma, emb_r_mu, emb_r_sigma, emb_t_mu, emb_t_sigma

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
