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
        super(KG2E, self).__init__()
        self.config = config
        if self.config.distance_measure == "expected_likelihood":
            self.model_name = 'KG2E_EL'
        else:
            self.model_name = 'KG2E_KL'

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
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        emb_initializer = tf.initializers.glorot_normal()

        # the mean for each element in the embedding space. 
        self.ent_embeddings_mu = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embeddings_mu")
        self.rel_embeddings_mu = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embeddings_mu")

        # as the paper suggested, sigma is simplified to be the diagonal element in the covariance matrix. 
        self.ent_embeddings_sigma = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embeddings_sigma")
        self.rel_embeddings_sigma = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embeddings_sigma")
        
        self.parameter_list = [self.ent_embeddings_mu, self.ent_embeddings_sigma, self.rel_embeddings_mu, self.rel_embeddings_sigma]

        self.ent_embeddings_sigma = tf.maximum(self.config.cmin, tf.minimum(self.config.cmax, (self.ent_embeddings_sigma + 1.0)))
        self.rel_embeddings_sigma = tf.maximum(self.config.cmin, tf.minimum(self.config.cmax, (self.rel_embeddings_sigma + 1.0)))

    def get_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        """Defines the loss function for the algorithm."""
        pos_h_mu, pos_h_sigma, pos_r_mu, pos_r_sigma, pos_t_mu, pos_t_sigma = self.embed(pos_h, pos_r, pos_t)
        neg_h_mu, neg_h_sigma, neg_r_mu, neg_r_sigma, neg_t_mu, neg_t_sigma = self.embed(neg_h, neg_r, neg_t)

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

        loss = tf.reduce_sum(tf.maximum(score_pos + self.config.margin - score_neg, 0))

        return loss

    def predict(self, h, r, t, topk=-1):
        """Function that performs prediction for TransE. 
           shape of h can be either [num_tot_entity] or [1]. 
           shape of t can be either [num_tot_entity] or [1].

          Returns:
              Tensors: Returns ranks of head and tail.
        """
        h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma = self.embed(h, r, t)
        if self.config.distance_measure == "expected_likelihood":
            score = self.cal_score_expected_likelihood(h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma)
        else: 
            score = self.cal_score_kl_divergence(h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma)
        _, rank = tf.nn.top_k(score, k=topk)

        return rank

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
        det_fac = tf.reduce_sum(tf.math.log(h_sigma + t_sigma) - tf.math.log(r_sigma), -1)

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
        det_fac = tf.reduce_sum(tf.math.log(h_sigma + r_sigma + t_sigma), -1)

        return mul_fac + det_fac - self.config.hidden_size

    def test_batch(self, h_batch, r_batch, t_batch):
        """Function that performs batch testing for the algorithm.

            Returns:
                Tensors: Returns ranks of head and tail.
        """
        test_h_mu, test_h_sigma, test_r_mu, test_r_sigma, test_t_mu, test_t_sigma = self.embed(h_batch, r_batch, t_batch)
        test_h_mu = tf.expand_dims(test_h_mu, axis=1)
        test_h_sigma = tf.expand_dims(test_h_sigma, axis=1)
        test_r_mu = tf.expand_dims(test_r_mu, axis=1)
        test_r_sigma = tf.expand_dims(test_r_sigma, axis=1)
        test_t_mu = tf.expand_dims(test_t_mu, axis=1)
        test_t_sigma = tf.expand_dims(test_t_sigma, axis=1)

        norm_ent_embeddings_mu = tf.nn.l2_normalize(self.ent_embeddings_mu, axis=1)
        norm_ent_embeddings_sigma = tf.nn.l2_normalize(self.ent_embeddings_sigma, axis=1)

        if self.config.distance_measure == "expected_likelihood":
            score_head = self.cal_score_expected_likelihood(norm_ent_embeddings_mu, norm_ent_embeddings_sigma, test_r_mu, test_r_sigma, test_t_mu, test_t_sigma)
            score_tail = self.cal_score_expected_likelihood(test_h_mu, test_h_sigma, test_r_mu, test_r_sigma, norm_ent_embeddings_mu, norm_ent_embeddings_sigma)
        else:
            score_head = self.cal_score_kl_divergence(norm_ent_embeddings_mu, norm_ent_embeddings_sigma, test_r_mu, test_r_sigma, test_t_mu, test_t_sigma)
            score_tail = self.cal_score_kl_divergence(test_h_mu, test_h_sigma, test_r_mu, test_r_sigma, norm_ent_embeddings_mu, norm_ent_embeddings_sigma)

        _, head_rank = tf.nn.top_k(score_head, k=self.config.kg_meta.tot_entity)
        _, tail_rank = tf.nn.top_k(score_tail, k=self.config.kg_meta.tot_entity)

        return head_rank, tail_rank

    def embed(self, h, r, t):
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