#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


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
        >>> trainer = Trainer(model=model)
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
        self.model_name = 'KG2E_KL'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        # the mean for each element in the embedding space. 
        self.ent_embeddings_mu = nn.Embedding(num_total_ent, k)
        self.rel_embeddings_mu = nn.Embedding(num_total_rel, k)

        # as the paper suggested, sigma is simplified to be the diagonal element in the covariance matrix. 
        self.ent_embeddings_sigma = nn.Embedding(num_total_ent, k)
        self.rel_embeddings_sigma = nn.Embedding(num_total_rel, k)

        nn.init.xavier_uniform_(self.ent_embeddings_mu.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_mu.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_sigma.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_sigma.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings_mu, "ent_embeddings_mu"),
            NamedEmbedding(self.ent_embeddings_sigma, "ent_embeddings_sigma"),
            NamedEmbedding(self.rel_embeddings_mu, "rel_embeddings_mu"),
            NamedEmbedding(self.rel_embeddings_sigma, "rel_embeddings_sigma"),
        ]

        min_ent = torch.min(torch.FloatTensor().new_full(self.ent_embeddings_sigma.weight.shape, self.config.cmax), torch.add(self.ent_embeddings_sigma.weight, 1.0))
        self.ent_embeddings_sigma.weight = nn.Parameter(torch.max(torch.FloatTensor().new_full(self.ent_embeddings_sigma.weight.shape, self.config.cmin), min_ent))
        min_rel = torch.min(torch.FloatTensor().new_full(self.rel_embeddings_sigma.weight.shape, self.config.cmax), torch.add(self.rel_embeddings_sigma.weight, 1.0))
        self.rel_embeddings_sigma.weight = nn.Parameter(torch.max(torch.FloatTensor().new_full(self.rel_embeddings_sigma.weight.shape, self.config.cmin), min_rel))

    def forward(self, h, r, t):
        h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma = self.embed(h, r, t)
        return self._cal_score_expected_likelihood(h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma)

    def embed(self, h, r, t):
        """Function to get the embedding.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """

        self.ent_embeddings_mu.weight.data = self.get_normalized_data(self.ent_embeddings_mu, self.config.kg_meta.tot_entity)
        self.rel_embeddings_mu.weight.data = self.get_normalized_data(self.rel_embeddings_mu, self.config.kg_meta.tot_relation)

        self.ent_embeddings_sigma.weight.data = self.get_normalized_data(self.ent_embeddings_sigma, self.config.kg_meta.tot_entity)
        self.rel_embeddings_sigma.weight.data = self.get_normalized_data(self.rel_embeddings_sigma, self.config.kg_meta.tot_relation)

        emb_h_mu = self.ent_embeddings_mu(h)
        emb_r_mu = self.rel_embeddings_mu(r)
        emb_t_mu = self.ent_embeddings_mu(t)

        emb_h_sigma = self.ent_embeddings_sigma(h)
        emb_r_sigma = self.rel_embeddings_sigma(r)
        emb_t_sigma = self.ent_embeddings_sigma(t)

        return emb_h_mu, emb_h_sigma, emb_r_mu, emb_r_sigma, emb_t_mu, emb_t_sigma

    @staticmethod
    def get_normalized_data(embedding, num_embeddings, p=2, dim=1):
        norms = torch.norm(embedding.weight, p, dim).data
        return embedding.weight.data.div(norms.view(num_embeddings, 1).expand_as(embedding.weight))

    def _cal_score_expected_likelihood(self, h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma):
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
        mul_fac = torch.sum((h_mu + r_mu - t_mu) ** 2 / (h_sigma + r_sigma + t_sigma), -1)
        det_fac = torch.sum(torch.log(h_sigma + r_sigma + t_sigma), -1)

        return mul_fac + det_fac - self.config.hidden_size

class KG2E_EL(KG2E):
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
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    Portion of the code based on `this Source`_.
    
    .. _this Source:
        https://github.com/mana-ysh/gaussian-embedding/blob/master/src/models/gaussian_model.py

    .. _Learning to Represent Knowledge Graphs with Gaussian Embedding:
        https://pdfs.semanticscholar.org/0ddd/f37145689e5f2899f8081d9971882e6ff1e9.pdf
    
    """

    def __init__(self, config):
        super(KG2E_EL, self).__init__(config)
        self.config = config
        self.model_name = 'KG2E_EL'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    def forward(self, h, r, t):
        h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma = self.embed(h, r, t)
        return self._cal_score_kl_divergence(h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma)

    def _cal_score_kl_divergence(self, h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma):
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
        trace_fac = torch.sum((h_sigma + t_sigma) / r_sigma, -1)
        mul_fac = torch.sum((- h_mu + t_mu - r_mu) ** 2 / r_sigma, -1)
        det_fac = torch.sum(torch.log(h_sigma + t_sigma) - torch.log(r_sigma), -1)

        return trace_fac + mul_fac - det_fac - self.config.hidden_size