#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(ModelMeta):
    """ `Translating Embeddings for Modeling Multi-relational Data`_

        TransE is an energy based model which represents the
        relationships as translations in the embedding space. Which
        means that if (h,l,t) holds then the embedding of the tail
        't' should be close to the embedding of head entity 'h'
        plus some vector that depends on the relationship 'l'.
        Both entities and relations are vectors in the same space.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.TransE import TransE
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransE()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on `OpenKE_TransE`_ and `wencolani`_.

        .. _OpenKE_TransE: https://github.com/thunlp/OpenKE/blob/master/models/TransE.py

        .. _wencolani: https://github.com/wencolani/TransE.git

        .. _Translating Embeddings for Modeling Multi-relational Data:
            http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela
    """

    def __init__(self, config):

        super(TransE, self).__init__()

        self.config = config
        self.model_name = 'TransE'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               ent_embeddings (Tensor Variable): Lookup variable containing  embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing  embedding of the relations.
               parameter_list  (list): List of Tensor parameters.
        """
        self.num_tot_ent = self.config.kg_meta.tot_entity
        self.num_tot_rel = self.config.kg_meta.tot_relation

        self.ent_embeddings = nn.Embedding(self.num_tot_ent, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.num_tot_rel, self.config.hidden_size)

        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

    def forward(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids.
               t (Tensor): Tail entity ids.

            Returns:
                Tensors: the scores of evaluationReturns head, relation and tail embedding Tensors.
        """
        h_e, r_e, t_e = self.embed(h, r, t)

        norm_h_e = F.normalize(h_e, p=2, dim=-1)
        norm_r_e = F.normalize(r_e, p=2, dim=-1)
        norm_t_e = F.normalize(t_e, p=2, dim=-1)

        score = norm_h_e + norm_r_e  - norm_t_e

        if self.config.L1_flag:
            score = torch.norm(score, p=1, dim=-1)
        else:
            score = torch.norm(score, p=2, dim=-1)

        return score

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids.
               t (Tensor): Tail entity ids.

            Returns:
                Tensors: Returns a tuple of head, relation and tail embedding Tensors.
        """
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        t_e = self.ent_embeddings(t)

        return h_e, r_e, t_e