#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pykg2vec.models.KGMeta import ModelMeta
from pykg2vec.models.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class CP(ModelMeta):

    def __init__(self, config):
        super(CP, self).__init__()
        self.config = config
        self.model_name = 'CP'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.sub_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.obj_embeddings = nn.Embedding(num_total_ent, k)
        nn.init.xavier_uniform_(self.sub_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.obj_embeddings.weight)

        self.parameter_list = [
            NamedEmbedding(self.sub_embeddings, "sub_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.obj_embeddings, "obj_embedding"),
        ]

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = self.sub_embeddings(h)
        emb_r = self.rel_embeddings(r)
        emb_t = self.obj_embeddings(t)
        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        return -torch.sum(h_e * r_e * t_e, -1)

    def get_reg(self, h, r, t, type='N3'):
        h_e, r_e, t_e = self.embed(h, r, t)
        if type.lower() == 'f2':
            regul_term = torch.mean(torch.sum(h_e**2, -1) + torch.sum(r_e**2, -1) + torch.sum(t_e**2, -1))
        elif type.lower() == 'n3':
            regul_term = torch.mean(torch.sum(h_e**3, -1) + torch.sum(r_e**3, -1) + torch.sum(t_e**3, -1))
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % type)

        return self.config.lmbda * regul_term
