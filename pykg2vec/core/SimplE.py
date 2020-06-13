#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class SimplE(ModelMeta):

    def __init__(self, config):
        super(SimplE, self).__init__()
        self.config = config
        self.model_name = 'SimplE_avg'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_head_embeddings = nn.Embedding(num_total_ent, k)
        self.ent_tail_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.rel_inv_embeddings = nn.Embedding(num_total_rel, k)
        nn.init.xavier_uniform_(self.ent_head_embeddings.weight)
        nn.init.xavier_uniform_(self.ent_tail_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_head_embeddings, "ent_head_embedding"),
            NamedEmbedding(self.ent_tail_embeddings, "ent_tail_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.rel_inv_embeddings, "rel_inv_embedding"),
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
        emb_h1 = self.ent_head_embeddings(h)
        emb_h2 = self.ent_head_embeddings(t)
        emb_r1 = self.rel_embeddings(r)
        emb_r2 = self.rel_inv_embeddings(r)
        emb_t1 = self.ent_tail_embeddings(t)
        emb_t2 = self.ent_tail_embeddings(h)
        return emb_h1, emb_h2, emb_r1, emb_r2, emb_t1, emb_t2

    def forward(self, h, r, t):
        h1_e, h2_e, r1_e, r2_e, t1_e, t2_e = self.embed(h, r, t)

        init = torch.sum(h1_e*r1_e*t1_e, 1) + torch.sum(h2_e*r2_e*t2_e, 1) / 2.0
        return -torch.clamp(init, -20, 20)

    def get_reg(self, h, r, t):
        num_batch = math.ceil(self.config.kg_meta.tot_train_triples / self.config.batch_size)
        regul_term = (self._get_l2_loss(self.ent_head_embeddings) + self._get_l2_loss(self.ent_tail_embeddings) +
                      self._get_l2_loss(self.rel_embeddings) + self._get_l2_loss(self.rel_inv_embeddings)) / num_batch**2
        return self.config.lmbda * regul_term

    @staticmethod
    def _get_l2_loss(embeddings):
        return torch.sum(embeddings.weight**2) / 2


class SimplE_ignr(SimplE):

    def __init__(self, config):
        super(SimplE_ignr, self).__init__(config)
        self.config = config
        self.model_name = 'SimplE_ignr'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = self._concat_selected_embeddings(self.ent_head_embeddings, h, self.ent_head_embeddings, t)
        emb_r = self._concat_selected_embeddings(self.rel_embeddings, r, self.rel_inv_embeddings, r)
        emb_t = self._concat_selected_embeddings(self.ent_tail_embeddings, t, self.ent_tail_embeddings, h)

        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)

        init = torch.sum(h_e*r_e*t_e, 1)
        return -torch.clamp(init, -20, 20)

    def get_reg(self, h, r, t):
        return 2.0 * super().get_reg(h, r, t)

    @staticmethod
    def _concat_selected_embeddings(e1, t1, e2, t2):
        return torch.cat([torch.index_select(e1.weight, 0, t1), torch.index_select(e2.weight, 0, t2)], 1)



