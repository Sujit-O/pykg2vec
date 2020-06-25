#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class TransD(ModelMeta):
    """ `Knowledge Graph Embedding via Dynamic Mapping Matrix`_

        TransD constructs a dynamic mapping matrix for each entity-relation pair by considering the diversity of entities and relations simultaneously.
        Compared with TransR/CTransR, TransD has fewer parameters and has no matrix vector multiplication.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.core.TransD import TransD
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransD()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on `OpenKE_TransD`_.

        .. _OpenKE_TransD:
            https://github.com/thunlp/OpenKE/blob/master/models/TransD.py

        .. _Knowledge Graph Embedding via Dynamic Mapping Matrix:
            https://www.aclweb.org/anthology/P15-1067
    """

    def __init__(self, config):
        super(TransD, self).__init__()
        self.config = config
        self.model_name = 'TransD'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        self.ent_embeddings = nn.Embedding(self.config.kg_meta.tot_entity, d)
        self.rel_embeddings = nn.Embedding(self.config.kg_meta.tot_relation, k)
        self.ent_mappings = nn.Embedding(self.config.kg_meta.tot_entity, d)
        self.rel_mappings = nn.Embedding(self.config.kg_meta.tot_relation, k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.ent_mappings.weight)
        nn.init.xavier_uniform_(self.rel_mappings.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.ent_mappings, "ent_mappings"),
            NamedEmbedding(self.rel_mappings, "rel_mappings"),
        ]

    def projection(self, emb_e, emb_m, proj_vec):
        # [b, k] + sigma ([b, k] * [b, k]) * [b, k]
        return emb_e + torch.sum(emb_e * emb_m, axis=-1, keepdims=True) * proj_vec
        
    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = self.ent_embeddings(h)
        emb_r = self.rel_embeddings(r)
        emb_t = self.ent_embeddings(t)

        h_m = self.ent_mappings(h)
        r_m = self.rel_mappings(r)
        t_m = self.ent_mappings(t)

        emb_h = self.projection(emb_h, h_m, r_m)
        emb_t = self.projection(emb_t, t_m, r_m)

        return emb_h, emb_r, emb_t

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

        if self.config.L1_flag:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=1, dim=-1)
        else:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=2, dim=-1)