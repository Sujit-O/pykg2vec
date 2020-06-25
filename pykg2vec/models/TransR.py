#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykg2vec.models.KGMeta import ModelMeta
from pykg2vec.models.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class TransR(ModelMeta):
    """ `Learning Entity and Relation Embeddings for Knowledge Graph Completion`_

        TranR is a translation based knowledge graph embedding method. Similar to TransE and TransH, it also
        builds entity and relation embeddings by regarding a relation as translation from head entity to tail
        entity. However, compared to them, it builds the entity and relation embeddings in a separate entity
        and relation spaces.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.models.TransR import TransR
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransR()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on `thunlp_transR`_.

         .. _thunlp_transR:
             https://github.com/thunlp/TensorFlow-TransX/blob/master/transR.py

        .. _Learning Entity and Relation Embeddings for Knowledge Graph Completion:
            http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf
    """

    def __init__(self, config):
        super(TransR, self).__init__()
        self.config = config
        self.model_name = 'TransR'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.ent_hidden_size
        d = self.config.rel_hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, d)
        self.rel_matrix = nn.Embedding(num_total_rel, k * d)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_matrix.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.rel_matrix, "rel_matrix"),
        ]

    def transform(self, e, matrix):
        matrix = matrix.view(-1, self.config.ent_hidden_size, self.config.rel_hidden_size)
        if e.shape[0] != matrix.shape[0]:
            e = e.view(-1, matrix.shape[0], self.config.ent_hidden_size).permute(1, 0, 2)
            e = torch.matmul(e, matrix).permute(1, 0, 2)
        else:
            e = e.view(-1, 1, self.config.ent_hidden_size)
            e = torch.matmul(e, matrix)
        return e.view(-1, self.config.rel_hidden_size)

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        t_e = self.ent_embeddings(t)

        h_e = F.normalize(h_e, p=2, dim=-1)
        r_e = F.normalize(r_e, p=2, dim=-1)
        t_e = F.normalize(t_e, p=2, dim=-1)

        h_e = torch.unsqueeze(h_e, 1)
        t_e = torch.unsqueeze(t_e, 1)
        # [b, 1, k]

        matrix = self.rel_matrix(r)
        # [b, k, d]

        transform_h_e = self.transform(h_e, matrix)
        transform_t_e = self.transform(t_e, matrix)
        # [b, 1, d] = [b, 1, k] * [b, k, d]

        h_e = torch.squeeze(transform_h_e, axis=1)
        t_e = torch.squeeze(transform_t_e, axis=1)
        # [b, d]
        return h_e, r_e, t_e

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