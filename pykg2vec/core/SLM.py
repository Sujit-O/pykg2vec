#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class SLM(ModelMeta):
    """`Reasoning With Neural Tensor Networks for Knowledge Base Completion`_

        SLM model is designed as a baseline of Neural Tensor Network.
        The model constructs a nonlinear neural network to represent the score function.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.SLM import SLM
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = SLM()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _Reasoning With Neural Tensor Networks for Knowledge Base Completion:
            https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    """

    def __init__(self, config):
        super(SLM, self).__init__()
        self.config = config
        self.model_name = 'SLM'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, d)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.mr1 = nn.Embedding(d, k)
        self.mr2 = nn.Embedding(d, k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.mr1.weight)
        nn.init.xavier_uniform_(self.mr2.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.mr1, "mr1"),
            NamedEmbedding(self.mr2, "mr2"),
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
        emb_h = self.ent_embeddings(h)
        emb_r = self.rel_embeddings(r)
        emb_t = self.ent_embeddings(t)
        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        norm_h = F.normalize(h_e, p=2, dim=-1)
        norm_r = F.normalize(r_e, p=2, dim=-1)
        norm_t = F.normalize(t_e, p=2, dim=-1)
        return -torch.sum(norm_r * self.layer(norm_h, norm_t), -1)

    def layer(self, h, t):
        """Defines the forward pass layer of the algorithm.

          Args:
              h (Tensor): Head entities ids.
              t (Tensor): Tail entity ids of the triple.
        """       
        mr1h = torch.matmul(h, self.mr1.weight) # h => [m, d], self.mr1 => [d, k]
        mr2t = torch.matmul(t, self.mr2.weight) # t => [m, d], self.mr2 => [d, k]
        return torch.tanh(mr1h + mr2t)