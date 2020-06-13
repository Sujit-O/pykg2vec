#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class HoLE(ModelMeta):
    """`Holographic Embeddings of Knowledge Graphs`_.

    HoLE employs the circular correlation to create composition correlations. It
    is able to represent and capture the interactions betweek entities and relations
    while being efficient to compute, easier to train and scalable to large dataset.

    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
    
    Examples:
        >>> from pykg2vec.core.HoLE import HoLE
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = HoLE()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Holographic Embeddings of Knowledge Graphs:
        https://arxiv.org/pdf/1510.04935.pdf

    """

    def __init__(self, config):
        super(HoLE, self).__init__()
        self.config = config
        self.model_name = 'HoLE'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation

        self.ent_embeddings = nn.Embedding(num_total_ent, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(num_total_rel, self.config.hidden_size)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
        ]

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        r_e = F.normalize(r_e, p=2, dim=-1)
        h_e = torch.stack((h_e, torch.zeros_like(h_e)), -1)
        t_e = torch.stack((t_e, torch.zeros_like(t_e)), -1)
        e, _ = torch.unbind(torch.ifft(torch.conj(torch.fft(h_e, 1)) * torch.fft(t_e, 1), 1), -1)
        return -F.sigmoid(torch.sum(r_e * e, 1))

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