#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class TuckER(ModelMeta):
    """ `TuckER-Tensor Factorization for Knowledge Graph Completion`_

        TuckER is a Tensor-factorization-based embedding technique based on
        the Tucker decomposition of a third-order binary tensor of triplets. Although
        being fully expressive, the number of parameters used in Tucker only grows linearly
        with respect to embedding dimension as the number of entities or relations in a
        knowledge graph increases.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.TuckER import TuckER
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TuckER()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _TuckER-Tensor Factorization for Knowledge Graph Completion:
            https://arxiv.org/pdf/1901.09590.pdf

    """

    def __init__(self, config=None):
        super(TuckER, self).__init__()
        self.config = config
        self.model_name = 'TuckER'
        self.training_strategy = TrainingStrategy.PROJECTION_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        self.d1 = self.config.ent_hidden_size
        self.d2 = self.config.rel_hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, self.d1)
        self.rel_embeddings = nn.Embedding(num_total_rel, self.d2)
        self.W = nn.Embedding(self.d2, self.d1 * self.d1)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.W.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.W, "W"),
        ]

        self.inp_drop = nn.Dropout(self.config.input_dropout)
        self.hidden_dropout1 = nn.Dropout(self.config.hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(self.config.hidden_dropout2)

    def forward(self, e1, r, direction=None):
        """Implementation of the layer.

            Args:
                e1(Tensor): entities id.
                r(Tensor): Relation id.

            Returns:
                Tensors: Returns the activation values.
        """
        e1 = self.ent_embeddings(e1)
        e1 = F.normalize(e1, p=2, dim=1)
        e1 = self.inp_drop(e1)
        e1 = e1.view(-1, 1, self.d1)

        rel = self.rel_embeddings(r)
        W_mat = torch.matmul(rel, self.W.weight.view(self.d2, -1))
        W_mat = W_mat.view(-1, self.d1, self.d1)
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.matmul(e1, W_mat)
        x = x.view(-1, self.d1)
        x = F.normalize(x, p=2, dim=1)
        x = self.hidden_dropout2(x)
        x = torch.matmul(x, self.transpose(self.ent_embeddings.weight))
        return F.sigmoid(x)

    def predict_tail_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r), k=topk)
        return rank

    def predict_head_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r), k=topk)
        return rank

    @staticmethod
    def transpose(tensor):
        dims = tuple(range(len(tensor.shape)-1, -1, -1))    # (rank-1...0)
        return tensor.permute(dims)