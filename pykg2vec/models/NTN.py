#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykg2vec.models.KGMeta import ModelMeta
from pykg2vec.models.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class NTN(ModelMeta):
    """ `Reasoning With Neural Tensor Networks for Knowledge Base Completion`_

    It is a neural tensor network which represents entities as an average of their
    constituting word vectors. It then projects entities to their vector embeddings
    in the input layer. The two entities are then combined and mapped to a non-linear hidden layer.
    https://github.com/siddharth-agrawal/Neural-Tensor-Network/blob/master/neuralTensorNetwork.py

     Args:
        config (object): Model configuration parameters.

     Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

     Examples:
        >>> from pykg2vec.models.NTN import NTN
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = NTN()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

     Portion of the code based on `siddharth-agrawal`_.

     .. _siddharth-agrawal:
         https://github.com/siddharth-agrawal/Neural-Tensor-Network/blob/master/neuralTensorNetwork.py

     .. _Reasoning With Neural Tensor Networks for Knowledge Base Completion:
         https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    """

    def __init__(self, config):
        super(NTN, self).__init__()
        self.config = config
        self.model_name = 'NTN'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, d)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.mr1 = nn.Embedding(d, k)
        self.mr2 = nn.Embedding(d, k)
        self.br = nn.Embedding(1, k)
        self.mr = nn.Embedding(k, d*d)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.mr1.weight)
        nn.init.xavier_uniform_(self.mr2.weight)
        nn.init.xavier_uniform_(self.br.weight)
        nn.init.xavier_uniform_(self.mr.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.mr1, "mr1"),
            NamedEmbedding(self.mr2, "mr2"),
            NamedEmbedding(self.br, "br"),
            NamedEmbedding(self.mr, "mr"),
        ]

    def train_layer(self, h, t):
        """Defines the forward pass training layers of the algorithm.

           Args:
               h (Tensor): Head entities ids.
               t (Tensor): Tail entity ids of the triple.
        """
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size
        
        mr1h = torch.matmul(h, self.mr1.weight) # h => [m, d], self.mr1 => [d, k]
        mr2t = torch.matmul(t, self.mr2.weight) # t => [m, d], self.mr2 => [d, k]

        expanded_h = h.unsqueeze(dim=0).repeat(k, 1, 1) # [k, m, d]
        expanded_t = t.unsqueeze(dim=-1) # [m, d, 1]

        temp = (torch.matmul(expanded_h, self.mr.weight.view(k, d, d))).permute(1, 0, 2) # [m, k, d]
        htmrt = torch.squeeze(torch.matmul(temp, expanded_t), dim=-1) # [m, k]

        return F.tanh(htmrt + mr1h + mr2t + self.br.weight)

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
        return -torch.sum(norm_r*self.train_layer(norm_h, norm_t), -1)

    def get_reg(self):
        return self.config.lmbda*torch.sqrt(sum([torch.sum(torch.pow(var.weight, 2)) for var in self.parameter_list]))
