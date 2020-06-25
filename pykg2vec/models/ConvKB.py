#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pykg2vec.models.KGMeta import ModelMeta
from pykg2vec.models.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class ConvKB(ModelMeta):
    """`A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network`_

    ConvKB, each triple (head entity, relation, tail entity) is represented as a 3-
    column matrix where each column vector represents a triple element

    Args:
        config (object): Model configuration parameters.
    
    Attributes:
        config (object): Model configuration.
        data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
        model (str): Name of the model.
        last_dim (int): The size of the last dimesion, depends on hidden size.

    
    Examples:
        >>> from pykg2vec.models.ConvKB import ConvKB
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = ConvKB()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()
    
    Portion of the code based on Niubohan_ and BookmanHan_.
    .. _daiquocnguyen:
        https://github.com/daiquocnguyen/ConvKB

    .. _A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network:
        https://www.aclweb.org/anthology/N18-2053

    """

    def __init__(self, config):
        super(ConvKB, self).__init__()
        self.config = config
        self.model_name = 'ConvKB'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
        ]

        self.conv_list = [nn.Conv2d(1, self.config.num_filters, (3, filter_size), stride=(1, 1)).to(self.config.device) for filter_size in self.config.filter_sizes]
        conv_out_dim = self.config.num_filters*sum([(self.config.hidden_size-filter_size+1) for filter_size in self.config.filter_sizes])
        self.fc1 = nn.Linear(in_features=conv_out_dim, out_features=1, bias=True)

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
        h_emb, r_emb, t_emb = self.embed(h, r, t)
        first_dimen = list(h_emb.shape)[0]
        
        stacked_h = torch.unsqueeze(h_emb, dim=1)
        stacked_r = torch.unsqueeze(r_emb, dim=1)
        stacked_t = torch.unsqueeze(t_emb, dim=1)

        stacked_hrt = torch.cat([stacked_h, stacked_r, stacked_t], dim=1)
        stacked_hrt = torch.unsqueeze(stacked_hrt, dim=1)  # [b, 1, 3, k]

        stacked_hrt = [conv_layer(stacked_hrt) for conv_layer in self.conv_list]
        stacked_hrt = torch.cat(stacked_hrt, dim=3)
        stacked_hrt = stacked_hrt.view(first_dimen, -1)
        preds = self.fc1(stacked_hrt)
        preds = torch.squeeze(preds, dim=-1)
        return preds