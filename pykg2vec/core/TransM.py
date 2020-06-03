#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy


class TransM(ModelMeta):
    """ `Transition-based Knowledge Graph Embedding with Relational Mapping Properties`_

        TransM is another line of research that improves TransE by relaxing the overstrict requirement of
        h+r ==> t. TransM associates each fact (h, r, t) with a weight theta(r) specific to the relation.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.core.TransM import TransM
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransM()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _Transition-based Knowledge Graph Embedding with Relational Mapping Properties:
            https://pdfs.semanticscholar.org/0ddd/f37145689e5f2899f8081d9971882e6ff1e9.pdf
    """

    def __init__(self, config):
        super(TransM, self).__init__()
        self.config = config
        self.model_name = 'TransM'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)

        rel_head = {x: [] for x in range(num_total_rel)}
        rel_tail = {x: [] for x in range(num_total_rel)}
        rel_counts = {x: 0 for x in range(num_total_rel)}
        train_triples_ids = self.config.knowledge_graph.read_cache_data('triplets_train')
        for t in train_triples_ids:
            rel_head[t.r].append(t.h)
            rel_tail[t.r].append(t.t)
            rel_counts[t.r] += 1

        theta = [1/np.log(2+rel_counts[x]/(1+len(rel_tail[x])) + rel_counts[x]/(1+len(rel_head[x]))) for x in range(num_total_rel)]
        self.theta = torch.from_numpy(np.asarray(theta, dtype=np.float32))
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.theta]

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

        r_theta = self.theta[r]

        if self.config.L1_flag:
            return r_theta*torch.norm(norm_h_e + norm_r_e - norm_t_e, p=1, dim=-1)
        else:
            return r_theta*torch.norm(norm_h_e + norm_r_e - norm_t_e, p=2, dim=-1)

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