#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class TransH(ModelMeta):
    """ `Knowledge Graph Embedding by Translating on Hyperplanes`_

        TransH models a relation as a hyperplane together with a translation operation on it.
        By doing this, it aims to preserve the mapping properties of relations such as reflexive,
        one-to-many, many-to-one, and many-to-many with almost the same model complexity of TransE.

         Args:
            config (object): Model configuration parameters.

         Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.

         Examples:
            >>> from pykg2vec.core.TransH import TransH
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransH()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

         Portion of the code based on `OpenKE_TransH`_ and `thunlp_TransH`_.

         .. _OpenKE_TransH:
             https://github.com/thunlp/OpenKE/blob/master/models/TransH.py

         .. _thunlp_TransH:
             https://github.com/thunlp/TensorFlow-TransX/blob/master/transH.py

         .. _Knowledge Graph Embedding by Translating on Hyperplanes:
             https://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf
    """

    def __init__(self, config):
        super(TransH, self).__init__()
        self.config = config
        self.model_name = 'TransH'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation

        self.ent_embeddings = nn.Embedding(num_total_ent, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(num_total_rel, self.config.hidden_size)
        self.w = nn.Embedding(num_total_rel, self.config.hidden_size)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.w.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.w, "w"),
        ]

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)

        norm_h_e = F.normalize(h_e, p=2, dim=-1)
        norm_r_e = F.normalize(r_e, p=2, dim=-1)
        norm_t_e = F.normalize(t_e, p=2, dim=-1)

        if self.config.L1_flag:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=1, dim=-1)
        else:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=2, dim=-1)
    
    def projection(self, emb_e, proj_vec):
        """Calculates the projection of entities"""
        proj_vec = F.normalize(proj_vec, p=2, dim=-1)

        # [b, k], [b, k]
        return emb_e - torch.sum(emb_e * proj_vec, dim=-1, keepdims=True) * proj_vec

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
        proj_vec = self.w(r)

        emb_h = self.projection(emb_h, proj_vec)
        emb_t = self.projection(emb_t, proj_vec)

        return emb_h, emb_r, emb_t