#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy


class Rescal(ModelMeta):
    """`A Three-Way Model for Collective Learning on Multi-Relational Data`_

        RESCAL is a tensor factorization approach to knowledge representation learning,
        which is able to perform collective learning via the latent components of the factorization.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.core.Rescal import Rescal
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = Rescal()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on mnick_ and `OpenKE_Rescal`_.

         .. _mnick: https://github.com/mnick/rescal.py/blob/master/rescal/rescal.py

         .. _OpenKE_Rescal: https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py

         .. _A Three-Way Model for Collective Learning on Multi-Relational Data : http://www.icml-2011.org/papers/438_icmlpaper.pdf
    """

    def __init__(self, config):
        super(Rescal, self).__init__()
        self.config = config
        self.model_name = 'Rescal'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_matrices = nn.Embedding(num_total_rel, k * k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_matrices.weight)

        self.parameter_list = [self.ent_embeddings, self.rel_matrices]

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        k = self.config.hidden_size

        self.ent_embeddings.weight.data = self.get_normalized_data(self.ent_embeddings, self.config.kg_meta.tot_entity, dim=-1)
        self.rel_matrices.weight.data = self.get_normalized_data(self.rel_matrices, self.config.kg_meta.tot_relation, dim=-1)

        emb_h = self.ent_embeddings(h)
        emb_r = self.rel_matrices(r)
        emb_t = self.ent_embeddings(t)
        emb_h = emb_h.view(-1, k, 1)
        emb_r = emb_r.view(-1, k, k)
        emb_t = emb_t.view(-1, k, 1)

        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        # dim of h: [m, k, 1]
        #        r: [m, k, k]
        #        t: [m, k, 1]
        return -torch.sum(h_e * torch.matmul(r_e, t_e), [1, 2])

    def get_normalized_data(self, embedding, num_embeddings, p=2, dim=1):
        norms = torch.norm(embedding.weight, p, dim).data
        return embedding.weight.data.div(norms.view(num_embeddings, 1).expand_as(embedding.weight))
