#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class RotatE(ModelMeta):
    """ `Rotate-Knowledge graph embedding by relation rotation in complex space`_

        RotatE models the entities and the relations in the complex vector space.
        The translational relation in RotatE is defined as the element-wise 2D
        rotation in which the head entity h will be rotated to the tail entity t by
        multiplying the unit-length relation r in complex number form.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.RotatE import RotatE
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = RotatE()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _Rotate-Knowledge graph embedding by relation rotation in complex space:
            https://openreview.net/pdf?id=HkgEQnRqYQ
    """

    def __init__(self, config):
        super(RotatE, self).__init__()
        self.config = config
        self.model_name = 'RotatE'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation

        k = self.config.hidden_size
        self.embedding_range = (self.config.margin + 2.0) / k

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        self.ent_embeddings_imag = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        nn.init.uniform_(self.ent_embeddings.weight, -self.embedding_range, self.embedding_range)
        nn.init.uniform_(self.ent_embeddings_imag.weight, -self.embedding_range, self.embedding_range)
        nn.init.uniform_(self.rel_embeddings.weight, -self.embedding_range, self.embedding_range)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embeddings_real"),
            NamedEmbedding(self.ent_embeddings_imag, "ent_embeddings_imag"),
            NamedEmbedding(self.rel_embeddings, "rel_embeddings_real"),
        ]

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns real and imaginary values of head, relation and tail embedding.
        """
        pi = 3.14159265358979323846
        h_e_r = self.ent_embeddings(h)
        h_e_i = self.ent_embeddings_imag(h)
        r_e_r = self.rel_embeddings(r)
        t_e_r = self.ent_embeddings(t)
        t_e_i = self.ent_embeddings_imag(t)
        r_e_r = r_e_r / (self.embedding_range / pi)
        r_e_i = torch.sin(r_e_r)
        r_e_r = torch.cos(r_e_r)
        return h_e_r, h_e_i, r_e_r, r_e_i, t_e_r, t_e_i
   
    def forward(self, h, r, t):
        h_e_r, h_e_i, r_e_r, r_e_i, t_e_r, t_e_i = self.embed(h, r, t)
        score_r = h_e_r * r_e_r - h_e_i * r_e_i - t_e_r
        score_i = h_e_r * r_e_i + h_e_i * r_e_r - t_e_i
        return -(self.config.margin - torch.sum(score_r**2 + score_i**2, axis=-1))