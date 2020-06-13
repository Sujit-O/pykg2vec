#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class Complex(ModelMeta):
    """`Complex Embeddings for Simple Link Prediction`_.

    ComplEx is an enhanced version of DistMult in that it uses complex-valued embeddings
    to represent both entities and relations. Using the complex-valued embedding allows
    the defined scoring function in ComplEx to differentiate that facts with assymmetric relations.
    
    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
        model (str): Name of the model.
    
    Examples:
        >>> from pykg2vec.core.Complex import Complex
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = Complex()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf
    """

    def __init__(self, config):
        super(Complex, self).__init__()
        self.config = config
        self.model_name = 'Complex'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_embeddings_real = nn.Embedding(num_total_ent, k)
        self.ent_embeddings_img  = nn.Embedding(num_total_ent, k)
        self.rel_embeddings_real = nn.Embedding(num_total_rel, k)
        self.rel_embeddings_img  = nn.Embedding(num_total_rel, k)
        nn.init.xavier_uniform_(self.ent_embeddings_real.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_img.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_real.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_img.weight)

        self.parameter_list = [self.ent_embeddings_real, self.ent_embeddings_img, self.rel_embeddings_real, self.rel_embeddings_img]

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings_real, "emb_e_real"),
            NamedEmbedding(self.ent_embeddings_img, "emb_e_img"),
            NamedEmbedding(self.rel_embeddings_real, "emb_rel_real"),
            NamedEmbedding(self.rel_embeddings_img, "emb_rel_img"),
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
        h_emb_real = self.ent_embeddings_real(h)
        h_emb_img  = self.ent_embeddings_img(h)

        r_emb_real = self.rel_embeddings_real(r)
        r_emb_img  = self.rel_embeddings_img(r)

        t_emb_real = self.ent_embeddings_real(t)
        t_emb_img  = self.ent_embeddings_img(t)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img

    def forward(self, h, r, t):
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)
        return -torch.sum(h_e_real * t_e_real * r_e_real + h_e_img * t_e_img * r_e_real
                          + h_e_real * t_e_img * r_e_img - h_e_img * t_e_real * r_e_img, -1)

    def get_reg(self, h, r, t):
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)
        regul_term = torch.mean(torch.sum(h_e_real**2, -1) + torch.sum(h_e_img**2, -1) + torch.sum(r_e_real**2,-1)
                                + torch.sum(r_e_img**2, -1) + torch.sum(t_e_real**2, -1) + torch.sum(t_e_img**2, -1))
        return self.config.lmbda*regul_term

class ComplexN3(Complex):
    """`Complex Embeddings for Simple Link Prediction`_.

    ComplEx is an enhanced version of DistMult in that it uses complex-valued embeddings
    to represent both entities and relations. Using the complex-valued embedding allows
    the defined scoring function in ComplEx to differentiate that facts with assymmetric relations.
    
    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
        model (str): Name of the model.
    
    Examples:
        >>> from pykg2vec.core.Complex import Complex
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = Complex()
        >>> trainer = Trainer(model=model, debug=False)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf
    """

    def __init__(self, config):
        super(ComplexN3, self).__init__(config)
        self.model_name = 'ComplexN3'

    def get_reg(self, h, r, t):
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)
        regul_term = torch.mean(torch.sum(h_e_real.abs()**3, -1) + torch.sum(h_e_img.abs()**3, -1)
                              + torch.sum(r_e_real.abs()**3, -1) + torch.sum(r_e_img.abs()**3, -1)
                              + torch.sum(t_e_real.abs()**3, -1) + torch.sum(t_e_img.abs()**3, -1))
        return self.config.lmbda*regul_term