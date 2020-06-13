#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class ANALOGY(ModelMeta):

    def __init__(self, config):
        super(ANALOGY, self).__init__()
        
        self.config = config
        self.model_name = 'ANALOGY'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.ent_embeddings_real = nn.Embedding(num_total_ent, k // 2)
        self.ent_embeddings_img  = nn.Embedding(num_total_ent, k // 2)
        self.rel_embeddings_real = nn.Embedding(num_total_rel, k // 2)
        self.rel_embeddings_img  = nn.Embedding(num_total_rel, k // 2)

        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_real.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_img.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_real.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_img.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
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
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        h_emb = self.ent_embeddings(h)
        r_emb = self.rel_embeddings(r)
        t_emb = self.ent_embeddings(t)

        return h_emb, r_emb, t_emb

    def embed_complex(self, h, r, t):
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
        h_e, r_e, t_e = self.embed(h, r, t)
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed_complex(h, r, t)

        complex_loss = -(h_e_real * t_e_real * r_e_real + h_e_img * t_e_img * r_e_real + h_e_real * t_e_img * r_e_img - h_e_img * t_e_real * r_e_img).sum(axis=-1)
        distmult_loss = -(h_e * r_e * t_e).sum(axis=-1)

        return complex_loss + distmult_loss

    def get_reg(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed_complex(h, r, t)

        regul_term = (h_e_real**2+h_e_img**2+r_e_real**2+r_e_img**2+t_e_real**2+t_e_img**2).sum(axis=-1).mean()
        regul_term += (h_e**2+r_e**2+t_e**2).sum(axis=-1).mean()
        return self.config.lmbda*regul_term
