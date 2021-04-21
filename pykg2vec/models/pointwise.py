#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from numpy.random import RandomState

from pykg2vec.models.KGMeta import PointwiseModel
from pykg2vec.models.Domain import NamedEmbedding
from pykg2vec.utils.criterion import Criterion


class ANALOGY(PointwiseModel):
    """
       `Analogical Inference for Multi-relational Embeddings`_

       Args:
           config (object): Model configuration parameters.

       .. _Analogical Inference for Multi-relational Embeddings:
           http://proceedings.mlr.press/v70/liu17d/liu17d.pdf

    """

    def __init__(self, **kwargs):
        super(ANALOGY, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        k = self.hidden_size

        self.ent_embeddings = NamedEmbedding("ent_embedding", self.tot_entity, k)
        self.rel_embeddings = NamedEmbedding("rel_embedding", self.tot_relation, k)
        self.ent_embeddings_real = NamedEmbedding("emb_e_real", self.tot_entity, k // 2)
        self.ent_embeddings_img = NamedEmbedding("emb_e_img", self.tot_entity, k // 2)
        self.rel_embeddings_real = NamedEmbedding("emb_rel_real", self.tot_relation, k // 2)
        self.rel_embeddings_img = NamedEmbedding("emb_rel_img", self.tot_relation, k // 2)

        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_real.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_img.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_real.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_img.weight)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
            self.ent_embeddings_real,
            self.ent_embeddings_img,
            self.rel_embeddings_real,
            self.rel_embeddings_img,
        ]

        self.loss = Criterion.pointwise_logistic

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
        h_emb_img = self.ent_embeddings_img(h)

        r_emb_real = self.rel_embeddings_real(r)
        r_emb_img = self.rel_embeddings_img(r)

        t_emb_real = self.ent_embeddings_real(t)
        t_emb_img = self.ent_embeddings_img(t)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed_complex(h, r, t)

        complex_loss = -(h_e_real * t_e_real * r_e_real + h_e_img * t_e_img * r_e_real + h_e_real * t_e_img * r_e_img - h_e_img * t_e_real * r_e_img).sum(axis=-1)
        distmult_loss = -(h_e * r_e * t_e).sum(axis=-1)

        return complex_loss + distmult_loss

    def get_reg(self, h, r, t, reg_type="F2"):
        h_e, r_e, t_e = self.embed(h, r, t)
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed_complex(h, r, t)

        if reg_type.lower() == 'f2':
            regul_term = (h_e_real ** 2 + h_e_img ** 2 + r_e_real ** 2 + r_e_img ** 2 + t_e_real ** 2 + t_e_img ** 2).sum(axis=-1).mean()
            regul_term += (h_e ** 2 + r_e ** 2 + t_e ** 2).sum(axis=-1).mean()
        elif reg_type.lower() == 'n3':
            regul_term = (h_e_real ** 3 + h_e_img ** 3 + r_e_real ** 3 + r_e_img ** 3 + t_e_real ** 3 + t_e_img ** 3).sum(axis=-1).mean()
            regul_term += (h_e ** 3 + r_e ** 3 + t_e ** 3).sum(axis=-1).mean()
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % reg_type)

        return self.lmbda*regul_term


class Complex(PointwiseModel):
    """
        `Complex Embeddings for Simple Link Prediction`_ (ComplEx) is an enhanced version of DistMult in that it uses complex-valued embeddings
        to represent both entities and relations. Using the complex-valued embedding allows
        the defined scoring function in ComplEx to differentiate that facts with assymmetric relations.

        Args:
            config (object): Model configuration parameters.

        .. _Complex Embeddings for Simple Link Prediction:
            http://proceedings.mlr.press/v48/trouillon16.pdf

    """
    def __init__(self, **kwargs):
        super(Complex, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size

        self.ent_embeddings_real = NamedEmbedding("emb_e_real", num_total_ent, k)
        self.ent_embeddings_img = NamedEmbedding("emb_e_img", num_total_ent, k)
        self.rel_embeddings_real = NamedEmbedding("emb_rel_real", num_total_rel, k)
        self.rel_embeddings_img = NamedEmbedding("emb_rel_img", num_total_rel, k)
        nn.init.xavier_uniform_(self.ent_embeddings_real.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_img.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_real.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_img.weight)

        self.parameter_list = [
            self.ent_embeddings_real,
            self.ent_embeddings_img,
            self.rel_embeddings_real,
            self.rel_embeddings_img,
        ]

        self.loss = Criterion.pointwise_logistic

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
        h_emb_img = self.ent_embeddings_img(h)

        r_emb_real = self.rel_embeddings_real(r)
        r_emb_img = self.rel_embeddings_img(r)

        t_emb_real = self.ent_embeddings_real(t)
        t_emb_img = self.ent_embeddings_img(t)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img

    def forward(self, h, r, t):
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)
        return -torch.sum(h_e_real * t_e_real * r_e_real + h_e_img * t_e_img * r_e_real +
                          h_e_real * t_e_img * r_e_img - h_e_img * t_e_real * r_e_img, -1)

    def get_reg(self, h, r, t, reg_type="F2"):
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)

        if reg_type.lower() == 'f2':
            regul_term = torch.mean(torch.sum(h_e_real ** 2, -1) + torch.sum(h_e_img ** 2, -1) + torch.sum(r_e_real ** 2, -1) +
                                    torch.sum(r_e_img ** 2, -1) + torch.sum(t_e_real ** 2, -1) + torch.sum(t_e_img ** 2, -1))
        elif reg_type.lower() == 'n3':
            regul_term = torch.mean(torch.sum(h_e_real ** 3, -1) + torch.sum(h_e_img ** 3, -1) + torch.sum(r_e_real ** 3, -1) +
                                    torch.sum(r_e_img ** 3, -1) + torch.sum(t_e_real ** 3, -1) + torch.sum(t_e_img ** 3, -1))
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % reg_type)

        return self.lmbda*regul_term


class ComplexN3(Complex):
    """
        `Complex Embeddings for Simple Link Prediction`_ (ComplEx) is an enhanced version of DistMult in that it uses complex-valued embeddings
        to represent both entities and relations. Using the complex-valued embedding allows
        the defined scoring function in ComplEx to differentiate that facts with assymmetric relations.

        Args:
            config (object): Model configuration parameters.

        .. _Complex Embeddings for Simple Link Prediction:
            http://proceedings.mlr.press/v48/trouillon16.pdf

    """

    def __init__(self, **kwargs):
        super(ComplexN3, self).__init__(**kwargs)
        self.model_name = 'complexn3'
        self.loss = Criterion.pointwise_logistic

    def get_reg(self, h, r, t, reg_type="N3"):
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)

        if reg_type.lower() == 'f2':
            regul_term = torch.mean(torch.sum(h_e_real.abs() ** 2, -1) + torch.sum(h_e_img.abs() ** 2, -1) +
                                    torch.sum(r_e_real.abs() ** 2, -1) + torch.sum(r_e_img.abs() ** 2, -1) +
                                    torch.sum(t_e_real.abs() ** 2, -1) + torch.sum(t_e_img.abs() ** 2, -1))
        elif reg_type.lower() == 'n3':
            regul_term = torch.mean(torch.sum(h_e_real.abs() ** 3, -1) + torch.sum(h_e_img.abs() ** 3, -1) +
                                    torch.sum(r_e_real.abs() ** 3, -1) + torch.sum(r_e_img.abs() ** 3, -1) +
                                    torch.sum(t_e_real.abs() ** 3, -1) + torch.sum(t_e_img.abs() ** 3, -1))
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % reg_type)

        return self.lmbda*regul_term


class ConvKB(PointwiseModel):
    """
        In `A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network`_ (ConvKB),
        each triple (head entity, relation, tail entity) is represented as a 3-column matrix where each column vector represents a triple element

        Portion of the code based on daiquocnguyen_.

        Args:
            config (object): Model configuration parameters.

        .. _daiquocnguyen:
            https://github.com/daiquocnguyen/ConvKB

        .. _A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network:
            https://www.aclweb.org/anthology/N18-2053
    """
    def __init__(self, **kwargs):
        super(ConvKB, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "num_filters", "filter_sizes"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size
        num_filters = self.num_filters
        filter_sizes = self.filter_sizes
        device = kwargs["device"]

        self.ent_embeddings = NamedEmbedding("ent_embedding", num_total_ent, k)
        self.rel_embeddings = NamedEmbedding("rel_embedding", num_total_rel, k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
        ]

        self.conv_list = [nn.Conv2d(1, num_filters, (3, filter_size), stride=(1, 1)).to(device) for filter_size in filter_sizes]
        conv_out_dim = num_filters*sum([(k-filter_size+1) for filter_size in filter_sizes])
        self.fc1 = nn.Linear(in_features=conv_out_dim, out_features=1, bias=True)

        self.loss = Criterion.pointwise_logistic

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


class CP(PointwiseModel):
    """
           `Canonical Tensor Decomposition for Knowledge Base Completion`_

           Args:
               config (object): Model configuration parameters.

           .. _Canonical Tensor Decomposition for Knowledge Base Completion:
               http://proceedings.mlr.press/v80/lacroix18a/lacroix18a.pdf

    """
    def __init__(self, **kwargs):
        super(CP, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size

        self.sub_embeddings = NamedEmbedding("sub_embedding", num_total_ent, k)
        self.rel_embeddings = NamedEmbedding("rel_embedding", num_total_rel, k)
        self.obj_embeddings = NamedEmbedding("obj_embedding", num_total_ent, k)

        nn.init.xavier_uniform_(self.sub_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.obj_embeddings.weight)

        self.parameter_list = [
            self.sub_embeddings,
            self.rel_embeddings,
            self.obj_embeddings,
        ]

        self.loss = Criterion.pointwise_logistic

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = self.sub_embeddings(h)
        emb_r = self.rel_embeddings(r)
        emb_t = self.obj_embeddings(t)
        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        return -torch.sum(h_e * r_e * t_e, -1)

    def get_reg(self, h, r, t, reg_type='N3'):
        h_e, r_e, t_e = self.embed(h, r, t)

        if reg_type.lower() == 'f2':
            regul_term = torch.mean(torch.sum(h_e**2, -1) + torch.sum(r_e**2, -1) + torch.sum(t_e**2, -1))
        elif reg_type.lower() == 'n3':
            regul_term = torch.mean(torch.sum(h_e**3, -1) + torch.sum(r_e**3, -1) + torch.sum(t_e**3, -1))
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % reg_type)

        return self.lmbda * regul_term


class DistMult(PointwiseModel):
    """
        `EMBEDDING ENTITIES AND RELATIONS FOR LEARNING AND INFERENCE IN KNOWLEDGE BASES`_ (DistMult) is a simpler model comparing with RESCAL in that it simplifies
        the weight matrix used in RESCAL to a diagonal matrix. The scoring
        function used DistMult can capture the pairwise interactions between
        the head and the tail entities. However, DistMult has limitation on modeling asymmetric relations.

        Args:
            config (object): Model configuration parameters.

        .. _EMBEDDING ENTITIES AND RELATIONS FOR LEARNING AND INFERENCE IN KNOWLEDGE BASES:
            https://arxiv.org/pdf/1412.6575.pdf

    """
    def __init__(self, **kwargs):
        super(DistMult, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size

        self.ent_embeddings = NamedEmbedding("ent_embedding", num_total_ent, k)
        self.rel_embeddings = NamedEmbedding("rel_embedding", num_total_rel, k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
        ]

        self.loss = Criterion.pointwise_logistic

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

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        return -torch.sum(h_e*r_e*t_e, -1)

    def get_reg(self, h, r, t, reg_type="F2"):
        h_e, r_e, t_e = self.embed(h, r, t)

        if reg_type.lower() == 'f2':
            regul_term = torch.mean(torch.sum(h_e ** 2, -1) + torch.sum(r_e ** 2, -1) + torch.sum(t_e ** 2, -1))
        elif reg_type.lower() == 'n3':
            regul_term = torch.mean(torch.sum(h_e ** 3, -1) + torch.sum(r_e ** 3, -1) + torch.sum(t_e ** 3, -1))
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % reg_type)

        return self.lmbda*regul_term


class SimplE(PointwiseModel):
    """
           `SimplE Embedding for Link Prediction in Knowledge Graphs`_

           Args:
               config (object): Model configuration parameters.

           .. _SimplE Embedding for Link Prediction in Knowledge Graphs:
               https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf

    """
    def __init__(self, **kwargs):
        super(SimplE, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size
        self.tot_train_triples = kwargs['tot_train_triples']
        self.batch_size = kwargs['batch_size']

        self.ent_head_embeddings = NamedEmbedding("ent_head_embedding", num_total_ent, k)
        self.ent_tail_embeddings = NamedEmbedding("ent_tail_embedding", num_total_ent, k)
        self.rel_embeddings = NamedEmbedding("rel_embedding", num_total_rel, k)
        self.rel_inv_embeddings = NamedEmbedding("rel_inv_embedding", num_total_rel, k)

        nn.init.xavier_uniform_(self.ent_head_embeddings.weight)
        nn.init.xavier_uniform_(self.ent_tail_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight)

        self.parameter_list = [
            self.ent_head_embeddings,
            self.ent_tail_embeddings,
            self.rel_embeddings,
            self.rel_inv_embeddings,
        ]

        self.loss = Criterion.pointwise_logistic

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h1 = self.ent_head_embeddings(h)
        emb_h2 = self.ent_head_embeddings(t)
        emb_r1 = self.rel_embeddings(r)
        emb_r2 = self.rel_inv_embeddings(r)
        emb_t1 = self.ent_tail_embeddings(t)
        emb_t2 = self.ent_tail_embeddings(h)
        return emb_h1, emb_h2, emb_r1, emb_r2, emb_t1, emb_t2

    def forward(self, h, r, t):
        h1_e, h2_e, r1_e, r2_e, t1_e, t2_e = self.embed(h, r, t)

        init = torch.sum(h1_e*r1_e*t1_e, 1) + torch.sum(h2_e*r2_e*t2_e, 1) / 2.0
        return -torch.clamp(init, -20, 20)

    def get_reg(self, h, r, t, reg_type="F2"):
        if reg_type.lower() == 'f2':
            regul_term = torch.mean(torch.sum(h.type(torch.FloatTensor) ** 2, -1) + torch.sum(r.type(torch.FloatTensor) ** 2, -1) + torch.sum(t.type(torch.FloatTensor) ** 2, -1))
        elif reg_type.lower() == 'n3':
            regul_term = torch.mean(torch.sum(h.type(torch.FloatTensor) ** 3, -1) + torch.sum(r.type(torch.FloatTensor) ** 3, -1) + torch.sum(t.type(torch.FloatTensor) ** 3, -1))
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % reg_type)

        return self.lmbda * regul_term


class SimplE_ignr(SimplE):
    """
           `SimplE Embedding for Link Prediction in Knowledge Graphs`_

           Args:
               config (object): Model configuration parameters.

           .. _SimplE Embedding for Link Prediction in Knowledge Graphs:
               https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf

    """

    def __init__(self, **kwargs):
        super(SimplE_ignr, self).__init__(**kwargs)
        self.model_name = 'simple_ignr'
        self.loss = Criterion.pointwise_logistic

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h = self._concat_selected_embeddings(self.ent_head_embeddings, h, self.ent_head_embeddings, t)
        emb_r = self._concat_selected_embeddings(self.rel_embeddings, r, self.rel_inv_embeddings, r)
        emb_t = self._concat_selected_embeddings(self.ent_tail_embeddings, t, self.ent_tail_embeddings, h)

        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)

        init = torch.sum(h_e*r_e*t_e, 1)
        return -torch.clamp(init, -20, 20)

    @staticmethod
    def _concat_selected_embeddings(e1, t1, e2, t2):
        return torch.cat([torch.index_select(e1.weight, 0, t1), torch.index_select(e2.weight, 0, t2)], 1)


class QuatE(PointwiseModel):
    """
        `Quaternion Knowledge Graph Embeddings`_

        Args:
            config (object): Model configuration parameters.

        .. _cheungdaven: https://github.com/cheungdaven/QuatE.git

        .. _Quaternion Knowledge Graph Embeddings:
            https://arxiv.org/abs/1904.10281

    """

    def __init__(self, **kwargs):
        super(QuatE, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size

        self.ent_s_embedding = NamedEmbedding("ent_s_embedding", num_total_ent, k)
        self.ent_x_embedding = NamedEmbedding("ent_x_embedding", num_total_ent, k)
        self.ent_y_embedding = NamedEmbedding("ent_y_embedding", num_total_ent, k)
        self.ent_z_embedding = NamedEmbedding("ent_z_embedding", num_total_ent, k)
        self.rel_s_embedding = NamedEmbedding("rel_s_embedding", num_total_rel, k)
        self.rel_x_embedding = NamedEmbedding("rel_x_embedding", num_total_rel, k)
        self.rel_y_embedding = NamedEmbedding("rel_y_embedding", num_total_rel, k)
        self.rel_z_embedding = NamedEmbedding("rel_z_embedding", num_total_rel, k)
        self.rel_w_embedding = NamedEmbedding("rel_w_embedding", num_total_rel, k)
        self.fc = nn.Linear(100, 50, bias=False)
        self.ent_dropout = nn.Dropout(0)
        self.rel_dropout = nn.Dropout(0)
        self.bn = nn.BatchNorm1d(k)

        r, i, j, k = QuatE._quaternion_init(self.tot_entity, self.hidden_size)
        r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
        self.ent_s_embedding.weight.data = r.type_as(self.ent_s_embedding.weight.data)
        self.ent_x_embedding.weight.data = i.type_as(self.ent_x_embedding.weight.data)
        self.ent_y_embedding.weight.data = j.type_as(self.ent_y_embedding.weight.data)
        self.ent_z_embedding.weight.data = k.type_as(self.ent_z_embedding.weight.data)

        s, x, y, z = QuatE._quaternion_init(self.tot_entity, self.hidden_size)
        s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
        self.rel_s_embedding.weight.data = s.type_as(self.rel_s_embedding.weight.data)
        self.rel_x_embedding.weight.data = x.type_as(self.rel_x_embedding.weight.data)
        self.rel_y_embedding.weight.data = y.type_as(self.rel_y_embedding.weight.data)
        self.rel_z_embedding.weight.data = z.type_as(self.rel_z_embedding.weight.data)

        nn.init.xavier_uniform_(self.ent_s_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_x_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_y_embedding.weight.data)
        nn.init.xavier_uniform_(self.ent_z_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_s_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_x_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_y_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_z_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_w_embedding.weight.data)

        self.parameter_list = [
            self.ent_s_embedding,
            self.ent_x_embedding,
            self.ent_y_embedding,
            self.ent_z_embedding,
            self.rel_s_embedding,
            self.rel_x_embedding,
            self.rel_y_embedding,
            self.rel_z_embedding,
            self.rel_w_embedding,
        ]

        self.loss = Criterion.pointwise_logistic

    def embed(self, h, r, t):
        s_emb_h = self.ent_s_embedding(h)
        x_emb_h = self.ent_x_embedding(h)
        y_emb_h = self.ent_y_embedding(h)
        z_emb_h = self.ent_z_embedding(h)

        s_emb_t = self.ent_s_embedding(t)
        x_emb_t = self.ent_x_embedding(t)
        y_emb_t = self.ent_y_embedding(t)
        z_emb_t = self.ent_z_embedding(t)

        s_emb_r = self.rel_s_embedding(r)
        x_emb_r = self.rel_x_embedding(r)
        y_emb_r = self.rel_y_embedding(r)
        z_emb_r = self.rel_z_embedding(r)

        return s_emb_h, x_emb_h, y_emb_h, z_emb_h, s_emb_t, x_emb_t, y_emb_t, z_emb_t, s_emb_r, x_emb_r, y_emb_r, z_emb_r

    def forward(self, h, r, t):
        s_emb_h, x_emb_h, y_emb_h, z_emb_h, s_emb_t, x_emb_t, y_emb_t, z_emb_t, s_emb_r, x_emb_r, y_emb_r, z_emb_r = self.embed(h, r, t)

        denominator_b = torch.sqrt(s_emb_r ** 2 + x_emb_r ** 2 + y_emb_r ** 2 + z_emb_r ** 2)
        s_emb_r = s_emb_r / denominator_b
        x_emb_r = x_emb_r / denominator_b
        y_emb_r = y_emb_r / denominator_b
        z_emb_r = z_emb_r / denominator_b

        a = s_emb_h * s_emb_r - x_emb_h * x_emb_r - y_emb_h * y_emb_r - z_emb_h * z_emb_r
        b = s_emb_h * x_emb_r + s_emb_r * x_emb_h + y_emb_h * z_emb_r - y_emb_r * z_emb_h
        c = s_emb_h * y_emb_r + s_emb_r * y_emb_h + z_emb_h * x_emb_r - z_emb_r * x_emb_h
        d = s_emb_h * z_emb_r + s_emb_r * z_emb_h + x_emb_h * y_emb_r - x_emb_r * y_emb_h

        score_r = (a * s_emb_t + b * x_emb_t + c * y_emb_t + d * z_emb_t)

        return -torch.sum(score_r, -1)

    def get_reg(self, h, r, t, reg_type='N3'):
        s_emb_h, x_emb_h, y_emb_h, z_emb_h, s_emb_t, x_emb_t, y_emb_t, z_emb_t, s_emb_r, x_emb_r, y_emb_r, z_emb_r = self.embed(h, r, t)
        if reg_type.lower() == 'f2':
            regul = (torch.mean(torch.abs(s_emb_h) ** 2)
                     + torch.mean(torch.abs(x_emb_h) ** 2)
                     + torch.mean(torch.abs(y_emb_h) ** 2)
                     + torch.mean(torch.abs(z_emb_h) ** 2)
                     + torch.mean(torch.abs(s_emb_t) ** 2)
                     + torch.mean(torch.abs(x_emb_t) ** 2)
                     + torch.mean(torch.abs(y_emb_t) ** 2)
                     + torch.mean(torch.abs(z_emb_t) ** 2)
                     )
            regul2 = (torch.mean(torch.abs(s_emb_r) ** 2)
                      + torch.mean(torch.abs(x_emb_r) ** 2)
                      + torch.mean(torch.abs(y_emb_r) ** 2)
                      + torch.mean(torch.abs(z_emb_r) ** 2))
        elif reg_type.lower() == 'n3':
            regul = (torch.mean(torch.abs(s_emb_h) ** 3)
                     + torch.mean(torch.abs(x_emb_h) ** 3)
                     + torch.mean(torch.abs(y_emb_h) ** 3)
                     + torch.mean(torch.abs(z_emb_h) ** 3)
                     + torch.mean(torch.abs(s_emb_t) ** 3)
                     + torch.mean(torch.abs(x_emb_t) ** 3)
                     + torch.mean(torch.abs(y_emb_t) ** 3)
                     + torch.mean(torch.abs(z_emb_t) ** 3)
                     )
            regul2 = (torch.mean(torch.abs(s_emb_r) ** 3)
                      + torch.mean(torch.abs(x_emb_r) ** 3)
                      + torch.mean(torch.abs(y_emb_r) ** 3)
                      + torch.mean(torch.abs(z_emb_r) ** 3))
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % reg_type)

        return self.lmbda * (regul + regul2)

    @staticmethod
    def _quaternion_init(in_features, out_features, criterion='he'):

        fan_in = in_features
        fan_out = out_features

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        rng = RandomState(123)

        kernel_shape = (in_features, out_features)

        number_of_weights = np.prod(kernel_shape)
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)

        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape)
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

        weight_r = modulus * np.cos(phase)
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)

        return weight_r, weight_i, weight_j, weight_k


class OctonionE(PointwiseModel):
    """
        `Quaternion Knowledge Graph Embeddings`_

        Args:
            config (object): Model configuration parameters.

        .. _cheungdaven: https://github.com/cheungdaven/QuatE.git

        .. _Quaternion Knowledge Graph Embeddings:
            https://arxiv.org/abs/1904.10281

    """

    def __init__(self, **kwargs):
        super(OctonionE, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size

        self.ent_embedding_1 = NamedEmbedding("ent_embedding_1", num_total_ent, k)
        self.ent_embedding_2 = NamedEmbedding("ent_embedding_2", num_total_ent, k)
        self.ent_embedding_3 = NamedEmbedding("ent_embedding_3", num_total_ent, k)
        self.ent_embedding_4 = NamedEmbedding("ent_embedding_4", num_total_ent, k)
        self.ent_embedding_5 = NamedEmbedding("ent_embedding_5", num_total_ent, k)
        self.ent_embedding_6 = NamedEmbedding("ent_embedding_6", num_total_ent, k)
        self.ent_embedding_7 = NamedEmbedding("ent_embedding_7", num_total_ent, k)
        self.ent_embedding_8 = NamedEmbedding("ent_embedding_8", num_total_ent, k)
        self.rel_embedding_1 = NamedEmbedding("rel_embedding_1", num_total_rel, k)
        self.rel_embedding_2 = NamedEmbedding("rel_embedding_2", num_total_rel, k)
        self.rel_embedding_3 = NamedEmbedding("rel_embedding_3", num_total_rel, k)
        self.rel_embedding_4 = NamedEmbedding("rel_embedding_4", num_total_rel, k)
        self.rel_embedding_5 = NamedEmbedding("rel_embedding_5", num_total_rel, k)
        self.rel_embedding_6 = NamedEmbedding("rel_embedding_6", num_total_rel, k)
        self.rel_embedding_7 = NamedEmbedding("rel_embedding_7", num_total_rel, k)
        self.rel_embedding_8 = NamedEmbedding("rel_embedding_8", num_total_rel, k)
        self.rel_w_embedding = NamedEmbedding("rel_w_embedding", num_total_rel, k)

        nn.init.xavier_uniform_(self.ent_embedding_1.weight.data)
        nn.init.xavier_uniform_(self.ent_embedding_2.weight.data)
        nn.init.xavier_uniform_(self.ent_embedding_3.weight.data)
        nn.init.xavier_uniform_(self.ent_embedding_4.weight.data)
        nn.init.xavier_uniform_(self.ent_embedding_5.weight.data)
        nn.init.xavier_uniform_(self.ent_embedding_6.weight.data)
        nn.init.xavier_uniform_(self.ent_embedding_7.weight.data)
        nn.init.xavier_uniform_(self.ent_embedding_8.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding_1.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding_2.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding_3.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding_4.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding_5.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding_6.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding_7.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding_8.weight.data)
        nn.init.xavier_uniform_(self.rel_w_embedding.weight.data)

        self.parameter_list = [
            self.ent_embedding_1,
            self.ent_embedding_2,
            self.ent_embedding_3,
            self.ent_embedding_4,
            self.ent_embedding_5,
            self.ent_embedding_6,
            self.ent_embedding_7,
            self.ent_embedding_8,
            self.rel_embedding_1,
            self.rel_embedding_2,
            self.rel_embedding_3,
            self.rel_embedding_4,
            self.rel_embedding_5,
            self.rel_embedding_6,
            self.rel_embedding_7,
            self.rel_embedding_8,
            self.rel_w_embedding,
        ]

        self.loss = Criterion.pointwise_logistic

    def embed(self, h, r, t):
        e_1_h = self.ent_embedding_1(h)
        e_2_h = self.ent_embedding_2(h)
        e_3_h = self.ent_embedding_3(h)
        e_4_h = self.ent_embedding_4(h)
        e_5_h = self.ent_embedding_5(h)
        e_6_h = self.ent_embedding_6(h)
        e_7_h = self.ent_embedding_7(h)
        e_8_h = self.ent_embedding_8(h)

        e_1_t = self.ent_embedding_1(t)
        e_2_t = self.ent_embedding_2(t)
        e_3_t = self.ent_embedding_3(t)
        e_4_t = self.ent_embedding_4(t)
        e_5_t = self.ent_embedding_5(t)
        e_6_t = self.ent_embedding_6(t)
        e_7_t = self.ent_embedding_7(t)
        e_8_t = self.ent_embedding_8(t)

        r_1 = self.rel_embedding_1(r)
        r_2 = self.rel_embedding_2(r)
        r_3 = self.rel_embedding_3(r)
        r_4 = self.rel_embedding_4(r)
        r_5 = self.rel_embedding_5(r)
        r_6 = self.rel_embedding_6(r)
        r_7 = self.rel_embedding_7(r)
        r_8 = self.rel_embedding_8(r)

        return e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h, \
              e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t, \
              r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

    def forward(self, h, r, t):
        e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h, \
        e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t, \
        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self.embed(h, r, t)

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = OctonionE._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)

        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = OctonionE._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                                  r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)

        score_r = (o_1 * e_1_t + o_2 * e_2_t + o_3 * e_3_t + o_4 * e_4_t
                   + o_5 * e_5_t + o_6 * e_6_t + o_7 * e_7_t + o_8 * e_8_t)

        return -torch.sum(score_r, -1)

    def get_reg(self, h, r, t, reg_type='N3'):
        e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h, \
        e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t, \
        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self.embed(h, r, t)
        if reg_type.lower() == 'f2':
            regul = (torch.mean(torch.abs(e_1_h) ** 2)
                     + torch.mean(torch.abs(e_2_h) ** 2)
                     + torch.mean(torch.abs(e_3_h) ** 2)
                     + torch.mean(torch.abs(e_4_h) ** 2)
                     + torch.mean(torch.abs(e_5_h) ** 2)
                     + torch.mean(torch.abs(e_6_h) ** 2)
                     + torch.mean(torch.abs(e_7_h) ** 2)
                     + torch.mean(torch.abs(e_8_h) ** 2)
                     + torch.mean(torch.abs(e_1_t) ** 2)
                     + torch.mean(torch.abs(e_2_t) ** 2)
                     + torch.mean(torch.abs(e_3_t) ** 2)
                     + torch.mean(torch.abs(e_4_t) ** 2)
                     + torch.mean(torch.abs(e_5_t) ** 2)
                     + torch.mean(torch.abs(e_6_t) ** 2)
                     + torch.mean(torch.abs(e_7_t) ** 2)
                     + torch.mean(torch.abs(e_8_t) ** 2)
                     )
            regul2 = (torch.mean(torch.abs(r_1) ** 2)
                      + torch.mean(torch.abs(r_2) ** 2)
                      + torch.mean(torch.abs(r_3) ** 2)
                      + torch.mean(torch.abs(r_4) ** 2)
                      + torch.mean(torch.abs(r_5) ** 2)
                      + torch.mean(torch.abs(r_6) ** 2)
                      + torch.mean(torch.abs(r_7) ** 2)
                      + torch.mean(torch.abs(r_8) ** 2))
        elif reg_type.lower() == 'n3':
            regul = (torch.mean(torch.abs(e_1_h) ** 3)
                     + torch.mean(torch.abs(e_2_h) ** 3)
                     + torch.mean(torch.abs(e_3_h) ** 3)
                     + torch.mean(torch.abs(e_4_h) ** 3)
                     + torch.mean(torch.abs(e_5_h) ** 3)
                     + torch.mean(torch.abs(e_6_h) ** 3)
                     + torch.mean(torch.abs(e_7_h) ** 3)
                     + torch.mean(torch.abs(e_8_h) ** 3)
                     + torch.mean(torch.abs(e_1_t) ** 3)
                     + torch.mean(torch.abs(e_2_t) ** 3)
                     + torch.mean(torch.abs(e_3_t) ** 3)
                     + torch.mean(torch.abs(e_4_t) ** 3)
                     + torch.mean(torch.abs(e_5_t) ** 3)
                     + torch.mean(torch.abs(e_6_t) ** 3)
                     + torch.mean(torch.abs(e_7_t) ** 3)
                     + torch.mean(torch.abs(e_8_t) ** 3)
                     )
            regul2 = (torch.mean(torch.abs(r_1) ** 3)
                      + torch.mean(torch.abs(r_2) ** 3)
                      + torch.mean(torch.abs(r_3) ** 3)
                      + torch.mean(torch.abs(r_4) ** 3)
                      + torch.mean(torch.abs(r_5) ** 3)
                      + torch.mean(torch.abs(r_6) ** 3)
                      + torch.mean(torch.abs(r_7) ** 3)
                      + torch.mean(torch.abs(r_8) ** 3))
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % reg_type)

        return self.lmbda * (regul + regul2)

    @staticmethod
    def _qmult(s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        a = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        b = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        c = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        d = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        return a, b, c, d

    @staticmethod
    def _qstar(a, b, c, d):
        return a, -b, -c, -d

    @staticmethod
    def _omult(a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c_1, c_2, c_3, c_4, d_1, d_2, d_3, d_4):

        d_1_star, d_2_star, d_3_star, d_4_star = OctonionE._qstar(d_1, d_2, d_3, d_4)
        c_1_star, c_2_star, c_3_star, c_4_star = OctonionE._qstar(c_1, c_2, c_3, c_4)

        o_1, o_2, o_3, o_4 = OctonionE._qmult(a_1, a_2, a_3, a_4, c_1, c_2, c_3, c_4)
        o_1s, o_2s, o_3s, o_4s = OctonionE._qmult(d_1_star, d_2_star, d_3_star, d_4_star, b_1, b_2, b_3, b_4)

        o_5, o_6, o_7, o_8 = OctonionE._qmult(d_1, d_2, d_3, d_4, a_1, a_2, a_3, a_4)
        o_5s, o_6s, o_7s, o_8s = OctonionE._qmult(b_1, b_2, b_3, b_4, c_1_star, c_2_star, c_3_star, c_4_star)

        return o_1 - o_1s, o_2 - o_2s, o_3 - o_3s, o_4 - o_4s, \
                o_5 + o_5s, o_6 + o_6s, o_7 + o_7s, o_8 + o_8s

    @staticmethod
    def _onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8):
        denominator = torch.sqrt(r_1 ** 2 + r_2 ** 2 + r_3 ** 2 + r_4 ** 2
                                 + r_5 ** 2 + r_6 ** 2 + r_7 ** 2 + r_8 ** 2)
        r_1 = r_1 / denominator
        r_2 = r_2 / denominator
        r_3 = r_3 / denominator
        r_4 = r_4 / denominator
        r_5 = r_5 / denominator
        r_6 = r_6 / denominator
        r_7 = r_7 / denominator
        r_8 = r_8 / denominator

        return r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

class MuRP(PointwiseModel):
    """
       `Multi-relational Poincaré Graph Embeddings`_

       Args:
           config (object): Model configuration parameters.

       .. _Multi-relational Poincaré Graph Embeddings:
           https://arxiv.org/abs/1905.09791

    """

    def __init__(self, **kwargs):
        super(MuRP, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        k = self.hidden_size
        self.device = kwargs["device"]

        self.ent_embeddings = NamedEmbedding("ent_embedding", self.tot_entity, k, padding_idx=0)
        self.ent_embeddings.weight.data = (
                    1e-3 * torch.randn((self.tot_entity, k), dtype=torch.double, device=self.device))
        self.rel_embeddings = NamedEmbedding("rel_embedding", self.tot_relation, k, padding_idx=0)
        self.rel_embeddings.weight.data = (
                    1e-3 * torch.randn((self.tot_relation, k), dtype=torch.double, device=self.device))
        self.wu = nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (self.tot_relation, k)), dtype=torch.double, requires_grad=True,
                         device=self.device))
        self.bs = nn.Parameter(
            torch.zeros(self.tot_entity, dtype=torch.double, requires_grad=True, device=self.device))
        self.bo = nn.Parameter(
            torch.zeros(self.tot_entity, dtype=torch.double, requires_grad=True, device=self.device))

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
        ]

        self.loss = Criterion.pointwise_bce

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

    def forward(self, h, r, t):
        return self._poincare_forward(h, r, t)

    def predict_tail_rank(self, h, r, topk):
        del topk
        _, rank = torch.sort(self.forward(h, r, torch.LongTensor(list(range(self.tot_entity))).to(self.device)))
        return rank

    def predict_head_rank(self, t, r, topk):
        del topk
        _, rank = torch.sort(self.forward(torch.LongTensor(list(range(self.tot_entity))).to(self.device), r, t))
        return rank

    def predict_rel_rank(self, h, t, topk):
        del topk
        _, rank = torch.sort(self.forward(h, torch.LongTensor(list(range(self.tot_relation))).to(self.device), t))
        return rank

    def _poincare_forward(self, h, r, t):
        h_emb, r_emb, t_emb = self.embed(h, r, t)
        ru = self.wu[r]

        h_emb = torch.where(torch.norm(h_emb, 2, dim=-1, keepdim=True) >= 1,
                            h_emb / (torch.norm(h_emb, 2, dim=-1, keepdim=True) - 1e-5), h_emb)
        t_emb = torch.where(torch.norm(t_emb, 2, dim=-1, keepdim=True) >= 1,
                            t_emb / (torch.norm(t_emb, 2, dim=-1, keepdim=True) - 1e-5), t_emb)
        r_emb = torch.where(torch.norm(r_emb, 2, dim=-1, keepdim=True) >= 1,
                            r_emb / (torch.norm(r_emb, 2, dim=-1, keepdim=True) - 1e-5), r_emb)
        u_e = self._p_log_map(h_emb)
        u_w = u_e * ru
        u_m = self._p_exp_map(u_w)
        v_m = self._p_sum(t_emb, r_emb)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1,
                          u_m / (torch.norm(u_m, 2, dim=-1, keepdim=True) - 1e-5), u_m)
        v_m = torch.where(torch.norm(v_m, 2, dim=-1, keepdim=True) >= 1,
                          v_m / (torch.norm(v_m, 2, dim=-1, keepdim=True) - 1e-5), v_m)

        sqdist = (2. * self._arsech(
            torch.clamp(torch.norm(self._p_sum(-u_m, v_m), 2, dim=-1), 1e-10, 1 - 1e-5))) ** 2
        return -(sqdist - self.bs[h] - self.bo[t])

    def _euclidean_forward(self, h, r, t):
        h_emb, r_emb, t_emb = self.embed(h, r, t)
        ru = self.wu[r]
        u_w = h_emb * ru

        sqdist = torch.sum(torch.pow(u_w - (t_emb + r_emb), 2), dim=-1)
        return -(sqdist - self.bs[h] - self.bo[t])

    @staticmethod
    def _arsech(x):
        return torch.log((1 + torch.sqrt(1 - x.pow(2))) / x)

    @staticmethod
    def _p_exp_map(v):
        normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
        return (1 / torch.cosh(normv)) * v / normv

    @staticmethod
    def _p_log_map(v):
        normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1 - 1e-5)
        return MuRP._arsech(normv) * v / normv

    @staticmethod
    def _p_sum(x, y):
        sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1 - 1e-5)
        sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1 - 1e-5)
        dotxy = torch.sum(x * y, dim=-1, keepdim=True)
        numerator = (1 + 2 * dotxy + sqynorm) * x + (1 - sqxnorm) * y
        denominator = 1 + 2 * dotxy + sqxnorm * sqynorm
        return numerator / denominator
