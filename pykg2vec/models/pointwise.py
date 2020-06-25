#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn

from pykg2vec.models.KGMeta import ModelMeta
from pykg2vec.models.Domain import NamedEmbedding
from pykg2vec.data.generator import TrainingStrategy


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
        >>> from pykg2vec.models.Complex import Complex
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
        >>> from pykg2vec.models.Complex import Complex
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


class CP(ModelMeta):

    def __init__(self, config):
        super(CP, self).__init__()
        self.config = config
        self.model_name = 'CP'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.sub_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.obj_embeddings = nn.Embedding(num_total_ent, k)
        nn.init.xavier_uniform_(self.sub_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.obj_embeddings.weight)

        self.parameter_list = [
            NamedEmbedding(self.sub_embeddings, "sub_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.obj_embeddings, "obj_embedding"),
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
        emb_h = self.sub_embeddings(h)
        emb_r = self.rel_embeddings(r)
        emb_t = self.obj_embeddings(t)
        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        return -torch.sum(h_e * r_e * t_e, -1)

    def get_reg(self, h, r, t, type='N3'):
        h_e, r_e, t_e = self.embed(h, r, t)
        if type.lower() == 'f2':
            regul_term = torch.mean(torch.sum(h_e**2, -1) + torch.sum(r_e**2, -1) + torch.sum(t_e**2, -1))
        elif type.lower() == 'n3':
            regul_term = torch.mean(torch.sum(h_e**3, -1) + torch.sum(r_e**3, -1) + torch.sum(t_e**3, -1))
        else:
            raise NotImplementedError('Unknown regularizer type: %s' % type)

        return self.config.lmbda * regul_term


class DistMult(ModelMeta):
    """`EMBEDDING ENTITIES AND RELATIONS FOR LEARNING AND INFERENCE IN KNOWLEDGE BASES`_

        DistMult is a simpler model comparing with RESCAL in that it simplifies
        the weight matrix used in RESCAL to a diagonal matrix. The scoring
        function used DistMult can capture the pairwise interactions between
        the head and the tail entities. However, DistMult has limitation on modeling
        asymmetric relations.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            tot_ent (int): Total unique entites in the knowledge graph.
            tot_rel (int): Total unique relation in the knowledge graph.
            model (str): Name of the model.

        Examples:
            >>> from pykg2vec.models.Complex import DistMult
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = DistMult()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _EMBEDDING ENTITIES AND RELATIONS FOR LEARNING AND INFERENCE IN KNOWLEDGE BASES:
            https://arxiv.org/pdf/1412.6575.pdf

    """

    def __init__(self, config):
        super(DistMult, self).__init__()
        self.config = config
        self.model_name = 'DistMult'
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

    def get_reg(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        regul_term = torch.mean(torch.sum(h_e**2, -1) + torch.sum(r_e**2, -1) + torch.sum(t_e**2,-1))
        return self.config.lmbda*regul_term


class SimplE(ModelMeta):

    def __init__(self, config):
        super(SimplE, self).__init__()
        self.config = config
        self.model_name = 'SimplE_avg'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_head_embeddings = nn.Embedding(num_total_ent, k)
        self.ent_tail_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.rel_inv_embeddings = nn.Embedding(num_total_rel, k)
        nn.init.xavier_uniform_(self.ent_head_embeddings.weight)
        nn.init.xavier_uniform_(self.ent_tail_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_inv_embeddings.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_head_embeddings, "ent_head_embedding"),
            NamedEmbedding(self.ent_tail_embeddings, "ent_tail_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.rel_inv_embeddings, "rel_inv_embedding"),
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

    def get_reg(self, h, r, t):
        num_batch = math.ceil(self.config.kg_meta.tot_train_triples / self.config.batch_size)
        regul_term = (self._get_l2_loss(self.ent_head_embeddings) + self._get_l2_loss(self.ent_tail_embeddings) +
                      self._get_l2_loss(self.rel_embeddings) + self._get_l2_loss(self.rel_inv_embeddings)) / num_batch**2
        return self.config.lmbda * regul_term

    @staticmethod
    def _get_l2_loss(embeddings):
        return torch.sum(embeddings.weight**2) / 2


class SimplE_ignr(SimplE):

    def __init__(self, config):
        super(SimplE_ignr, self).__init__(config)
        self.config = config
        self.model_name = 'SimplE_ignr'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

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

    def get_reg(self, h, r, t):
        return 2.0 * super().get_reg(h, r, t)

    @staticmethod
    def _concat_selected_embeddings(e1, t1, e2, t2):
        return torch.cat([torch.index_select(e1.weight, 0, t1), torch.index_select(e2.weight, 0, t2)], 1)