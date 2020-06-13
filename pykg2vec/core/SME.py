#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class SME(ModelMeta):
    """ `A Semantic Matching Energy Function for Learning with Multi-relational Data`_

    Semantic Matching Energy (SME) is an algorithm for embedding multi-relational data into vector spaces.
    SME conducts semantic matching using neural network architectures. Given a fact (h, r, t), it first projects
    entities and relations to their embeddings in the input layer. Later the relation r is combined with both h and t
    to get gu(h, r) and gv(r, t) in its hidden layer. The score is determined by calculating the matching score of gu and gv.

    There are two versions of SME: a linear version(SMELinear) as well as bilinear(SMEBilinear) version which differ in how the hidden layer is defined.

    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

    Examples:
        >>> from pykg2vec.core.SME import SME
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = SME()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    Portion of the code based on glorotxa_.
    
    .. _glorotxa: https://github.com/glorotxa/SME/blob/master/model.py

    .. _A Semantic Matching Energy Function for Learning with Multi-relational Data: http://www.thespermwhale.com/jaseweston/papers/ebrm_mlj.pdf
    
    """

    def __init__(self, config):
        super(SME, self).__init__()
        self.config = config
        self.model_name = 'SME_Linear'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.mu1 = nn.Embedding(k, k)
        self.mu2 = nn.Embedding(k, k)
        self.bu = nn.Embedding(k, 1)
        self.mv1 = nn.Embedding(k, k)
        self.mv2 = nn.Embedding(k, k)
        self.bv = nn.Embedding(k, 1)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.mu1.weight)
        nn.init.xavier_uniform_(self.mu2.weight)
        nn.init.xavier_uniform_(self.bu.weight)
        nn.init.xavier_uniform_(self.mv1.weight)
        nn.init.xavier_uniform_(self.mv2.weight)
        nn.init.xavier_uniform_(self.bv.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.mu1, "mu1"),
            NamedEmbedding(self.mu2, "mu2"),
            NamedEmbedding(self.bu, "bu"),
            NamedEmbedding(self.mv1, "mv1"),
            NamedEmbedding(self.mv2, "mv2"),
            NamedEmbedding(self.bv, "bv"),
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
        emb_h = self.ent_embeddings(h)
        emb_r = self.rel_embeddings(r)
        emb_t = self.ent_embeddings(t)
        return emb_h, emb_r, emb_t

    def _gu_linear(self, h, r):
        """Function to calculate linear loss.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.

            Returns:
                Tensors: Returns the bilinear loss.
        """
        mu1h = torch.matmul(self.mu1.weight, self.transpose(h)) # [k, b]
        mu2r = torch.matmul(self.mu2.weight, self.transpose(r)) # [k, b]
        return self.transpose(mu1h + mu2r + self.bu.weight)  # [b, k]

    def _gv_linear(self, r, t):
        """Function to calculate linear loss.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.

            Returns:
                Tensors: Returns the bilinear loss.
        """
        mv1t = torch.matmul(self.mv1.weight, self.transpose(t)) # [k, b]
        mv2r = torch.matmul(self.mv2.weight, self.transpose(r)) # [k, b]
        return self.transpose(mv1t + mv2r + self.bv.weight)  # [b, k]

    def forward(self, h, r, t):
        """Function to that performs semanting matching.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.
                t (Tensor): Tail ids of the triple.

            Returns:
                Tensors: Returns the semantic matchin score.
        """
        h_e, r_e, t_e = self.embed(h, r, t)
        norm_h = F.normalize(h_e, p=2, dim=-1)
        norm_r = F.normalize(r_e, p=2, dim=-1)
        norm_t = F.normalize(t_e, p=2, dim=-1)

        return -torch.sum(self._gu_linear(norm_h, norm_r) * self._gv_linear(norm_r, norm_t), 1)

    @staticmethod
    def transpose(tensor):
        dims = tuple(range(len(tensor.shape)-1, -1, -1))    # (rank-1...0)
        return tensor.permute(dims)

class SME_BL(SME):
    """ `A Semantic Matching Energy Function for Learning with Multi-relational Data`_

    Semantic Matching Energy (SME) is an algorithm for embedding multi-relational data into vector spaces.
    SME conducts semantic matching using neural network architectures. Given a fact (h, r, t), it first projects
    entities and relations to their embeddings in the input layer. Later the relation r is combined with both h and t
    to get gu(h, r) and gv(r, t) in its hidden layer. The score is determined by calculating the matching score of gu and gv.

    There are two versions of SME: a linear version(SMELinear) as well as bilinear(SMEBilinear) version which differ in how the hidden layer is defined.

    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

    Examples:
        >>> from pykg2vec.core.SME import SME
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = SME()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    Portion of the code based on glorotxa_.
    
    .. _glorotxa: https://github.com/glorotxa/SME/blob/master/model.py

    .. _A Semantic Matching Energy Function for Learning with Multi-relational Data: http://www.thespermwhale.com/jaseweston/papers/ebrm_mlj.pdf
    
    """

    def __init__(self, config):
        super(SME_BL, self).__init__(config)
        self.config = config
        self.model_name = 'SME_Bilinear'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    def _gu_bilinear(self, h, r):
        """Function to calculate bilinear loss.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.

            Returns:
                Tensors: Returns the bilinear loss.
        """
        mu1h = torch.matmul(self.mu1.weight, self.transpose(h)) # [k, b]
        mu2r = torch.matmul(self.mu2.weight, self.transpose(r)) # [k, b]
        return self.transpose(mu1h * mu2r + self.bu.weight)  # [b, k]

    def _gv_bilinear(self, r, t):
        """Function to calculate bilinear loss.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.

            Returns:
                Tensors: Returns the bilinear loss.
        """
        mv1t = torch.matmul(self.mv1.weight, self.transpose(t)) # [k, b]
        mv2r = torch.matmul(self.mv2.weight, self.transpose(r)) # [k, b]
        return self.transpose(mv1t * mv2r + self.bv.weight)  # [b, k]

    def forward(self, h, r, t):
        """Function to that performs semanting matching.

            Args:
                h (Tensor): Head entities ids.
                r (Tensor): Relation ids of the triple.
                t (Tensor): Tail ids of the triple.

            Returns:
                Tensors: Returns the semantic matchin score.
        """
        h_e, r_e, t_e = self.embed(h, r, t)
        norm_h = F.normalize(h_e, p=2, dim=-1)
        norm_r = F.normalize(r_e, p=2, dim=-1)
        norm_t = F.normalize(t_e, p=2, dim=-1)

        return torch.sum(self._gu_bilinear(norm_h, norm_r) * self._gv_bilinear(norm_r, norm_t), -1)