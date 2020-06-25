#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pykg2vec.models.KGMeta import ModelMeta
from pykg2vec.models.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class TransE(ModelMeta):
    """ `Translating Embeddings for Modeling Multi-relational Data`_

        TransE is an energy based model which represents the
        relationships as translations in the embedding space. Which
        means that if (h,l,t) holds then the embedding of the tail
        't' should be close to the embedding of head entity 'h'
        plus some vector that depends on the relationship 'l'.
        Both entities and relations are vectors in the same space.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.models.TransE import TransE
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransE()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on `OpenKE_TransE`_ and `wencolani`_.

        .. _OpenKE_TransE: https://github.com/thunlp/OpenKE/blob/master/models/TransE.py

        .. _wencolani: https://github.com/wencolani/TransE.git

        .. _Translating Embeddings for Modeling Multi-relational Data:
            http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-rela
    """

    def __init__(self, config):

        super(TransE, self).__init__()

        self.config = config
        self.model_name = 'TransE'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        self.ent_embeddings = nn.Embedding(self.config.kg_meta.tot_entity, self.config.hidden_size)
        self.rel_embeddings = nn.Embedding(self.config.kg_meta.tot_relation, self.config.hidden_size)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
        ]

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

        if self.config.L1_flag:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=1, dim=-1)
        else:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=2, dim=-1)

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids.
               t (Tensor): Tail entity ids.

            Returns:
                Tensors: Returns a tuple of head, relation and tail embedding Tensors.
        """
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        t_e = self.ent_embeddings(t)

        return h_e, r_e, t_e


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
            >>> from pykg2vec.models.TransH import TransH
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


class TransD(ModelMeta):
    """ `Knowledge Graph Embedding via Dynamic Mapping Matrix`_

        TransD constructs a dynamic mapping matrix for each entity-relation pair by considering the diversity of entities and relations simultaneously.
        Compared with TransR/CTransR, TransD has fewer parameters and has no matrix vector multiplication.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.models.TransD import TransD
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransD()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on `OpenKE_TransD`_.

        .. _OpenKE_TransD:
            https://github.com/thunlp/OpenKE/blob/master/models/TransD.py

        .. _Knowledge Graph Embedding via Dynamic Mapping Matrix:
            https://www.aclweb.org/anthology/P15-1067
    """

    def __init__(self, config):
        super(TransD, self).__init__()
        self.config = config
        self.model_name = 'TransD'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        self.ent_embeddings = nn.Embedding(self.config.kg_meta.tot_entity, d)
        self.rel_embeddings = nn.Embedding(self.config.kg_meta.tot_relation, k)
        self.ent_mappings = nn.Embedding(self.config.kg_meta.tot_entity, d)
        self.rel_mappings = nn.Embedding(self.config.kg_meta.tot_relation, k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.ent_mappings.weight)
        nn.init.xavier_uniform_(self.rel_mappings.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.ent_mappings, "ent_mappings"),
            NamedEmbedding(self.rel_mappings, "rel_mappings"),
        ]

    def projection(self, emb_e, emb_m, proj_vec):
        # [b, k] + sigma ([b, k] * [b, k]) * [b, k]
        return emb_e + torch.sum(emb_e * emb_m, axis=-1, keepdims=True) * proj_vec
        
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

        h_m = self.ent_mappings(h)
        r_m = self.rel_mappings(r)
        t_m = self.ent_mappings(t)

        emb_h = self.projection(emb_h, h_m, r_m)
        emb_t = self.projection(emb_t, t_m, r_m)

        return emb_h, emb_r, emb_t

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

        if self.config.L1_flag:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=1, dim=-1)
        else:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=2, dim=-1)


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
            >>> from pykg2vec.models.TransM import TransM
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
        self.theta = torch.from_numpy(np.asarray(theta, dtype=np.float32)).to(self.config.device)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
        ]

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


class TransR(ModelMeta):
    """ `Learning Entity and Relation Embeddings for Knowledge Graph Completion`_

        TranR is a translation based knowledge graph embedding method. Similar to TransE and TransH, it also
        builds entity and relation embeddings by regarding a relation as translation from head entity to tail
        entity. However, compared to them, it builds the entity and relation embeddings in a separate entity
        and relation spaces.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            model_name (str): Name of the model.
            data_stats (object): Class object with knowlege graph statistics.

        Examples:
            >>> from pykg2vec.models.TransR import TransR
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = TransR()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        Portion of the code based on `thunlp_transR`_.

         .. _thunlp_transR:
             https://github.com/thunlp/TensorFlow-TransX/blob/master/transR.py

        .. _Learning Entity and Relation Embeddings for Knowledge Graph Completion:
            http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf
    """

    def __init__(self, config):
        super(TransR, self).__init__()
        self.config = config
        self.model_name = 'TransR'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.ent_hidden_size
        d = self.config.rel_hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, d)
        self.rel_matrix = nn.Embedding(num_total_rel, k * d)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_matrix.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.rel_matrix, "rel_matrix"),
        ]

    def transform(self, e, matrix):
        matrix = matrix.view(-1, self.config.ent_hidden_size, self.config.rel_hidden_size)
        if e.shape[0] != matrix.shape[0]:
            e = e.view(-1, matrix.shape[0], self.config.ent_hidden_size).permute(1, 0, 2)
            e = torch.matmul(e, matrix).permute(1, 0, 2)
        else:
            e = e.view(-1, 1, self.config.ent_hidden_size)
            e = torch.matmul(e, matrix)
        return e.view(-1, self.config.rel_hidden_size)

    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        t_e = self.ent_embeddings(t)

        h_e = F.normalize(h_e, p=2, dim=-1)
        r_e = F.normalize(r_e, p=2, dim=-1)
        t_e = F.normalize(t_e, p=2, dim=-1)

        h_e = torch.unsqueeze(h_e, 1)
        t_e = torch.unsqueeze(t_e, 1)
        # [b, 1, k]

        matrix = self.rel_matrix(r)
        # [b, k, d]

        transform_h_e = self.transform(h_e, matrix)
        transform_t_e = self.transform(t_e, matrix)
        # [b, 1, d] = [b, 1, k] * [b, k, d]

        h_e = torch.squeeze(transform_h_e, axis=1)
        t_e = torch.squeeze(transform_t_e, axis=1)
        # [b, d]
        return h_e, r_e, t_e

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

        if self.config.L1_flag:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=1, dim=-1)
        else:
            return torch.norm(norm_h_e + norm_r_e - norm_t_e, p=2, dim=-1)


class SLM(ModelMeta):
    """`Reasoning With Neural Tensor Networks for Knowledge Base Completion`_

        SLM model is designed as a baseline of Neural Tensor Network.
        The model constructs a nonlinear neural network to represent the score function.

        Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.models.SLM import SLM
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = SLM()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _Reasoning With Neural Tensor Networks for Knowledge Base Completion:
            https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    """

    def __init__(self, config):
        super(SLM, self).__init__()
        self.config = config
        self.model_name = 'SLM'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, d)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.mr1 = nn.Embedding(d, k)
        self.mr2 = nn.Embedding(d, k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.mr1.weight)
        nn.init.xavier_uniform_(self.mr2.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.mr1, "mr1"),
            NamedEmbedding(self.mr2, "mr2"),
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

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        norm_h = F.normalize(h_e, p=2, dim=-1)
        norm_r = F.normalize(r_e, p=2, dim=-1)
        norm_t = F.normalize(t_e, p=2, dim=-1)
        return -torch.sum(norm_r * self.layer(norm_h, norm_t), -1)

    def layer(self, h, t):
        """Defines the forward pass layer of the algorithm.

          Args:
              h (Tensor): Head entities ids.
              t (Tensor): Tail entity ids of the triple.
        """       
        mr1h = torch.matmul(h, self.mr1.weight) # h => [m, d], self.mr1 => [d, k]
        mr2t = torch.matmul(t, self.mr2.weight) # t => [m, d], self.mr2 => [d, k]
        return torch.tanh(mr1h + mr2t)


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
        >>> from pykg2vec.models.SME import SME
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
        >>> from pykg2vec.models.SME import SME
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
            >>> from pykg2vec.models.RotatE import RotatE
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
            >>> from pykg2vec.models.Rescal import Rescal
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

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_matrices, "rel_matrices"),
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

    @staticmethod
    def get_normalized_data(embedding, num_embeddings, p=2, dim=1):
        norms = torch.norm(embedding.weight, p, dim).data
        return embedding.weight.data.div(norms.view(num_embeddings, 1).expand_as(embedding.weight))


class NTN(ModelMeta):
    """ `Reasoning With Neural Tensor Networks for Knowledge Base Completion`_

    It is a neural tensor network which represents entities as an average of their
    constituting word vectors. It then projects entities to their vector embeddings
    in the input layer. The two entities are then combined and mapped to a non-linear hidden layer.
    https://github.com/siddharth-agrawal/Neural-Tensor-Network/blob/master/neuralTensorNetwork.py

     Args:
        config (object): Model configuration parameters.

     Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

     Examples:
        >>> from pykg2vec.models.NTN import NTN
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = NTN()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

     Portion of the code based on `siddharth-agrawal`_.

     .. _siddharth-agrawal:
         https://github.com/siddharth-agrawal/Neural-Tensor-Network/blob/master/neuralTensorNetwork.py

     .. _Reasoning With Neural Tensor Networks for Knowledge Base Completion:
         https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf
    """

    def __init__(self, config):
        super(NTN, self).__init__()
        self.config = config
        self.model_name = 'NTN'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, d)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.mr1 = nn.Embedding(d, k)
        self.mr2 = nn.Embedding(d, k)
        self.br = nn.Embedding(1, k)
        self.mr = nn.Embedding(k, d*d)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.mr1.weight)
        nn.init.xavier_uniform_(self.mr2.weight)
        nn.init.xavier_uniform_(self.br.weight)
        nn.init.xavier_uniform_(self.mr.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.mr1, "mr1"),
            NamedEmbedding(self.mr2, "mr2"),
            NamedEmbedding(self.br, "br"),
            NamedEmbedding(self.mr, "mr"),
        ]

    def train_layer(self, h, t):
        """Defines the forward pass training layers of the algorithm.

           Args:
               h (Tensor): Head entities ids.
               t (Tensor): Tail entity ids of the triple.
        """
        d = self.config.ent_hidden_size
        k = self.config.rel_hidden_size
        
        mr1h = torch.matmul(h, self.mr1.weight) # h => [m, d], self.mr1 => [d, k]
        mr2t = torch.matmul(t, self.mr2.weight) # t => [m, d], self.mr2 => [d, k]

        expanded_h = h.unsqueeze(dim=0).repeat(k, 1, 1) # [k, m, d]
        expanded_t = t.unsqueeze(dim=-1) # [m, d, 1]

        temp = (torch.matmul(expanded_h, self.mr.weight.view(k, d, d))).permute(1, 0, 2) # [m, k, d]
        htmrt = torch.squeeze(torch.matmul(temp, expanded_t), dim=-1) # [m, k]

        return F.tanh(htmrt + mr1h + mr2t + self.br.weight)

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
        h_e, r_e, t_e = self.embed(h, r, t)
        norm_h = F.normalize(h_e, p=2, dim=-1)
        norm_r = F.normalize(r_e, p=2, dim=-1)
        norm_t = F.normalize(t_e, p=2, dim=-1)
        return -torch.sum(norm_r*self.train_layer(norm_h, norm_t), -1)

    def get_reg(self):
        return self.config.lmbda*torch.sqrt(sum([torch.sum(torch.pow(var.weight, 2)) for var in self.parameter_list]))


class KG2E(ModelMeta):
    """`Learning to Represent Knowledge Graphs with Gaussian Embedding`_

    Instead of assumming entities and relations as determinstic points in the
    embedding vector spaces, KG2E models both entities and relations (h, r and t)
    using random variables derived from multivariate Gaussian distribution.
    KG2E then evaluates a fact using translational relation by evaluating the
    distance between two distributions, r and t-h. KG2E provides two distance
    measures (KL-divergence and estimated likelihood).

    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

    Examples:
        >>> from pykg2vec.models.KG2E import KG2E
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = KG2E()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    Portion of the code based on `this Source`_.
    
    .. _this Source:
        https://github.com/mana-ysh/gaussian-embedding/blob/master/src/models/gaussian_model.py

    .. _Learning to Represent Knowledge Graphs with Gaussian Embedding:
        https://pdfs.semanticscholar.org/0ddd/f37145689e5f2899f8081d9971882e6ff1e9.pdf
    
    """

    def __init__(self, config=None):
        super(KG2E, self).__init__()
        self.config = config
        self.model_name = 'KG2E_KL'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        # the mean for each element in the embedding space. 
        self.ent_embeddings_mu = nn.Embedding(num_total_ent, k)
        self.rel_embeddings_mu = nn.Embedding(num_total_rel, k)

        # as the paper suggested, sigma is simplified to be the diagonal element in the covariance matrix. 
        self.ent_embeddings_sigma = nn.Embedding(num_total_ent, k)
        self.rel_embeddings_sigma = nn.Embedding(num_total_rel, k)

        nn.init.xavier_uniform_(self.ent_embeddings_mu.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_mu.weight)
        nn.init.xavier_uniform_(self.ent_embeddings_sigma.weight)
        nn.init.xavier_uniform_(self.rel_embeddings_sigma.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings_mu, "ent_embeddings_mu"),
            NamedEmbedding(self.ent_embeddings_sigma, "ent_embeddings_sigma"),
            NamedEmbedding(self.rel_embeddings_mu, "rel_embeddings_mu"),
            NamedEmbedding(self.rel_embeddings_sigma, "rel_embeddings_sigma"),
        ]

        min_ent = torch.min(torch.FloatTensor().new_full(self.ent_embeddings_sigma.weight.shape, self.config.cmax), torch.add(self.ent_embeddings_sigma.weight, 1.0))
        self.ent_embeddings_sigma.weight = nn.Parameter(torch.max(torch.FloatTensor().new_full(self.ent_embeddings_sigma.weight.shape, self.config.cmin), min_ent))
        min_rel = torch.min(torch.FloatTensor().new_full(self.rel_embeddings_sigma.weight.shape, self.config.cmax), torch.add(self.rel_embeddings_sigma.weight, 1.0))
        self.rel_embeddings_sigma.weight = nn.Parameter(torch.max(torch.FloatTensor().new_full(self.rel_embeddings_sigma.weight.shape, self.config.cmin), min_rel))

    def forward(self, h, r, t):
        h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma = self.embed(h, r, t)
        return self._cal_score_expected_likelihood(h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma)

    def embed(self, h, r, t):
        """Function to get the embedding.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """

        self.ent_embeddings_mu.weight.data = self.get_normalized_data(self.ent_embeddings_mu, self.config.kg_meta.tot_entity)
        self.rel_embeddings_mu.weight.data = self.get_normalized_data(self.rel_embeddings_mu, self.config.kg_meta.tot_relation)

        self.ent_embeddings_sigma.weight.data = self.get_normalized_data(self.ent_embeddings_sigma, self.config.kg_meta.tot_entity)
        self.rel_embeddings_sigma.weight.data = self.get_normalized_data(self.rel_embeddings_sigma, self.config.kg_meta.tot_relation)

        emb_h_mu = self.ent_embeddings_mu(h)
        emb_r_mu = self.rel_embeddings_mu(r)
        emb_t_mu = self.ent_embeddings_mu(t)

        emb_h_sigma = self.ent_embeddings_sigma(h)
        emb_r_sigma = self.rel_embeddings_sigma(r)
        emb_t_sigma = self.ent_embeddings_sigma(t)

        return emb_h_mu, emb_h_sigma, emb_r_mu, emb_r_sigma, emb_t_mu, emb_t_sigma

    @staticmethod
    def get_normalized_data(embedding, num_embeddings, p=2, dim=1):
        norms = torch.norm(embedding.weight, p, dim).data
        return embedding.weight.data.div(norms.view(num_embeddings, 1).expand_as(embedding.weight))

    def _cal_score_expected_likelihood(self, h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma):
        """ It calculates the expected likelihood as a score.

            mul_fac: (mu_h + mu_r - mu_t).T * sigma_r-1 * (mu_h + mu_r - mu_t)
            det_fac: log(det(sigma_r + sigma_h + sigma_t))

            Args:
                 h_mu (Tensor): Mean of the embedding value of the head.
                 h_sigma(Tensor): Variance of the embedding value of the head.
                 r_mu(Tensor): Mean of the embedding value of the relation.
                 r_sigma(Tensor): Variance of the embedding value of the relation.
                 t_mu(Tensor): Mean of the embedding value of the tail.
                 t_sigma(Tensor): Variance of the embedding value of the tail.

            Returns:
                Tensor: Score after calculating the expected likelihood.
        """
        mul_fac = torch.sum((h_mu + r_mu - t_mu) ** 2 / (h_sigma + r_sigma + t_sigma), -1)
        det_fac = torch.sum(torch.log(h_sigma + r_sigma + t_sigma), -1)

        return mul_fac + det_fac - self.config.hidden_size


class KG2E_EL(KG2E):
    """`Learning to Represent Knowledge Graphs with Gaussian Embedding`_

    Instead of assumming entities and relations as determinstic points in the
    embedding vector spaces, KG2E models both entities and relations (h, r and t)
    using random variables derived from multivariate Gaussian distribution.
    KG2E then evaluates a fact using translational relation by evaluating the
    distance between two distributions, r and t-h. KG2E provides two distance
    measures (KL-divergence and estimated likelihood).

    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
        data_stats (object): Class object with knowlege graph statistics.

    Examples:
        >>> from pykg2vec.models.KG2E import KG2E
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = KG2E()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    Portion of the code based on `this Source`_.
    
    .. _this Source:
        https://github.com/mana-ysh/gaussian-embedding/blob/master/src/models/gaussian_model.py

    .. _Learning to Represent Knowledge Graphs with Gaussian Embedding:
        https://pdfs.semanticscholar.org/0ddd/f37145689e5f2899f8081d9971882e6ff1e9.pdf
    
    """

    def __init__(self, config):
        super(KG2E_EL, self).__init__(config)
        self.config = config
        self.model_name = 'KG2E_EL'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    def forward(self, h, r, t):
        h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma = self.embed(h, r, t)
        return self._cal_score_kl_divergence(h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma)

    def _cal_score_kl_divergence(self, h_mu, h_sigma, r_mu, r_sigma, t_mu, t_sigma):
        """ It calculates the kl_divergence as a score.

            trace_fac: tr(sigma_r-1 * (sigma_h + sigma_t))
            mul_fac: (mu_h + mu_r - mu_t).T * sigma_r-1 * (mu_h + mu_r - mu_t)
            det_fac: log(det(sigma_r)/det(sigma_h + sigma_t))

            Args:
                 h_mu (Tensor): Mean of the embedding value of the head.
                 h_sigma(Tensor): Variance of the embedding value of the head.
                 r_mu(Tensor): Mean of the embedding value of the relation.
                 r_sigma(Tensor): Variance of the embedding value of the relation.
                 t_mu(Tensor): Mean of the embedding value of the tail.
                 t_sigma(Tensor): Variance of the embedding value of the tail.

            Returns:
                Tensor: Score after calculating the KL_Divergence.

        """
        trace_fac = torch.sum((h_sigma + t_sigma) / r_sigma, -1)
        mul_fac = torch.sum((- h_mu + t_mu - r_mu) ** 2 / r_sigma, -1)
        det_fac = torch.sum(torch.log(h_sigma + t_sigma) - torch.log(r_sigma), -1)

        return trace_fac + mul_fac - det_fac - self.config.hidden_size


class HoLE(ModelMeta):
    """`Holographic Embeddings of Knowledge Graphs`_.

    HoLE employs the circular correlation to create composition correlations. It
    is able to represent and capture the interactions betweek entities and relations
    while being efficient to compute, easier to train and scalable to large dataset.

    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        model_name (str): Name of the model.
    
    Examples:
        >>> from pykg2vec.models.HoLE import HoLE
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = HoLE()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Holographic Embeddings of Knowledge Graphs:
        https://arxiv.org/pdf/1510.04935.pdf

    """

    def __init__(self, config):
        super(HoLE, self).__init__()
        self.config = config
        self.model_name = 'HoLE'
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

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

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        r_e = F.normalize(r_e, p=2, dim=-1)
        h_e = torch.stack((h_e, torch.zeros_like(h_e)), -1)
        t_e = torch.stack((t_e, torch.zeros_like(t_e)), -1)
        e, _ = torch.unbind(torch.ifft(torch.conj(torch.fft(h_e, 1)) * torch.fft(t_e, 1), 1), -1)
        return -F.sigmoid(torch.sum(r_e * e, 1))

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
