#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.core.Domain import NamedEmbedding
from pykg2vec.utils.generator import TrainingStrategy


class ProjE_pointwise(ModelMeta):
    """`ProjE-Embedding Projection for Knowledge Graph Completion`_.

        Instead of measuring the distance or matching scores between the pair of the
        head entity and relation and then tail entity in embedding space ((h,r) vs (t)).
        ProjE projects the entity candidates onto a target vector representing the
        input data. The loss in ProjE is computed by the cross-entropy between
        the projected target vector and binary label vector, where the included
        entities will have value 0 if in negative sample set and value 1 if in
        positive sample set.

         Args:
            config (object): Model configuration parameters.

        Attributes:
            config (object): Model configuration.
            data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
            model_name (str): Name of the model.

        Examples:
            >>> from pykg2vec.core.ProjE_pointwise import ProjE_pointwise
            >>> from pykg2vec.utils.trainer import Trainer
            >>> model = ProjE_pointwise()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _ProjE-Embedding Projection for Knowledge Graph Completion:
            https://arxiv.org/abs/1611.05425

    """

    def __init__(self, config):
        super(ProjE_pointwise, self).__init__()
        self.config = config
        self.model_name = 'ProjE_pointwise'
        self.training_strategy = TrainingStrategy.PROJECTION_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        self.rel_embeddings = nn.Embedding(num_total_rel, k)
        self.bc1 = nn.Embedding(1, k)
        self.De1 = nn.Embedding(1, k)
        self.Dr1 = nn.Embedding(1, k)
        self.bc2 = nn.Embedding(1, k)
        self.De2 = nn.Embedding(1, k)
        self.Dr2 = nn.Embedding(1, k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.bc1.weight)
        nn.init.xavier_uniform_(self.De1.weight)
        nn.init.xavier_uniform_(self.Dr1.weight)
        nn.init.xavier_uniform_(self.bc2.weight)
        nn.init.xavier_uniform_(self.De2.weight)
        nn.init.xavier_uniform_(self.Dr2.weight)

        self.parameter_list = [
            NamedEmbedding(self.ent_embeddings, "ent_embedding"),
            NamedEmbedding(self.rel_embeddings, "rel_embedding"),
            NamedEmbedding(self.bc1, "bc1"),
            NamedEmbedding(self.De1, "De1"),
            NamedEmbedding(self.Dr1, "Dr1"),
            NamedEmbedding(self.bc2, "bc2"),
            NamedEmbedding(self.De2, "De2"),
            NamedEmbedding(self.Dr2, "Dr2"),
        ]

    def get_reg(self):
        return self.config.lmbda*(torch.sum(torch.abs(self.De1.weight) + torch.abs(self.Dr1.weight)) + torch.sum(torch.abs(self.De2.weight)
               + torch.abs(self.Dr2.weight)) + torch.sum(torch.abs(self.ent_embeddings.weight)) + torch.sum(torch.abs(self.rel_embeddings.weight)))

    def forward(self, e, r, er_e2, direction="tail"):
        emb_hr_e = self.ent_embeddings(e)  # [m, k]
        emb_hr_r = self.rel_embeddings(r)  # [m, k]
        
        if direction == "tail":
            ere2_sigmoid = self.g(torch.dropout(self.f1(emb_hr_e, emb_hr_r), p=self.config.hidden_dropout, train=True), self.ent_embeddings.weight)
        else:
            ere2_sigmoid = self.g(torch.dropout(self.f2(emb_hr_e, emb_hr_r), p=self.config.hidden_dropout, train=True), self.ent_embeddings.weight)

        ere2_loss_left = -torch.sum((torch.log(torch.clamp(ere2_sigmoid, 1e-10, 1.0)) * torch.max(torch.FloatTensor([0]), er_e2)))
        ere2_loss_right = -torch.sum((torch.log(torch.clamp(1 - ere2_sigmoid, 1e-10, 1.0)) * torch.max(torch.FloatTensor([0]), torch.neg(er_e2))))

        hrt_loss = ere2_loss_left + ere2_loss_right

        return hrt_loss

    def f1(self, h, r):
        """Defines froward layer for head.

            Args:
                   h (Tensor): Head entities ids.
                   r (Tensor): Relation ids of the triple.
        """
        return torch.tanh(h * self.De1.weight + r * self.Dr1.weight + self.bc1.weight)

    def f2(self, t, r):
        """Defines forward layer for tail.

            Args:
               t (Tensor): Tail entities ids.
               r (Tensor): Relation ids of the triple.
        """
        return torch.tanh(t * self.De2.weight + r * self.Dr2.weight + self.bc2.weight)

    def g(self, f, w):
        """Defines activation layer.

            Args:
               f (Tensor): output of the forward layers.
               W (Tensor): Matrix for multiplication.
        """
        # [b, k] [k, tot_ent]
        return torch.sigmoid(torch.matmul(f, self.transpose(w)))

    def predict_tail_rank(self, h, r, topk=-1):
        emb_h = self.ent_embeddings(h)  # [1, k]
        emb_r = self.rel_embeddings(r)  # [1, k]
        
        hrt_sigmoid = -self.g(self.f1(emb_h, emb_r), self.ent_embeddings.weight)
        _, rank = torch.topk(hrt_sigmoid, k=topk)

        return rank

    def predict_head_rank(self, t, r, topk=-1):
        emb_t = self.ent_embeddings(t)  # [m, k]
        emb_r = self.rel_embeddings(r)  # [m, k]
        
        hrt_sigmoid = -self.g(self.f2(emb_t, emb_r), self.ent_embeddings.weight)
        _, rank = torch.topk(hrt_sigmoid, k=topk)

        return rank

    @staticmethod
    def transpose(tensor):
        dims = tuple(range(len(tensor.shape)-1, -1, -1))    # (rank-1...0)
        return tensor.permute(dims)
