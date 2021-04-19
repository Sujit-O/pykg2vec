#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pykg2vec.models.KGMeta import ProjectionModel
from pykg2vec.models.Domain import NamedEmbedding
from pykg2vec.utils.criterion import Criterion


class ConvE(ProjectionModel):
    """
        `Convolutional 2D Knowledge Graph Embeddings`_ (ConvE) is a multi-layer convolutional network model for link prediction,
        it is a embedding model which is highly parameter efficient.
        ConvE is the first non-linear model that uses a global 2D convolution operation on the combined and head entity and relation embedding vectors. The obtained feature maps are made flattened and then transformed through a fully connected layer. The projected target vector is then computed by performing linear transformation (passing through the fully connected layer) and activation function, and finally an inner product with the latent representation of every entities.

        Args:
            config (object): Model configuration parameters.

        .. _Convolutional 2D Knowledge Graph Embeddings:
            https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17366/15884

    """

    def __init__(self, **kwargs):
        super(ConvE, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "hidden_size_1",
                      "lmbda", "input_dropout", "feature_map_dropout", "hidden_dropout"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        self.hidden_size_2 = self.hidden_size // self.hidden_size_1

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size

        self.ent_embeddings = NamedEmbedding("ent_embedding", num_total_ent, k)

        # because conve considers the reciprocal relations,
        # so every rel should have its mirrored rev_rel in ConvE.
        self.rel_embeddings = NamedEmbedding("rel_embedding", num_total_rel*2, k)

        self.b = NamedEmbedding("b", 1, num_total_ent)

        self.bn0 = nn.BatchNorm2d(1)
        self.inp_drop = nn.Dropout(self.input_dropout)
        self.conv2d_1 = nn.Conv2d(1, 32, (3, 3), stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.feat_drop = nn.Dropout2d(self.feature_map_dropout)
        self.fc = nn.Linear((2*self.hidden_size_2-3+1)*(self.hidden_size_1-3+1)*32, k) # use the conv output shape * out_channel
        self.hidden_drop = nn.Dropout(self.hidden_dropout)
        self.bn2 = nn.BatchNorm1d(k)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
            self.b,
        ]

        self.loss = Criterion.multi_class_bce

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

    def embed2(self, e, r):
        emb_e = self.ent_embeddings(e)
        emb_r = self.rel_embeddings(r)
        return emb_e, emb_r

    def inner_forward(self, st_inp, first_dimension_size):
        """Implements the forward pass layers of the algorithm."""
        x = self.bn0(st_inp) # 2d batch norm over feature dimension.
        x = self.inp_drop(x) # [b, 1, 2*hidden_size_2, hidden_size_1]
        x = self.conv2d_1(x) # [b, 32, 2*hidden_size_2-3+1, hidden_size_1-3+1]
        x = self.bn1(x) # 2d batch normalization across feature dimension
        x = torch.relu(x)
        x = self.feat_drop(x)
        x = x.view(first_dimension_size, -1) # flatten => [b, 32*(2*hidden_size_2-3+1)*(hidden_size_1-3+1)
        x = self.fc(x) # dense layer => [b, k]
        x = self.hidden_drop(x)
        if self.training:
            x = self.bn2(x) # batch normalization across the last axis
        x = torch.relu(x)
        x = torch.matmul(x, self.ent_embeddings.weight.T) # [b, k] * [k, tot_ent] => [b, tot_ent]
        x = torch.add(x, self.b.weight) # add a bias value
        return torch.sigmoid(x) # sigmoid activation

    def forward(self, e, r, direction="tail"):
        assert direction in ("head", "tail"), "Unknown forward direction"
        if direction == "head":
            e_emb, r_emb = self.embed2(e, r + self.tot_relation)
        else:
            e_emb, r_emb = self.embed2(e, r)

        stacked_e = e_emb.view(-1, 1, self.hidden_size_2, self.hidden_size_1)
        stacked_r = r_emb.view(-1, 1, self.hidden_size_2, self.hidden_size_1)
        stacked_er = torch.cat([stacked_e, stacked_r], 2)

        preds = self.inner_forward(stacked_er, list(e.shape)[0])

        return preds

    def predict_tail_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="tail"), k=topk)
        return rank

    def predict_head_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="head"), k=topk)
        return rank


class ProjE_pointwise(ProjectionModel):
    """
        `ProjE-Embedding Projection for Knowledge Graph Completion`_. (ProjE) Instead of measuring the distance or matching scores between the pair of the
        head entity and relation and then tail entity in embedding space ((h,r) vs (t)).
        ProjE projects the entity candidates onto a target vector representing the
        input data. The loss in ProjE is computed by the cross-entropy between
        the projected target vector and binary label vector, where the included
        entities will have value 0 if in negative sample set and value 1 if in
        positive sample set.
        Instead of measuring the distance or matching scores between the pair of the head entity and relation and then tail entity in embedding space ((h,r) vs (t)). ProjE projects the entity candidates onto a target vector representing the input data. The loss in ProjE is computed by the cross-entropy between the projected target vector and binary label vector, where the included entities will have value 0 if in negative sample set and value 1 if in positive sample set.


        Args:
            config (object): Model configuration parameters.

        .. _ProjE-Embedding Projection for Knowledge Graph Completion:
            https://arxiv.org/abs/1611.05425

    """

    def __init__(self, **kwargs):
        super(ProjE_pointwise, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "lmbda", "hidden_dropout"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size
        self.device = kwargs["device"]

        self.ent_embeddings = NamedEmbedding("ent_embedding", num_total_ent, k)
        self.rel_embeddings = NamedEmbedding("rel_embedding", num_total_rel, k)
        self.bc1 = NamedEmbedding("bc1", 1, k)
        self.De1 = NamedEmbedding("De1", 1, k)
        self.Dr1 = NamedEmbedding("Dr1", 1, k)
        self.bc2 = NamedEmbedding("bc2", 1, k)
        self.De2 = NamedEmbedding("De2", 1, k)
        self.Dr2 = NamedEmbedding("Dr2", 1, k)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.bc1.weight)
        nn.init.xavier_uniform_(self.De1.weight)
        nn.init.xavier_uniform_(self.Dr1.weight)
        nn.init.xavier_uniform_(self.bc2.weight)
        nn.init.xavier_uniform_(self.De2.weight)
        nn.init.xavier_uniform_(self.Dr2.weight)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
            self.bc1,
            self.De1,
            self.Dr1,
            self.bc2,
            self.De2,
            self.Dr2,
        ]

        self.loss = Criterion.multi_class

    def get_reg(self, h, r, t):
        return self.lmbda*(torch.sum(torch.abs(self.De1.weight) + torch.abs(self.Dr1.weight)) +
                           torch.sum(torch.abs(self.De2.weight) + torch.abs(self.Dr2.weight)) +
                           torch.sum(torch.abs(self.ent_embeddings.weight)) + torch.sum(torch.abs(self.rel_embeddings.weight)))

    def forward(self, e, r, er_e2, direction="tail"):
        assert direction in ("head", "tail"), "Unknown forward direction"

        emb_hr_e = self.ent_embeddings(e)  # [m, k]
        emb_hr_r = self.rel_embeddings(r)  # [m, k]

        if direction == "tail":
            ere2_sigmoid = ProjE_pointwise.g(torch.dropout(self.f1(emb_hr_e, emb_hr_r), p=self.hidden_dropout, train=True), self.ent_embeddings.weight)
        else:
            ere2_sigmoid = ProjE_pointwise.g(torch.dropout(self.f2(emb_hr_e, emb_hr_r), p=self.hidden_dropout, train=True), self.ent_embeddings.weight)

        ere2_loss_left = -torch.sum((torch.log(torch.clamp(ere2_sigmoid, 1e-10, 1.0)) * torch.max(torch.FloatTensor([0]).to(self.device), er_e2)))
        ere2_loss_right = -torch.sum((torch.log(torch.clamp(1 - ere2_sigmoid, 1e-10, 1.0)) * torch.max(torch.FloatTensor([0]).to(self.device), torch.neg(er_e2))))

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

    def predict_tail_rank(self, h, r, topk=-1):
        emb_h = self.ent_embeddings(h)  # [1, k]
        emb_r = self.rel_embeddings(r)  # [1, k]

        hrt_sigmoid = -ProjE_pointwise.g(self.f1(emb_h, emb_r), self.ent_embeddings.weight)
        _, rank = torch.topk(hrt_sigmoid, k=topk)

        return rank

    def predict_head_rank(self, t, r, topk=-1):
        emb_t = self.ent_embeddings(t)  # [m, k]
        emb_r = self.rel_embeddings(r)  # [m, k]

        hrt_sigmoid = -ProjE_pointwise.g(self.f2(emb_t, emb_r), self.ent_embeddings.weight)
        _, rank = torch.topk(hrt_sigmoid, k=topk)

        return rank

    @staticmethod
    def g(f, w):
        """Defines activation layer.

            Args:
               f (Tensor): output of the forward layers.
               w (Tensor): Matrix for multiplication.
        """
        # [b, k] [k, tot_ent]
        return torch.sigmoid(torch.matmul(f, w.T))

class TuckER(ProjectionModel):
    """
        `TuckER-Tensor Factorization for Knowledge Graph Completion`_ (TuckER)
        is a Tensor-factorization-based embedding technique based on
        the Tucker decomposition of a third-order binary tensor of triplets. Although
        being fully expressive, the number of parameters used in Tucker only grows linearly
        with respect to embedding dimension as the number of entities or relations in a
        knowledge graph increases.
        TuckER is a Tensor-factorization-based embedding technique based on the Tucker decomposition of a third-order binary tensor of triplets. Although being fully expressive, the number of parameters used in Tucker only grows linearly with respect to embedding dimension as the number of entities or relations in a knowledge graph increases. The author also showed in paper that the models, such as RESCAL, DistMult, ComplEx, are all special case of TuckER.


        Args:
            config (object): Model configuration parameters.

        .. _TuckER-Tensor Factorization for Knowledge Graph Completion:
            https://arxiv.org/pdf/1901.09590.pdf

    """

    def __init__(self, **kwargs):
        super(TuckER, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "ent_hidden_size",
                      "rel_hidden_size", "lmbda", "input_dropout",
                      "hidden_dropout1", "hidden_dropout2"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        self.d1 = self.ent_hidden_size
        self.d2 = self.rel_hidden_size

        self.ent_embeddings = NamedEmbedding("ent_embedding", num_total_ent, self.d1)
        self.rel_embeddings = NamedEmbedding("rel_embedding", num_total_rel, self.d2)
        self.W = NamedEmbedding("W", self.d2, self.d1 * self.d1)
        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.W.weight)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
            self.W,
        ]

        self.inp_drop = nn.Dropout(self.input_dropout)
        self.hidden_dropout1 = nn.Dropout(self.hidden_dropout1)
        self.hidden_dropout2 = nn.Dropout(self.hidden_dropout2)

        self.loss = Criterion.multi_class_bce

    def forward(self, e1, r, direction="head"):
        """Implementation of the layer.

            Args:
                e1(Tensor): entities id.
                r(Tensor): Relation id.

            Returns:
                Tensors: Returns the activation values.
        """
        assert direction in ("head", "tail"), "Unknown forward direction"
        e1 = self.ent_embeddings(e1)
        e1 = F.normalize(e1, p=2, dim=1)
        e1 = self.inp_drop(e1)
        e1 = e1.view(-1, 1, self.d1)

        rel = self.rel_embeddings(r)
        W_mat = torch.matmul(rel, self.W.weight.view(self.d2, -1))
        W_mat = W_mat.view(-1, self.d1, self.d1)
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.matmul(e1, W_mat)
        x = x.view(-1, self.d1)
        x = F.normalize(x, p=2, dim=1)
        x = self.hidden_dropout2(x)
        x = torch.matmul(x, self.ent_embeddings.weight.T)
        return F.sigmoid(x)

    def predict_tail_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="tail"), k=topk)
        return rank

    def predict_head_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="head"), k=topk)
        return rank


class InteractE(ProjectionModel):
    """
       `InteractE\: Improving Convolution-based Knowledge Graph Embeddings by Increasing Feature Interactions`_

       Args:
           config (object): Model configuration parameters.

       .. _InteractE\: Improving Convolution-based Knowledge Graph Embeddings by Increasing Feature Interactions:
            https://arxiv.org/abs/1911.00219

    """

    def __init__(self, **kwargs):
        super(InteractE, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "input_dropout", "hidden_dropout", "feature_map_dropout",
                      "feature_permutation", "num_filters", "kernel_size", "reshape_height", "reshape_width"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        self.hidden_size = self.reshape_width * self.reshape_height
        self.device = kwargs["device"]

        self.ent_embeddings = NamedEmbedding("ent_embeddings", self.tot_entity, self.hidden_size, padding_idx=None)
        self.rel_embeddings = NamedEmbedding("rel_embeddings", self.tot_relation, self.hidden_size, padding_idx=None)
        self.bceloss = nn.BCELoss()

        self.inp_drop = nn.Dropout(self.input_dropout)
        self.hidden_drop = nn.Dropout(self.hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(self.feature_map_dropout)
        self.bn0 = nn.BatchNorm2d(self.feature_permutation)

        flat_sz_h = self.reshape_height
        flat_sz_w = 2 * self.reshape_width
        self.padding = 0

        self.bn1 = nn.BatchNorm2d(self.num_filters * self.feature_permutation)
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filters * self.feature_permutation

        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.fc = nn.Linear(self.flat_sz, self.hidden_size)
        self.chequer_perm = self._get_chequer_perm()

        self.register_parameter("bias", nn.Parameter(torch.zeros(self.tot_entity)))
        self.register_parameter("conv_filt", nn.Parameter(torch.zeros(self.num_filters, 1, self.kernel_size, self.kernel_size)))

        nn.init.xavier_uniform_(self.ent_embeddings.weight)
        nn.init.xavier_uniform_(self.rel_embeddings.weight)
        nn.init.xavier_uniform_(self.conv_filt)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
        ]

        self.loss = Criterion.multi_class_bce

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

    def embed2(self, e, r):
        emb_e = self.ent_embeddings(e)
        emb_r = self.rel_embeddings(r)
        return emb_e, emb_r

    def forward(self, e, r, direction="tail"):
        assert direction in ("head", "tail"), "Unknown forward direction"
        emb_e, emb_r = self.embed2(e, r)
        emb_comb = torch.cat([emb_e, emb_r], dim=-1)
        chequer_perm = emb_comb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.feature_permutation, 2 * self.reshape_width, self.reshape_height))
        stack_inp = self.bn0(stack_inp)
        x = self.inp_drop(stack_inp)
        x = InteractE._circular_padding_chw(x, self.kernel_size // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.feature_permutation, 1, 1, 1), padding=self.padding, groups=self.feature_permutation)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        x += self.bias.expand_as(x)

        return torch.sigmoid(x)

    def predict_tail_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="tail"), k=topk)
        return rank

    def predict_head_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="head"), k=topk)
        return rank

    @staticmethod
    def _circular_padding_chw(batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def _get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.hidden_size) for _ in range(self.feature_permutation)])
        rel_perm = np.int32([np.random.permutation(self.hidden_size) for _ in range(self.feature_permutation)])

        comb_idx = []
        for k in range(self.feature_permutation):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.reshape_height):
                for _ in range(self.reshape_width):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.hidden_size)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.hidden_size)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.hidden_size)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.hidden_size)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm


class HypER(ProjectionModel):
    """
       `HypER\: Hypernetwork Knowledge Graph Embeddings`_

       Args:
           config (object): Model configuration parameters.

       .. _HypER\: Hypernetwork Knowledge Graph Embeddings:
            https://arxiv.org/abs/1808.07018

    """

    def __init__(self, **kwargs):
        super(HypER, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "ent_hidden_size", "rel_hidden_size", "input_dropout", "hidden_dropout", "feature_map_dropout"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)
        self.device = kwargs["device"]
        self.filt_h = 1
        self.filt_w = 9
        self.in_channels = 1
        self.out_channels = 32

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation

        self.ent_embeddings = NamedEmbedding("ent_embeddings", num_total_ent, self.ent_hidden_size, padding_idx=0)
        self.rel_embeddings = NamedEmbedding("rel_embeddings", num_total_rel, self.rel_hidden_size, padding_idx=0)
        self.inp_drop = nn.Dropout(self.input_dropout)
        self.hidden_drop = nn.Dropout(self.hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(self.feature_map_dropout)

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.ent_hidden_size)
        self.register_parameter("b", nn.Parameter(torch.zeros(num_total_ent)))
        fc_length = (1 - self.filt_h + 1) * (self.ent_hidden_size - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, self.ent_hidden_size)
        fc1_length = self.in_channels * self.out_channels * self.filt_h * self.filt_w
        self.fc1 = torch.nn.Linear(self.rel_hidden_size, fc1_length)

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
        ]

        self.loss = Criterion.multi_class_bce

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

    def embed2(self, e, r):
        emb_e = self.ent_embeddings(e)
        emb_r = self.rel_embeddings(r)
        return emb_e, emb_r

    def forward(self, e, r, direction="tail"):
        assert direction in ("head", "tail"), "Unknown forward direction"
        e_emb, r_emb = self.embed2(e, r)
        e_emb = e_emb.view(-1, 1, 1, self.ent_embeddings.weight.size(1))
        x = self.bn0(e_emb)
        x = self.inp_drop(x)

        k = self.fc1(r_emb)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e_emb.size(0) * self.in_channels * self.out_channels, 1, self.filt_h, self.filt_w)

        x = x.permute(1, 0, 2, 3)

        x = F.conv2d(x, k, groups=e_emb.size(0))
        x = x.view(e_emb.size(0), 1, self.out_channels, 1 - self.filt_h + 1, e_emb.size(3) - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e_emb.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)
        return pred

    def predict_tail_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="tail"), k=topk)
        return rank

    def predict_head_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="head"), k=topk)
        return rank


class AcrE(ProjectionModel):
    """
       `Knowledge Graph Embedding with Atrous Convolution and Residual Learning`_

       Args:
           config (object): Model configuration parameters.

       .. _Knowledge Graph Embedding with Atrous Convolution and Residual Learning:
            https://arxiv.org/abs/2010.12121

    """

    def __init__(self, **kwargs):
        super(AcrE, self).__init__(self.__class__.__name__.lower())
        param_list = ["tot_entity", "tot_relation", "hidden_size", "input_dropout", "hidden_dropout", "feature_map_dropout",
                      "in_channels", "way", "first_atrous", "second_atrous", "third_atrous", "acre_bias"]
        param_dict = self.load_params(param_list, kwargs)
        self.__dict__.update(param_dict)

        num_total_ent = self.tot_entity
        num_total_rel = self.tot_relation
        k = self.hidden_size

        self.ent_embeddings = NamedEmbedding("ent_embedding", num_total_ent, k, padding_idx=None)
        self.rel_embeddings = NamedEmbedding("rel_embedding", num_total_rel * 2, k, padding_idx=None)

        self.inp_drop = torch.nn.Dropout(self.input_dropout)
        self.hidden_drop = torch.nn.Dropout(self.hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout2d(self.feature_map_dropout)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn2 = torch.nn.BatchNorm1d(k)
        self.fc = torch.nn.Linear(self.in_channels * 400, k)
        self.padding = 0

        if self.way == "serial":
            self.conv1 = torch.nn.Conv2d(1, self.in_channels, (3, 3), 1, self.first_atrous, bias=self.acre_bias,
                                         dilation=self.first_atrous)
            self.conv2 = torch.nn.Conv2d(self.in_channels, self.in_channels, (3, 3), 1, self.second_atrous,
                                         bias=self.acre_bias, dilation=self.second_atrous)
            self.conv3 = torch.nn.Conv2d(self.in_channels, self.in_channels, (3, 3), 1, self.third_atrous, bias=self.acre_bias,
                                         dilation=self.third_atrous)
        else:
            self.conv1 = torch.nn.Conv2d(1, self.in_channels, (3, 3), 1, self.first_atrous, bias=self.acre_bias,
                                         dilation=self.first_atrous)
            self.conv2 = torch.nn.Conv2d(1, self.in_channels, (3, 3), 1, self.second_atrous, bias=self.acre_bias,
                                         dilation=self.second_atrous)
            self.conv3 = torch.nn.Conv2d(1, self.in_channels, (3, 3), 1, self.third_atrous, bias=self.acre_bias,
                                         dilation=self.third_atrous)
            self.W_gate_e = torch.nn.Linear(1600, 400)

        self.register_parameter("bias", nn.Parameter(torch.zeros(num_total_ent)))

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
        ]

        self.loss = Criterion.multi_class_bce

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

    def embed2(self, e, r):
        emb_e = self.ent_embeddings(e)
        emb_r = self.rel_embeddings(r)
        return emb_e, emb_r

    def forward(self, e, r, direction="tail"):
        assert direction in ("head", "tail"), "Unknown forward direction"
        emb_e, emb_r = self.embed2(e, r)
        sub_emb = emb_e.view(-1, 1, 10, 20)
        rel_emb = emb_r.view(-1, 1, 10, 20)
        comb_emb = torch.cat([sub_emb, rel_emb], dim=2)
        stack_inp = self.bn0(comb_emb)
        x = self.inp_drop(stack_inp)
        res = x
        if self.way == "serial":
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = x + res
        else:
            conv1 = self.conv1(x).view(-1, self.in_channels, 400)
            conv2 = self.conv2(x).view(-1, self.in_channels, 400)
            conv3 = self.conv3(x).view(-1, self.in_channels, 400)
            res = res.expand(-1, self.in_channels, 20, 20).view(-1, self.in_channels, 400)
            x = torch.cat((res, conv1, conv2, conv3), dim=2)
            x = self.W_gate_e(x).view(-1, self.in_channels, 20, 20)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.ent_embeddings.weight.transpose(1, 0))
        x += self.bias.expand_as(x)

        return torch.sigmoid(x)

    def predict_tail_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="tail"), k=topk)
        return rank

    def predict_head_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="head"), k=topk)
        return rank
