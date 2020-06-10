#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy


class ConvE(ModelMeta):
    """`Convolutional 2D Knowledge Graph Embeddings`_

    ConvE is a multi-layer convolutional network model for link prediction,
    it is a embedding model which is highly parameter efficient.

    Args:
        config (object): Model configuration parameters.
    
    Attributes:
        config (object): Model configuration.
        data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
        model (str): Name of the model.
        last_dim (int): The size of the last dimesion, depends on hidden size.

    
    Examples:
        >>> from pykg2vec.core.Complex import ConvE
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = ConvE()
        >>> trainer = Trainer(model=model)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Convolutional 2D Knowledge Graph Embeddings:
        https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17366/15884
    """

    def __init__(self, config):
        super(ConvE, self).__init__()
        self.config = config
        self.model_name = 'ConvE'
        self.training_strategy = TrainingStrategy.PROJECTION_BASED

        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        self.ent_embeddings = nn.Embedding(num_total_ent, k)
        # because conve considers the reciprocal relations,
        # so every rel should have its mirrored rev_rel in ConvE.
        self.rel_embeddings = nn.Embedding(num_total_rel*2, k)
        self.b = nn.Embedding(1, num_total_ent)

        self.bn0 = lambda x: nn.BatchNorm2d(x.shape[1])
        self.inp_drop = nn.Dropout(self.config.input_dropout)
        self.conv2d_1 = lambda x: nn.Conv2d(x.shape[1], 32, (3, 3), stride=(1, 1))
        self.bn1 = lambda x: nn.BatchNorm2d(x.shape[1])
        self.feat_drop = nn.Dropout2d(self.config.feature_map_dropout)
        self.fc1 = lambda x: nn.Linear(in_features=x.shape[1], out_features=self.config.hidden_size, bias=True)
        self.hidden_drop = nn.Dropout(self.config.hidden_dropout)
        self.bn2 = lambda x: nn.BatchNorm1d(x.shape[1])
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.b]

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
        st_inp = st_inp.permute(0, 3, 1, 2)   # convert to channel-first
        # batch normalization in the first axis
        x = self.bn0(st_inp)(st_inp)
        # input dropout
        x = self.inp_drop(x)
        # 2d convolution layer, output channel =32, kernel size = 3,3
        x = self.conv2d_1(x)(x)
        # batch normalization across feature dimension
        x = self.bn1(x)(x)
        # first non-linear activation
        x = torch.relu(x)
        # feature dropout
        x = self.feat_drop(x)
        # reshape the tensor to get the batch size
        '''10368 with k=200,5184 with k=100, 2592 with k=50'''
        x = x.view(first_dimension_size, -1)
        # pass the feature through fully connected layer, output size = batch size, hidden size
        x = self.fc1(x)(x)
        # dropout in the hidden layer
        x = self.hidden_drop(x)
        # batch normalization across feature dimension
        if(x.shape[0] > 1):
            x = self.bn2(x)(x)
        # second non-linear activation
        x = torch.relu(x)
        # project and get inner product with the tail triple
        x = torch.matmul(x, self.transpose(self.ent_embeddings.weight))    # [b, k] * [k, tot_ent]
        # add a bias value
        x = torch.add(x, self.b.weight)
        # sigmoid activation
        return torch.sigmoid(x)

    def forward(self, e, r, direction="tail"):
        if direction == "head":
            e_emb, r_emb = self.embed2(e, r + self.config.kg_meta.tot_relation)
        else:
            e_emb, r_emb = self.embed2(e, r)
        
        stacked_e = e_emb.  view(-1, self.config.hidden_size_2, self.config.hidden_size_1, 1)
        stacked_r = r_emb.view(-1, self.config.hidden_size_2, self.config.hidden_size_1, 1)
        stacked_er = torch.cat([stacked_e, stacked_r], 1)

        preds = self.inner_forward(stacked_er, list(e.shape)[0])
    
        return preds

    def predict_tail_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="tail"), k=topk)
        return rank

    def predict_head_rank(self, e, r, topk=-1):
        _, rank = torch.topk(-self.forward(e, r, direction="head"), k=topk)
        return rank

    def transpose(self, tensor):
        dims = tuple(range(len(tensor.shape)-1, -1, -1))    # (rank-1...0)
        return tensor.permute(dims)
