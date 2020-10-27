import torch
import torch.nn as nn
import numpy as np

from pykg2vec.models.KGMeta import HyperbolicSpaceModel
from pykg2vec.models.Domain import NamedEmbedding


class MuRP(HyperbolicSpaceModel):
    """
       `Multi-relational Poincaré Graph Embeddings`_

       Args:
           config (object): Model configuration parameters.

       Examples:
           >>> from pykg2vec.models.hyperbolic import MuRP
           >>> from pykg2vec.utils.trainer import Trainer
           >>> model = MuRP()
           >>> trainer = Trainer(model=model)
           >>> trainer.build_model()
           >>> trainer.train_model()

       .. _ibalazevic: https://github.com/ibalazevic/multirelational-poincare.git

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
        self.ent_embeddings.weight.data = (1e-3 * torch.randn((self.tot_entity, k), dtype=torch.double, device=self.device))
        self.rel_embeddings = NamedEmbedding("rel_embedding", self.tot_relation, k, padding_idx=0)
        self.rel_embeddings.weight.data = (1e-3 * torch.randn((self.tot_relation, k), dtype=torch.double, device=self.device))
        self.wu = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.tot_relation, k)), dtype=torch.double, requires_grad=True, device=self.device))
        self.bs = nn.Parameter(torch.zeros(self.tot_entity, dtype=torch.double, requires_grad=True, device=self.device))
        self.bo = nn.Parameter(torch.zeros(self.tot_entity, dtype=torch.double, requires_grad=True, device=self.device))
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.parameter_list = [
            self.ent_embeddings,
            self.rel_embeddings,
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
        u = self.ent_embeddings.weight[h]
        v = self.ent_embeddings.weight[h]
        ru = self.wu[r]
        rvh = self.rel_embeddings.weight[r]

        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= 1, u / (torch.norm(u, 2, dim=-1, keepdim=True) - 1e-5), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= 1, v / (torch.norm(v, 2, dim=-1, keepdim=True) - 1e-5), v)
        rvh = torch.where(torch.norm(rvh, 2, dim=-1, keepdim=True) >= 1, rvh / (torch.norm(rvh, 2, dim=-1, keepdim=True) - 1e-5), rvh)
        u_e = self.p_log_map(u)
        u_w = u_e * ru
        u_m = self.p_exp_map(u_w)
        v_m = self.p_sum(v, rvh)
        u_m = torch.where(torch.norm(u_m, 2, dim=-1, keepdim=True) >= 1, u_m / (torch.norm(u_m, 2, dim=-1, keepdim=True) - 1e-5), u_m)
        v_m = torch.where(torch.norm(v_m, 2, dim=-1, keepdim=True) >= 1, v_m / (torch.norm(v_m, 2, dim=-1, keepdim=True) - 1e-5), v_m)

        sqdist = (2. * self.artanh(torch.clamp(torch.norm(self.p_sum(-u_m, v_m), 2), 1e-10, 1 - 1e-5))) ** 2

        # targets = np.ones(h.shape)
        # targets = torch.DoubleTensor(targets)
        # return self.loss(-sqdist + self.bs[h] + self.bo[t], targets)
        return -sqdist + self.bs[h] + self.bo[t]

    def forward_euclidean(self, h, r, t):
        u = self.ent_embeddings.weight[h]
        v = self.ent_embeddings.weight[t]
        ru = self.wu[r]
        rvh = self.rel_embeddings.weight[r]
        u_w = u * ru

        sqdist = torch.sum(torch.pow(u_w - (v + rvh), 2), dim=-1)
        return -sqdist + self.bs[h] + self.bo[t]

    @staticmethod
    def artanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    @staticmethod
    def p_exp_map(v):
        normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
        return torch.tanh(normv) * v / normv

    @staticmethod
    def p_log_map(v):
        normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1 - 1e-5)
        return MuRP.artanh(normv) * v / normv

    @staticmethod
    def p_sum(x, y):
        sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1 - 1e-5)
        sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1 - 1e-5)
        dotxy = torch.sum(x * y, dim=-1, keepdim=True)
        numerator = (1 + 2 * dotxy + sqynorm) * x + (1 - sqxnorm) * y
        denominator = 1 + 2 * dotxy + sqxnorm * sqynorm
        return numerator / denominator
