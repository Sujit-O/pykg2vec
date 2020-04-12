#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy

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
            >>> from pykg2vec.core.Complex import DistMult
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

    def def_parameters(self):
        """Defines the model parameters.
           
           Attributes:
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings(Tensor Variable): Lookup variable containing embedding of entities.
               rel_embeddings (Tensor Variable): Lookup variable containing embedding of relations.
               parameter_list  (list): List of Tensor parameters. 
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size
        
        emb_initializer = tf.initializers.glorot_normal()

        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embedding")
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings]

    def embed(self, h, r, t):
        """Function to get the embedding value.
           
           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        h_emb = tf.nn.embedding_lookup(self.ent_embeddings, h)
        r_emb = tf.nn.embedding_lookup(self.rel_embeddings, r)
        t_emb = tf.nn.embedding_lookup(self.ent_embeddings, t)

        return h_emb, r_emb, t_emb

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        h_e_new = tf.concat([h_e, t_e], 1)
        r_e_new = tf.concat([r_e, r_e], 1)
        t_e_new = tf.concat([t_e, h_e], 1)
        return -tf.reduce_sum(h_e_new*r_e_new*t_e_new, -1)

    def get_reg(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)
        regul_term = tf.reduce_mean(tf.reduce_sum(h_e**2, -1) + tf.reduce_sum(r_e**2, -1) + tf.reduce_sum(t_e**2,-1))
        return self.config.lmbda*regul_term