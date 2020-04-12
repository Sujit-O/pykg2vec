from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy


class CP(ModelMeta):

    def __init__(self, config):
        super(CP, self).__init__()
        self.config = config
        self.model_name = 'CP'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing embedding of the relations.
               b  (Tensor Variable): Variable storing the bias values.
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
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_emb, r_emb, t_emb = self.embed(h, r, t)
        return -tf.reduce_sum(h_emb * r_emb * t_emb, -1)

    def get_reg(self, h, r, t):
        h_emb, r_emb, t_emb = self.embed(h, r, t)
        regul_term = tf.reduce_mean(tf.reduce_sum(h_emb**2, -1) + tf.reduce_sum(r_emb**2, -1) + tf.reduce_sum(t_emb**2, -1))
        return self.config.lmbda*regul_term