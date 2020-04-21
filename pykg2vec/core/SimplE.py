from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy


class SimplE(ModelMeta):

    def __init__(self, config):
        super(SimplE, self).__init__()
        self.config = config
        self.model_name = 'SimplE_avg'
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               num_total_ent (int): Total number of entities.
               num_total_rel (int): Total number of relations.
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_head_embeddings  (Tensor Variable): Lookup variable containing embedding of the head entities.
               ent_tail_embeddings  (Tensor Variable): Lookup variable containing embedding of the tail relations.
               rel_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
               rel_inv_embeddings  (Tensor Variable): Lookup variable containing embedding of the inverse relations.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        emb_initializer = tf.initializers.glorot_normal()
        self.ent_head_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_head_embedding")
        self.ent_tail_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_tail_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embedding")
        self.rel_inv_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_inv_embedding")
        self.parameter_list = [self.ent_head_embeddings, self.ent_tail_embeddings, self.rel_embeddings, self.rel_inv_embeddings]


    def embed(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns head, relation and tail embedding Tensors.
        """
        emb_h1 = tf.nn.embedding_lookup(self.ent_head_embeddings, h)
        emb_h2 = tf.nn.embedding_lookup(self.ent_head_embeddings, t)
        emb_r1 = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_r2 = tf.nn.embedding_lookup(self.rel_inv_embeddings, r)
        emb_t1 = tf.nn.embedding_lookup(self.ent_tail_embeddings, t)
        emb_t2 = tf.nn.embedding_lookup(self.ent_tail_embeddings, h)
        return emb_h1, emb_h2, emb_r1, emb_r2, emb_t1, emb_t2

    def forward(self, h, r, t):
        h1_e, h2_e, r1_e, r2_e, t1_e, t2_e = self.embed(h, r, t)

        init = tf.reduce_sum(h1_e*r1_e*t1_e, 1) + tf.reduce_sum(h2_e*r2_e*t2_e, 1) / 2.0
        return -tf.clip_by_value(init, -20, 20)

    def get_reg(self, h, r, t):
        num_batch = math.ceil(self.config.kg_meta.tot_train_triples / self.config.batch_size)
        regul_term = (tf.nn.l2_loss(self.ent_head_embeddings) + tf.nn.l2_loss(self.ent_tail_embeddings) +
                      tf.nn.l2_loss(self.rel_embeddings) + tf.nn.l2_loss(self.rel_inv_embeddings)) / num_batch**2
        return self.config.lmbda * regul_term


class SimplE_ignr(SimplE):

    def __init__(self, config):
        super(SimplE, self).__init__()
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

        emb_h = tf.concat([tf.gather(self.ent_head_embeddings, h), tf.gather(self.ent_head_embeddings, t)], 1)
        emb_r = tf.concat([tf.gather(self.rel_embeddings, r), tf.gather(self.rel_inv_embeddings, r)], 1)
        emb_t = tf.concat([tf.gather(self.ent_tail_embeddings, t), tf.gather(self.ent_tail_embeddings, h)], 1)

        return emb_h, emb_r, emb_t

    def forward(self, h, r, t):
        h_e, r_e, t_e = self.embed(h, r, t)

        init = tf.reduce_sum(h_e*r_e*t_e, 1)
        return -tf.clip_by_value(init, -20, 20)

    def get_reg(self, h, r, t):
        return 2.0 * super().get_reg(h, r, t)


