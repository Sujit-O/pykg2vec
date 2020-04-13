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
        self.training_strategy = TrainingStrategy.PROJECTION_BASED

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

    def forward(self, e, r, er_e2, direction=None):
        emb_e = tf.nn.embedding_lookup(self.ent_embeddings, e)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)

        return -tf.reduce_mean(tf.keras.backend.binary_crossentropy(er_e2, (emb_e * emb_r) @ tf.transpose(self.ent_embeddings)))

    def get_reg(self):
        return self.config.lmbda * (tf.reduce_sum(tf.reduce_sum(tf.abs(self.ent_embeddings) ** 3) + tf.reduce_sum(
            tf.abs(self.rel_embeddings) ** 3)))

    def predict_tail_rank(self, h, r, topk=-1):
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)

        candidates = -(emb_h * emb_r) @ tf.transpose(self.ent_embeddings)
        _, rank = tf.nn.top_k(candidates, k=topk)

        return rank

    def predict_head_rank(self, t, r, topk=-1):
        return self.predict_tail_rank(t, r, topk)
