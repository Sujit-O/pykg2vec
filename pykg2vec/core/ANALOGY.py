from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta
from pykg2vec.utils.generator import TrainingStrategy


class ANALOGY(ModelMeta):

    def __init__(self, config):
        super(ANALOGY, self).__init__()
        self.config = config
        self.model_name = 'ANALOGY'
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
        self.ent_embeddings_real = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="emb_e_real")
        self.ent_embeddings_img  = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="emb_e_img")
        self.rel_embeddings_real = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="emb_rel_real")
        self.rel_embeddings_img  = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="emb_rel_img")
        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.ent_embeddings_real, self.ent_embeddings_img, self.rel_embeddings_real, self.rel_embeddings_img]

    def embed1(self, h, r, t):
        """Function to get the embedding value.

           Args:
               h (Tensor): Head entities ids.
               r (Tensor): Relation ids of the triple.
               t (Tensor): Tail entity ids of the triple.

            Returns:
                Tensors: Returns real and imaginary values of head, relation and tail embedding.
        """
        h_emb_real = tf.nn.embedding_lookup(self.ent_embeddings_real, h)
        h_emb_img = tf.nn.embedding_lookup(self.ent_embeddings_img, h)

        r_emb_real = tf.nn.embedding_lookup(self.rel_embeddings_real, r)
        r_emb_img = tf.nn.embedding_lookup(self.rel_embeddings_img, r)

        t_emb_real = tf.nn.embedding_lookup(self.ent_embeddings_real, t)
        t_emb_img = tf.nn.embedding_lookup(self.ent_embeddings_img, t)

        return h_emb_real, h_emb_img, r_emb_real, r_emb_img, t_emb_real, t_emb_img

    def embed2(self, h, r, t):
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
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed1(h, r, t)
        h_e, r_e, t_e = self.embed2(h, r, t)
        return -tf.reduce_sum(
            h_e_real * t_e_real * r_e_real + h_e_img * t_e_img * r_e_real + h_e_real * t_e_img * r_e_img - h_e_img * t_e_real * r_e_img, -1) \
            + -tf.reduce_sum(h_e * r_e * t_e, -1)

    # TODO: double check if we need the regularizer here
    def get_reg(self, h, r, t):
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed1(h, r, t)
        h_e, r_e, t_e = self.embed2(h, r, t)

        regul_term = tf.reduce_mean(tf.reduce_sum(h_e_real**2, -1) + tf.reduce_sum(h_e_img**2, -1) + tf.reduce_sum(r_e_real**2,-1)
                                    + tf.reduce_sum(r_e_img**2, -1) + tf.reduce_sum(t_e_real**2, -1) + tf.reduce_sum(t_e_img**2, -1)
                                    + tf.reduce_sum(h_e**2, -1) + tf.reduce_sum(r_e**2, -1) + tf.reduce_sum(t_e**2,-1))
        return self.config.lmbda*regul_term
