#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.KGMeta import ModelMeta, InferenceMeta


class ProjE_pointwise(ModelMeta, InferenceMeta):
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
            >>> trainer = Trainer(model=model, debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()

        .. _ProjE-Embedding Projection for Knowledge Graph Completion:
            https://arxiv.org/abs/1611.05425

    """

    def __init__(self, config):
        self.config = config
        self.model_name = 'ProjE_pointwise'

    # Override
    def dissimilarity(self, h, r, t):
        """Function to calculate dissimilarity measure in embedding space.

        Args:
            h (Tensor): Head entities ids.
            r (Tensor): Relation ids of the triple.
            t (Tensor): Tail entity ids of the triple.
            axis (int): Determines the axis for reduction

        Returns:
            Tensors: Returns the dissimilarity measure.
        """
        if self.config.L1_flag:
            return tf.reduce_sum(tf.abs(h + r - t), axis=1)  # L1 norm
        else:
            return tf.reduce_sum((h + r - t) ** 2, axis=1)  # L2 norm

    def def_parameters(self):
        """Defines the model parameters.

           Attributes:
               k (Tensor): Size of the latent dimesnion for entities and relations.
               ent_embeddings  (Tensor Variable): Lookup variable containing embedding of the entities.
               rel_embeddings  (Tensor Variable): Lookup variable containing embedding of the relations.
               parameter_list  (list): List of Tensor parameters.
        """
        num_total_ent = self.config.kg_meta.tot_entity
        num_total_rel = self.config.kg_meta.tot_relation
        k = self.config.hidden_size

        emb_initializer = tf.initializers.glorot_normal()

        self.ent_embeddings = tf.Variable(emb_initializer(shape=(num_total_ent, k)), name="ent_embedding")
        self.rel_embeddings = tf.Variable(emb_initializer(shape=(num_total_rel, k)), name="rel_embedding")
        self.bc1 = tf.Variable(emb_initializer(shape=(1, k)), name="bc1")
        self.De1 = tf.Variable(emb_initializer(shape=(1, k)), name="De1")
        self.Dr1 = tf.Variable(emb_initializer(shape=(1, k)), name="Dr1")
        self.bc2 = tf.Variable(emb_initializer(shape=(1, k)), name="bc2")
        self.De2 = tf.Variable(emb_initializer(shape=(1, k)), name="De2")
        self.Dr2 = tf.Variable(emb_initializer(shape=(1, k)), name="Dr2")

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings, self.bc1, self.De1, self.Dr1, self.bc2, self.De2, self.Dr2]

    def get_loss(self, h, r, t, hr_t, tr_h):
        """Defines the loss function for the algorithm."""
        hrt_loss = self.forward(h, r, tf.cast(tf.sparse.to_dense(tf.sparse.reorder(hr_t)), dtype=tf.float32))
        trh_loss = self.backward(t, r, tf.cast(tf.sparse.to_dense(tf.sparse.reorder(tr_h)), dtype=tf.float32))

        regularizer_loss = tf.reduce_sum(tf.abs(self.De1) + tf.abs(self.Dr1)) + tf.reduce_sum(tf.abs(self.De2) + tf.abs(self.Dr2)) + tf.reduce_sum(tf.abs(self.ent_embeddings)) + tf.reduce_sum(tf.abs(self.rel_embeddings))
        
        loss = hrt_loss + trh_loss + regularizer_loss*self.config.lmbda
        return loss

    def forward(self, h, r, hr_t):
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, -1)  # [tot_ent, k]
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, -1)  # [tot_rel, k]

        emb_hr_h = tf.nn.embedding_lookup(norm_ent_embeddings, h)  # [m, k]
        emb_hr_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)  # [m, k]
        
        hrt_sigmoid = self.g(tf.nn.dropout(self.f1(emb_hr_h, emb_hr_r), 0.5), norm_ent_embeddings)
        
        hrt_loss_left = - tf.reduce_sum((tf.math.log(tf.clip_by_value(hrt_sigmoid, 1e-10, 1.0)) * tf.maximum(0., hr_t)))
        hrt_loss_right = - tf.reduce_sum((tf.math.log(tf.clip_by_value(1 - hrt_sigmoid, 1e-10, 1.0)) * tf.maximum(0., tf.negative(hr_t))))

        hrt_loss = hrt_loss_left + hrt_loss_right

        return hrt_loss

    def backward(self, h, r, hr_t):
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, -1)  # [tot_ent, k]
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, -1)  # [tot_rel, k]

        emb_hr_h = tf.nn.embedding_lookup(norm_ent_embeddings, h)  # [m, k]
        emb_hr_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)  # [m, k]
        
        hrt_sigmoid = self.g(tf.nn.dropout(self.f2(emb_hr_h, emb_hr_r), 0.5), norm_ent_embeddings)
        
        hrt_loss_left = - tf.reduce_sum((tf.math.log(tf.clip_by_value(hrt_sigmoid, 1e-10, 1.0)) * tf.maximum(0., hr_t)))
        hrt_loss_right = - tf.reduce_sum((tf.math.log(tf.clip_by_value(1 - hrt_sigmoid, 1e-10, 1.0)) * tf.maximum(0., tf.negative(hr_t))))

        hrt_loss = hrt_loss_left + hrt_loss_right

        return hrt_loss

    def f1(self, h, r):
        """Defines froward layer for head.

            Args:
                   h (Tensor): Head entities ids.
                   r (Tensor): Relation ids of the triple.
        """
        return tf.tanh(h * self.De1 + r * self.Dr1 + self.bc1)

    def f2(self, t, r):
        """Defines forward layer for tail.

            Args:
               t (Tensor): Tail entities ids.
               r (Tensor): Relation ids of the triple.
        """
        return tf.tanh(t * self.De2 + r * self.Dr2 + self.bc2)

    def g(self, f, W):
        """Defines activation layer.

            Args:
               f (Tensor): output of the forward layers.
               W (Tensor): Matrix for multiplication.
        """
        # [b, k] [k, tot_ent]
        return tf.sigmoid(tf.matmul(f, W, transpose_b=True))

    def predict_tail(self, e, r, topk=-1):
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, -1)  # [tot_ent, k]
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, -1)  # [tot_rel, k]

        emb_hr_e = tf.nn.embedding_lookup(norm_ent_embeddings, e)  # [m, k]
        emb_hr_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)  # [m, k]
        
        hrt_sigmoid = -self.g(self.f1(emb_hr_e, emb_hr_r), norm_ent_embeddings)
        _, rank = tf.nn.top_k(hrt_sigmoid, k=topk)

        return rank

    def predict_head(self, e, r, topk=-1):
        norm_ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, -1)  # [tot_ent, k]
        norm_rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, -1)  # [tot_rel, k]

        emb_hr_e = tf.nn.embedding_lookup(norm_ent_embeddings, e)  # [m, k]
        emb_hr_r = tf.nn.embedding_lookup(norm_rel_embeddings, r)  # [m, k]
        
        hrt_sigmoid = -self.g(self.f2(emb_hr_e, emb_hr_r), norm_ent_embeddings)
        _, rank = tf.nn.top_k(hrt_sigmoid, k=topk)

        return rank