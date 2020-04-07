#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from pykg2vec.core.KGMeta import ModelMeta
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

        self.parameter_list = [self.ent_embeddings, self.rel_embeddings,
                               self.bc1, self.De1, self.Dr1,
                               self.bc2, self.De2, self.Dr2]

    def get_reg(self):
        return self.config.lmbda*(tf.reduce_sum(tf.abs(self.De1) + tf.abs(self.Dr1)) + tf.reduce_sum(tf.abs(self.De2) + tf.abs(self.Dr2)) + tf.reduce_sum(tf.abs(self.ent_embeddings)) + tf.reduce_sum(tf.abs(self.rel_embeddings)))

    def forward(self, e, r, er_e2, direction="tail"):
        emb_hr_e = tf.nn.embedding_lookup(self.ent_embeddings, e)  # [m, k]
        emb_hr_r = tf.nn.embedding_lookup(self.rel_embeddings, r)  # [m, k]
        
        if direction == "tail":
            ere2_sigmoid = self.g(tf.nn.dropout(self.f1(emb_hr_e, emb_hr_r), self.config.hidden_dropout), self.ent_embeddings)
        else:
            ere2_sigmoid = self.g(tf.nn.dropout(self.f2(emb_hr_e, emb_hr_r), self.config.hidden_dropout), self.ent_embeddings) 

        ere2_loss_left = - tf.reduce_sum((tf.math.log(tf.clip_by_value(ere2_sigmoid, 1e-10, 1.0)) * tf.maximum(0., er_e2)))
        ere2_loss_right = - tf.reduce_sum((tf.math.log(tf.clip_by_value(1 - ere2_sigmoid, 1e-10, 1.0)) * tf.maximum(0., tf.negative(er_e2))))

        hrt_loss = ere2_loss_left + ere2_loss_right

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

    def g(self, f, w):
        """Defines activation layer.

            Args:
               f (Tensor): output of the forward layers.
               W (Tensor): Matrix for multiplication.
        """
        # [b, k] [k, tot_ent]
        return tf.sigmoid(tf.matmul(f, w, transpose_b=True))

    def predict_tail_rank(self, h, r, topk=-1):
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)  # [1, k]
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)  # [1, k]
        
        hrt_sigmoid = -self.g(self.f1(emb_h, emb_r), self.ent_embeddings)
        _, rank = tf.nn.top_k(hrt_sigmoid, k=topk)

        return rank

    def predict_head_rank(self, t, r, topk=-1):
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)  # [m, k]
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)  # [m, k]
        
        hrt_sigmoid = -self.g(self.f2(emb_t, emb_r), self.ent_embeddings)
        _, rank = tf.nn.top_k(hrt_sigmoid, k=topk)

        return rank