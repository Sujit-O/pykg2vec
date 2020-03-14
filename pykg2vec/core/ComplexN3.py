#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pykg2vec.core.Complex import Complex
from pykg2vec.core.KGMeta import ModelMeta


class ComplexN3(Complex):
    """`Complex Embeddings for Simple Link Prediction`_.

    ComplEx is an enhanced version of DistMult in that it uses complex-valued embeddings
    to represent both entities and relations. Using the complex-valued embedding allows
    the defined scoring function in ComplEx to differentiate that facts with assymmetric relations.
    
    Args:
        config (object): Model configuration parameters.

    Attributes:
        config (object): Model configuration.
        data_stats (object): ModelMeta object instance. It consists of the knowledge graph metadata.
        model (str): Name of the model.
    
    Examples:
        >>> from pykg2vec.core.Complex import Complex
        >>> from pykg2vec.utils.trainer import Trainer
        >>> model = Complex()
        >>> trainer = Trainer(model=model, debug=False)
        >>> trainer.build_model()
        >>> trainer.train_model()

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf
    """

    def __init__(self, config):
        super(ComplexN3, self).__init__(config)
        self.model_name = 'ComplexN3'

    def get_loss(self, h, r, t, y):
        """Defines the loss function for the algorithm."""
        h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img = self.embed(h, r, t)

        score = self.dissimilarity(h_e_real, h_e_img, r_e_real, r_e_img, t_e_real, t_e_img)

        regul_term = tf.abs(h_e_real)**3 + tf.abs(h_e_img)**3 + tf.abs(r_e_real)**3 + tf.abs(r_e_img)**3 + tf.abs(t_e_real)**3 + tf.abs(t_e_img)**3 
        
        loss = tf.reduce_sum(tf.nn.softplus(-score*y)) + self.config.lmbda*regul_term

        return loss