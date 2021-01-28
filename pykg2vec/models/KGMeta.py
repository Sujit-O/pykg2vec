#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Graph Meta Class
====================================
It provides Abstract class for the Knowledge graph models.
"""

from pykg2vec.common import TrainingStrategy
from abc import ABCMeta
import torch.nn as nn


class Model:
    """ Meta Class for knowledge graph embedding models"""

    def __init__(self):
        self.database = None

    def embed(self, h, r, t):
        """Function to get the embedding value"""
        raise NotImplementedError

    def forward(self, h, r, t):
        """Function to get the embedding value"""
        raise NotImplementedError

    def load_params(self, param_list, kwargs):
        """Function to load the hyperparameters"""
        for param_name in param_list:
            if param_name not in kwargs:
                raise Exception("hyperparameter %s not found!" % param_name)
            self.database[param_name] = kwargs[param_name]
        return self.database

    def get_reg(self, h, r, t, **kwargs):
        """Function to override if regularization is needed"""
        return 0.0


class PairwiseModel(nn.Module, Model):
    """ Meta Class for KGE models with translational distance"""

    __metaclass__ = ABCMeta

    def __init__(self, model_name):
        """Initialize and create the model to be trained and inferred"""
        super(PairwiseModel, self).__init__()

        self.model_name = model_name
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED
        self.database = {}  # dict to store model-specific hyperparameter


class PointwiseModel(nn.Module, Model):
    """ Meta Class for KGE models with semantic matching"""

    __metaclass__ = ABCMeta

    def __init__(self, model_name):
        """Initialize and create the model to be trained and inferred"""
        super(PointwiseModel, self).__init__()

        self.model_name = model_name
        self.training_strategy = TrainingStrategy.POINTWISE_BASED
        self.database = {}  # dict to store model-specific hyperparameter


class ProjectionModel(nn.Module, Model):
    """ Meta Class for KGE models with neural network"""

    __metaclass__ = ABCMeta

    def __init__(self, model_name):
        """Initialize and create the model to be trained and inferred"""
        super(ProjectionModel, self).__init__()

        self.model_name = model_name
        self.training_strategy = TrainingStrategy.PROJECTION_BASED
        self.database = {}  # dict to store model-specific hyperparameter
