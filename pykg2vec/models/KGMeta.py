#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Graph Meta Class
====================================
It provides Abstract class for the Knowledge graph models.
"""

from pykg2vec.common import TrainingStrategy
from abc import ABCMeta, abstractmethod
import torch.nn as nn

class PairwiseModel(nn.Module):
    """ Meta Class for knowledge graph embedding algorithms"""

    __metaclass__ = ABCMeta

    def __init__(self, model_name):
        """Initialize and create the model to be trained and inferred"""
        super(PairwiseModel, self).__init__()

        self.model_name = model_name
        # self.config = config
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

        self.database = {} # dict to store model-specific hyperparameter

    @abstractmethod
    def embed(self, h, r, t):
        """Function to get the embedding value"""
        pass

    @abstractmethod
    def forward(self, h, r, t):
        """Function to get the embedding value"""
        pass

    def register_param(self, param_name):
        """ store hyperparameters into """
        self.database[param_name] = None

    def load_values(self, kwargs):
        for key in self.database:
            if key not in kwargs:
                raise Exception("hyperparameter %s not found!" % key)
            self.database[key] = kwargs[key]
    
    def get_param(self, key):
        if key not in self.database:
            raise Exception("hyperparameter %s not in db!" % key)
        return self.database[key]

    def load_params(self, param_list, kwargs):
        for param_name in param_list:
            if param_name not in kwargs:
                raise Exception("hyperparameter %s not found!" % param_name)
            self.database[param_name] = kwargs[param_name]
        return self.database


class PointwiseModel(nn.Module):
    """ Meta Class for knowledge graph embedding algorithms"""

    __metaclass__ = ABCMeta

    def __init__(self, model_name, config):
        """Initialize and create the model to be trained and inferred"""
        super(PointwiseModel, self).__init__()

        self.model_name = model_name
        self.config = config
        self.training_strategy = TrainingStrategy.POINTWISE_BASED

    @abstractmethod
    def embed(self, h, r, t):
        """Function to get the embedding value"""
        pass

    @abstractmethod
    def forward(self, h, r, t):
        """Function to get the embedding value"""
        pass

class ProjectionModel(nn.Module):
    """ Meta Class for knowledge graph embedding algorithms"""

    __metaclass__ = ABCMeta

    def __init__(self, model_name, config):
        """Initialize and create the model to be trained and inferred"""
        super(ProjectionModel, self).__init__()

        self.model_name = model_name
        self.config = config
        self.training_strategy = TrainingStrategy.PROJECTION_BASED

    @abstractmethod
    def embed(self, h, r, t):
        """Function to get the embedding value"""
        pass

    @abstractmethod
    def forward(self, h, r, t):
        """Function to get the embedding value"""
        pass