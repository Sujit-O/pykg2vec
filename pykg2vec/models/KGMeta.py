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

    def __init__(self, model_name, config):
        """Initialize and create the model to be trained and inferred"""
        super(PairwiseModel, self).__init__()

        self.model_name = model_name
        self.config = config
        self.training_strategy = TrainingStrategy.PAIRWISE_BASED

    @abstractmethod
    def embed(self, h, r, t):
        """Function to get the embedding value"""
        pass

    @abstractmethod
    def forward(self, h, r, t):
        """Function to get the embedding value"""
        pass


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