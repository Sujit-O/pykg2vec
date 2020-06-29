#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Domain module for building Knowledge Graphs
"""
from torch.nn import Embedding

class NamedEmbedding(Embedding):
    """ Associate embeddings with human-readable names"""

    def __init__(self, name, *args, **kwargs):
        super(NamedEmbedding, self).__init__(*args, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name
