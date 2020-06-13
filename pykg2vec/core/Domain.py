#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Associate embeddings with human-readable names
"""
class NamedEmbedding(object):

    def __init__(self, embedding, name):
        self._embedding = embedding
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def weight(self):
        return self._embedding.weight

    @property
    def shape(self):
        return self._embedding.weight.shape