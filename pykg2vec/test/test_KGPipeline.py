#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of KGPipeline
"""
import pytest

from pykg2vec.utils.KGPipeline import KGPipeline
from pykg2vec.utils.kgcontroller import KnowledgeGraph

def test_kgpipeline():
    """Function to test the KGPipeline function."""
     # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset="Freebase15k")
    knowledge_graph.prepare_data()
    
    kg_pipeline = KGPipeline(model="transe", dataset ="Freebase15k", debug=True)
    kg_pipeline.tune()
    kg_pipeline.test()

