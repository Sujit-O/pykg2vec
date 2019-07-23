#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of KGPipeline
"""
import pytest

from pykg2vec.utils.KGPipeline import KGPipeline

def test_kgpipeline():
    """Function to test the KGPipeline function."""
    kg_pipeline = KGPipeline(model="transe", dataset ="Freebase15k", debug=True)
    kg_pipeline.tune()
    kg_pipeline.test()

