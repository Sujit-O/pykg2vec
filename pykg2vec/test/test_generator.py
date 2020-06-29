#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of generator
"""
import torch
from pykg2vec.data.generator import Generator
from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.data.kgcontroller import KnowledgeGraph

def test_generator_projection():
    """Function to test the generator for projection based algorithm."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.force_prepare_data()

    config_def, model_def = Importer().import_model_config("proje_pointwise")
    config = config_def(KGEArgParser().get_args([]))
    generator = Generator(model_def(**config.__dict__), config)
    generator.start_one_epoch(10)
    for _ in range(10):
        data = list(next(generator))
        assert len(data) == 5

        h = data[0]
        r = data[1]
        t = data[2]
        hr_t = data[3]
        tr_h = data[4]
        assert len(h) == len(r)
        assert len(h) == len(t)
        assert isinstance(hr_t, torch.Tensor)
        assert isinstance(tr_h, torch.Tensor)

    generator.stop()

def test_generator_pointwise():
    """Function to test the generator for pointwise based algorithm."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.force_prepare_data()

    config_def, model_def = Importer().import_model_config("complex")
    config = config_def(KGEArgParser().get_args([]))
    generator = Generator(model_def(**config.__dict__), config)
    generator.start_one_epoch(10)
    for _ in range(10):
        data = list(next(generator))
        assert len(data) == 4

        h = data[0]
        r = data[1]
        t = data[2]
        y = data[3]
        assert len(h) == len(r)
        assert len(h) == len(t)
        assert set(y) == {1, -1}

    generator.stop()

def test_generator_pairwise():
    """Function to test the generator for pairwise based algorithm."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.force_prepare_data()

    config_def, model_def = Importer().import_model_config('transe')
    config = config_def(KGEArgParser().get_args([]))
    generator = Generator(model_def(**config.__dict__), config)
    generator.start_one_epoch(10)
    for _ in range(10):
        data = list(next(generator))
        assert len(data) == 6

        ph = data[0]
        pr = data[1]
        pt = data[2]
        nh = data[3]
        nr = data[4]
        nt = data[5]
        assert len(ph) == len(pr)
        assert len(ph) == len(pt)
        assert len(ph) == len(nh)
        assert len(ph) == len(nr)
        assert len(ph) == len(nt)

    generator.stop()
