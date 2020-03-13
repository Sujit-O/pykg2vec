#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of generator
"""
import pytest
import tensorflow as tf

from pykg2vec.utils.generator import Generator, TrainingStrategy
from pykg2vec.utils.kgcontroller import KnowledgeGraph
from pykg2vec.config.config import *


def test_generator_proje():
    """Function to test the generator for ProjE algorithm."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.force_prepare_data()

    config_def, model_def = Importer().import_model_config("proje_pointwise")
    generator = Generator(model_def(config_def(KGEArgParser().get_args([]))))

    for i in range(10):
        data = list(next(generator))
        assert len(data) == 5

        h = data[0]
        r = data[1]
        t = data[2]
        hr_t = data[3]
        tr_h = data[4]
        assert len(h) == len(r)
        assert len(h) == len(t)
        assert isinstance(hr_t, tf.SparseTensor)
        assert isinstance(tr_h, tf.SparseTensor)

    generator.stop()

    ## pass if no exception raised amid the process.

@pytest.mark.parametrize('Config', [
    TransDConfig,
    TransEConfig,
    TransGConfig,
    TransHConfig,
    TransMConfig,
    TransRConfig,
])
def test_generator_trans(Config):
    """Function to test the generator for Translation distance based algorithm."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.force_prepare_data()

    config_def, model_def = Importer().import_model_config("transe")
    generator = Generator(model_def(config_def(KGEArgParser().get_args([]))))

    for i in range(10):
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

    ## pass if no exception raised amid the process.