#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of generator
"""
import pytest
import tensorflow as tf

from pykg2vec.config.global_config import GeneratorConfig
from pykg2vec.config.config import (
    TransDConfig,
    TransEConfig,
    TransGConfig,
    TransHConfig,
    TransMConfig,
    TransRConfig,
)
from pykg2vec.utils.generator import Generator
from pykg2vec.config.config import ProjE_pointwiseConfig, KGEArgParser
from pykg2vec.utils.kgcontroller import KnowledgeGraph


def test_generator_proje():
    """Function to test the generator for ProjE algorithm."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.force_prepare_data()

    dummy_config = ProjE_pointwiseConfig(KGEArgParser().get_args([]))
    generator_config = GeneratorConfig(data='train', training_strategy='projection_based')
    generator = Generator(config=generator_config, model_config=dummy_config)

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

    dummy_config = Config(KGEArgParser().get_args([]))
    generator_config = GeneratorConfig(data='train', training_strategy='pairwise_based')
    generator = Generator(config=generator_config, model_config=dummy_config)

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