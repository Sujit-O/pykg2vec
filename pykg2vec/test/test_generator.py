#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of generator
"""
from pykg2vec.config.global_config import GeneratorConfig
from pykg2vec.utils.generator import Generator
from pykg2vec.config.config import TransEConfig
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
        # print("----batch:", i)
        hr_hr = data[0]
        hr_t = data[1]
        tr_tr = data[2]
        tr_h = data[3]
        # print("hr_hr:", hr_hr)
        # print("hr_t:", hr_t)
        # print("tr_tr:", tr_tr)
        # print("tr_h:", tr_h)
    generator.stop()

    ## pass if no exception raised amid the process.

def test_generator_trane():
    """Function to test the generator for Translation distance based algorithm."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.force_prepare_data()
    
    dummy_config = TransEConfig(KGEArgParser().get_args([]))
    generator_config = GeneratorConfig(data='train', training_strategy='pairwise_based')
    generator = Generator(config=generator_config, model_config=dummy_config)
    
    for i in range(10):
        data = list(next(generator))
        h = data[0]
        r = data[1]
        t = data[2]
        # hr_t = data[3]
        # tr_h = data[4]  
    
    generator.stop()

    ## pass if no exception raised amid the process.