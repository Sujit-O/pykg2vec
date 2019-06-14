#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of generator
"""
import timeit

from pykg2vec.config.global_config import GeneratorConfig
from pykg2vec.utils.generator import Generator
from pykg2vec.config.config import TransEConfig
from pykg2vec.config.config import ProjE_pointwiseConfig
from pykg2vec.utils.kgcontroller import KnowledgeGraph


def test_generator_proje():
    """Function to test the generator for ProjE algorithm."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="uniform")
    knowledge_graph.force_prepare_data()

    config = ProjE_pointwiseConfig()

    gen = iter(Generator(config=GeneratorConfig(data='train', algo='ProjE'), model_config=config))
    
    for i in range(1000):
        data = list(next(gen))
        print("----batch:", i)
        
        hr_hr = data[0]
        hr_t = data[1]
        tr_tr = data[2]
        tr_h = data[3]

        print("hr_hr:", hr_hr)
        print("hr_t:", hr_t)
        print("tr_tr:", tr_tr)
        print("tr_h:", tr_h)
    gen.stop()

def test_generator_trane():
    """Function to test the generator for Translation distance based algorithm."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="uniform")
    knowledge_graph.force_prepare_data()
    
    start_time = timeit.default_timer()
    
    config = TransEConfig()

    gen = Generator(config=GeneratorConfig(data='train', algo='transe'), model_config=config)

    print("----init time:", timeit.default_timer() - start_time)
    
    for i in range(10):
        start_time_batch = timeit.default_timer()
        data = list(next(gen))
        h = data[0]
        r = data[1]
        t = data[2]
        # hr_t = data[3]
        # tr_h = data[4]
        print("----batch:", i, "----time:",timeit.default_timer() - start_time_batch)
        print(h,r,t)

    print("total time:", timeit.default_timer() - start_time)
    
    gen.stop()