#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for preparing the data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")

from config.global_config import KnowledgeGraph


class DataPrep(object):

    def __init__(self, name_dataset='Freebase15k', sampling="uniform", algo='ConvE'):
        '''store the information of database'''
       
        self.knowledge_graph = KnowledgeGraph(dataset=name_dataset, negative_sample=sampling)

        self.algo = algo
        self.sampling = sampling
        
    def prepare_data(self):
        '''the ways to prepare data are different across algorithms.'''
        tucker_series = ["tucker", "conve", "complex", "distmult"]
        other_algorithms = ['transe', 'transr', 'transh', 'transd', 'transm', \
        'kg2e', 'proje', 'rescal','slm', 'sme_bilinear', 'sme_linear', 'ntn', 'rotate']

        name_algo = self.algo.lower()
        # check if the algorithm is out of the list of supporting algorithms. 
        if not name_algo in tucker_series and not name_algo in other_algorithms:
            raise NotImplementedError("Data preparation is not implemented for algorithm:", self.algo)

        if not self.knowledge_graph.is_cache_exists():
            self.knowledge_graph.prepare_data()
            self.knowledge_graph.cache_data()


def test_data_prep():
    data_handler = DataPrep('Freebase15k', sampling="uniform", algo='transe')
    data_handler.prepare_data()

def test_data_prep_tucker():
    data_handler = DataPrep('Freebase15k', sampling="uniform", algo='tucker')
    data_handler.prepare_data()
    # data_handler.dump()


if __name__ == '__main__':
    test_data_prep_tucker()