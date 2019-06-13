#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of tuning model
"""
import pytest
import tensorflow as tf


from pykg2vec.utils.kgcontroller import KnowledgeGraph
from pykg2vec.config.hyperparams import KGETuneArgParser
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer


@pytest.mark.skip(reason="This is a functional method.")
def tunning_function(name):
    """Function to test the tuning of the models."""

    knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="uniform")
    knowledge_graph.prepare_data()

    # getting the customized configurations from the command-line arguments.
    args = KGETuneArgParser().get_args([])

    # initializing bayesian optimizer and prepare data.
    args.debug = True
    args.model = name
    
    bays_opt = BaysOptimizer(args=args)

    # perform the golden hyperparameter tuning. 
    bays_opt.optimize()

    tf.reset_default_graph()


def test_tuning_transe():
    """Function to test the tuning function."""
    tunning_function('transe')

def test_tuning_transh():
    """Function to test the tuning function."""
    tunning_function('transh')