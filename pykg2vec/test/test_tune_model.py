#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of tuning model
"""
import pytest

from unittest.mock import patch
from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.common import KGEArgParser
from pykg2vec.common import HyperparameterLoader
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer


@pytest.mark.skip(reason="This is a functional method.")
def tunning_function(name):
    """Function to test the tuning of the models."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.prepare_data()

    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args([])

    # initializing bayesian optimizer and prepare data.
    args.debug = True
    args.model_name = name

    bays_opt = BaysOptimizer(args=args)
    bays_opt.config_local.test_num = 10

    # perform the golden hyperparameter tuning.
    bays_opt.optimize()

    assert bays_opt.return_best() is not None

@pytest.mark.parametrize('model_name', [
    # 'acre',
    'analogy',
    'complex',
    'complexn3',
    # 'conve',
    # 'convkb',
    'cp',
    'distmult',
    'hole',
    # "hyper",
    # "interacte",
    'kg2e',
    'ntn',
    # 'proje_pointwise',
    'rescal',
    'rotate',
    'simple',
    'simple_ignr',
    'slm',
    'sme',
    'sme_bl',
    'transe',
    'transh',
    'transm',
    'transd',
    'transr',
    'tucker',
])
def test_tuning(model_name):
    """Function to test the tuning function."""
    tunning_function(model_name)

@pytest.mark.parametrize('model_name', [
    'acre',
    'analogy',
    'complex',
    'complexn3',
    'conve',
    'convkb',
    'cp',
    'distmult',
    'hole',
    'hyper',
    'interacte',
    'kg2e',
    'ntn',
    'proje_pointwise',
    'rescal',
    'rotate',
    'simple',
    'simple_ignr',
    'slm',
    'sme',
    'sme_bl',
    'transe',
    'transh',
    'transm',
    'transd',
    'transr',
    'tucker',
])
def test_hyperparamter_loader(model_name):
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.prepare_data()

    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args([])

    hyperparams = HyperparameterLoader(args).load_hyperparameter("freebase15k", model_name)

    assert hyperparams["optimizer"] is not None

@pytest.mark.parametrize('model_name', [
    # 'acre',
    'analogy',
    'complex',
    'complexn3',
    # 'conve',
    # 'convkb',
    'cp',
    'distmult',
    'hole',
    # "hyper",
    # "interacte",
    'kg2e',
    'ntn',
    # 'proje_pointwise',
    'rescal',
    'rotate',
    'simple',
    'simple_ignr',
    'slm',
    'sme',
    'sme_bl',
    'transe',
    'transh',
    'transm',
    'transd',
    'transr',
    'tucker',
])
def test_search_space_loader(model_name):
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.prepare_data()

    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args([])

    hyperparams = HyperparameterLoader(args).load_search_space(model_name)

    assert hyperparams["epochs"] is not None

@patch('pykg2vec.utils.bayesian_optimizer.fmin')
def test_return_empty_before_optimization(mocked_fmin):
    """Function to test the tuning of the models."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k")
    knowledge_graph.prepare_data()

    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args([])

    # initializing bayesian optimizer and prepare data.
    args.debug = True
    args.model_name = 'analogy'

    bays_opt = BaysOptimizer(args=args)
    bays_opt.config_local.test_num = 10

    with pytest.raises(Exception) as e:
        bays_opt.return_best()

    assert mocked_fmin.called is False
    assert e.value.args[0] == 'Cannot find golden setting. Has optimize() been called?'
