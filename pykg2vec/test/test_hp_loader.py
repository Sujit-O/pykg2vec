#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of the hyperparameter loader
"""
import os
import pytest
from pykg2vec.common import KGEArgParser
from pykg2vec.common import HyperparameterLoader

def test_load_default_hyperparameter_file():
    hp_loader = HyperparameterLoader(KGEArgParser().get_args([]))
    hyperparams = hp_loader.load_hyperparameter("freebase15k", "analogy")
    search_space = hp_loader.load_search_space("analogy")

    assert hyperparams["learning_rate"] == 0.1
    assert hyperparams["hidden_size"] == 200
    assert str(search_space["epochs"].inputs()[1]) == "0 Literal{10}"

def test_load_custom_hyperparameter_file():
    custom_hyperparamter_file = os.path.join(os.path.dirname(__file__), "resource", "custom_hyperparams", "custom.yaml")
    # import pdb; pdb.set_trace()
    hp_loader = HyperparameterLoader(KGEArgParser().get_args(["-hpf", custom_hyperparamter_file, "-ssf", custom_hyperparamter_file]))
    hyperparams = hp_loader.load_hyperparameter("freebase15k", "analogy")
    search_space = hp_loader.load_search_space("analogy")

    assert hyperparams["learning_rate"] == 0.01
    assert hyperparams["hidden_size"] == 200
    assert str(search_space["epochs"].inputs()[1]) == "0 Literal{100}"