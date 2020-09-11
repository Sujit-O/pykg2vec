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
    custom_hyperparamter_file = os.path.join(os.path.dirname(__file__), "resource", "custom_hyperparams", "custom_hpf.yaml")
    custom_ss_file = os.path.join(os.path.dirname(__file__), "resource", "custom_hyperparams", "custom_ssf.yaml")
    hp_loader = HyperparameterLoader(KGEArgParser().get_args(["-hpf", custom_hyperparamter_file, "-ssf", custom_ss_file]))
    hyperparams = hp_loader.load_hyperparameter("freebase15k", "analogy")
    search_space = hp_loader.load_search_space("analogy")

    assert hyperparams["learning_rate"] == 0.01
    assert hyperparams["hidden_size"] == 200
    assert str(search_space["epochs"].inputs()[1]) == "0 Literal{100}"

def test_exception_on_hyperparameter_file_not_exist():
    with pytest.raises(FileNotFoundError) as e:
        hp_loader = HyperparameterLoader(KGEArgParser().get_args(["-hpf", "not_exist_file"]))
        hp_loader.load_hyperparameter("freebase15k", "analogy")

    assert str(e.value) == "Cannot find configuration file not_exist_file"

def test_exception_on_search_space_file_not_exist():
    with pytest.raises(FileNotFoundError) as e:
        hp_loader = HyperparameterLoader(KGEArgParser().get_args(["-ssf", "not_exist_file"]))
        hp_loader.load_search_space("analogy")

    assert str(e.value) == "Cannot find configuration file not_exist_file"

def test_exception_on_hyperparameter_file_with_wrong_extension():
    custom_hyperparamter_file = os.path.join(os.path.dirname(__file__), "resource", "custom_hyperparams", "custom.txt")
    with pytest.raises(ValueError) as e:
        hp_loader = HyperparameterLoader(KGEArgParser().get_args(["-hpf", custom_hyperparamter_file]))
        hp_loader.load_hyperparameter("freebase15k", "analogy")

    assert str(e.value) == "Configuration file must have .yaml or .yml extension: %s" % custom_hyperparamter_file

def test_exception_on_search_space_file_with_wrong_extension():
    custom_hyperparamter_file = os.path.join(os.path.dirname(__file__), "resource", "custom_hyperparams", "custom.txt")
    with pytest.raises(ValueError) as e:
        hp_loader = HyperparameterLoader(KGEArgParser().get_args(["-ssf", custom_hyperparamter_file]))
        hp_loader.load_search_space("analogy")

    assert str(e.value) == "Configuration file must have .yaml or .yml extension: %s" % custom_hyperparamter_file
