#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of model
"""
import pytest


from pykg2vec.config.config import *
from pykg2vec.utils.trainer import Trainer
@pytest.mark.parametrize("model_name", [
    'analogy',
    'complex',
    'complexn3',
    'cp',
    'distmult',
    'hole',
    'proje_pointwise',
    'rescal',
    'rotate',
    'slm',
    'transe',
    'transh',
    'transr',
    'transd',
    'transm',
])
def test_KGE_methods(model_name):
    """Function to test a set of KGE algorithsm."""
    testing_function(model_name)
  

from pykg2vec.utils.kgcontroller import KnowledgeGraph

@pytest.mark.skip(reason="This is a functional method.")
def testing_function(name, distance_measure=None, bilinear=None, display=False, ent_hidden_size=None, rel_hidden_size=None, channels=None):
    """Function to test the models with arguments."""
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args(['-exp', 'True'])

    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(name)
    config = config_def(args)

    config.epochs     = 1
    config.test_step  = 1
    config.test_num   = 10
    config.disp_result= display
    config.save_model = False
    config.debug      = True

    if ent_hidden_size:
        config.ent_hidden_size = ent_hidden_size
    if rel_hidden_size:
        config.rel_hidden_size = rel_hidden_size

    if channels:
        config.channels = channels

    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model)
    trainer.build_model()
    trainer.train_model()

def test_NTN():
    testing_function('ntn', ent_hidden_size=10, rel_hidden_size=10) # for avoiding OOM.

def test_ConvE():
    testing_function('conve', channels=2) # for avoiding OOM.

def test_ConvKB():
    testing_function('convkb', channels=2) # for avoiding OOM.

def test_KG2E_EL_args():
    """Function to test KG2E Algorithm with arguments."""
    testing_function('kg2e_el', distance_measure="expected_likelihood")

def test_KG2E_KL_args():
    """Function to test KG2E Algorithm with arguments."""
    testing_function('kg2e', distance_measure="kl_divergence")

def test_SMEL_args():
    """Function to test SME Algorithm with arguments."""
    testing_function('sme', bilinear=False)

def test_SMEB_args():
    """Function to test SME Algorithm with arguments."""
    testing_function('sme_bl', bilinear=True)

def test_transE_display():
    """Function to test transE display."""
    testing_function('transe', display=True)