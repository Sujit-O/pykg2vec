#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of model
"""
import pytest
import tensorflow as tf


from pykg2vec.config.config import *
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.kgcontroller import KnowledgeGraph
  

@pytest.mark.skip(reason="This is a functional method.")
def testing_function(name, distance_measure=None, bilinear=None, display=False):
    """Function to test the models with arguments."""
    tf.reset_default_graph()
    
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args([])
    
    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, negative_sample=args.sampling)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(name)
    config = config_def(args=args)
    
    config.epochs     = 1
    config.test_step  = 1
    config.test_num   = 10
    config.disp_result= display
    config.save_model = False
    config.debug      = True

    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=True)
    trainer.build_model()
    trainer.train_model()

@pytest.mark.parametrize("model_name", ['complex', 'conve', 'convkb', 'distmult', 
                                        'ntn', 'proje_pointwise', 'rescal', 'rotate', 'slm',
                                        'transe', 'transh', 'transr', 'transd', 'transm', 'hole',
                                        'tucker', 'transg'])
def test_KGE_methods(model_name):
    """Function to test Complex Algorithm with arguments."""
    testing_function(model_name)

def test_KG2E_EL_args():
    """Function to test KG2E Algorithm with arguments."""
    testing_function('kg2e', distance_measure="expected_likelihood")

def test_KG2E_KL_args():
    """Function to test KG2E Algorithm with arguments."""
    testing_function('kg2e', distance_measure="kl_divergence")

def test_SMEL_args():
    """Function to test SME Algorithm with arguments."""
    testing_function('sme', bilinear=False)

def test_SMEB_args():
    """Function to test SME Algorithm with arguments."""
    testing_function('sme', bilinear=True)

def test_transE_display():
    """Function to test transE display."""
    testing_function('transe', display=True)