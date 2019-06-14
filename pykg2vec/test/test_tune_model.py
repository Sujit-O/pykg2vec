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

    tf.reset_default_graph()
    
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

    # tf.reset_default_graph()


def test_tuning_transe():
    """Function to test the tuning function."""
    tunning_function('transe')

def test_tuning_transh():
    """Function to test the tuning function."""
    tunning_function('transh')

def test_tuning_transm():
    """Function to test the tuning function."""
    tunning_function('transm')

def test_tuning_rescal():
    """Function to test the tuning function."""
    tunning_function('rescal')

def test_tuning_sme():
    """Function to test the tuning function."""
    tunning_function('sme')

def test_tuning_transd():
    """Function to test the tuning function."""
    tunning_function('transd')

def test_tuning_transr():
    """Function to test the tuning function."""
    tunning_function('transr')

def test_tuning_ntn():
    """Function to test the tuning function."""
    tunning_function('ntn')

def test_tuning_slm():
    """Function to test the tuning function."""
    tunning_function('slm')

def test_tuning_hole():
    """Function to test the tuning function."""
    tunning_function('hole')

def test_tuning_rotate():
    """Function to test the tuning function."""
    tunning_function('rotate')

def test_tuning_conve():
    """Function to test the tuning function."""
    tunning_function('conve')

def test_tuning_projE_pointwise():
    """Function to test the tuning function."""
    tunning_function('projE_pointwise')

def test_tuning_kg2e():
    """Function to test the tuning function."""
    tunning_function('kg2e')

def test_tuning_complex():
    """Function to test the tuning function."""
    tunning_function('complex')

def test_tuning_distmult():
    """Function to test the tuning function."""
    tunning_function('distmult')

def test_tuning_tucker():
    """Function to test the tuning function."""
    tunning_function('tucker')

# def test_tuning_transg():
#     """Function to test the tuning function."""
#     tunning_function('transg')