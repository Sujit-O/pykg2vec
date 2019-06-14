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
    """Function to test the models."""
    knowledge_graph = KnowledgeGraph(dataset="freebase15k", negative_sample="uniform")
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(name)
    config = config_def()
    
    config.epochs     = 1
    config.test_step  = 1
    config.test_num   = 10
    config.disp_result= display
    config.save_model = False

    if distance_measure is not None:
        config.distance_measure = distance_measure
    if bilinear is not None:
        config.bilinear = bilinear

    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=True)
    trainer.build_model()
    trainer.train_model()

    tf.reset_default_graph()

@pytest.mark.skip(reason="This is a functional method.")
def testing_function_with_args(name, distance_measure=None, bilinear=None, display=False):
    """Function to test the models with arguments."""
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

    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=True)
    trainer.build_model()
    trainer.train_model()

    tf.reset_default_graph()


def test_Complex():
    """Function to test Complex Algorithm."""
    testing_function('complex')

def test_ConvE():
    """Function to test ConvE Algorithm."""
    testing_function('conve')
    
def test_DistMult():
    """Function to test DistMult Algorithm."""
    testing_function('distmult')

def test_KG2E_EL():
    """Function to test KG2E Algorithm with expected likelihood."""
    testing_function('kg2e', distance_measure="expected_likelihood")

def test_KG2E_KL():
    """Function to test KG2E with KL divergence"""
    testing_function('kg2e', distance_measure="kl_divergence")

def test_NTN():
    """Function to test NTN Algorithm."""
    testing_function('ntn')

def test_ProjE():
    """Function to test ProjE Algorithm."""
    testing_function('proje_pointwise')

def test_RESCAL():
    """Function to test Rescal Algorithm."""
    testing_function('rescal')

def test_RotatE():
    """Function to test RotatE Algorithm."""
    testing_function('rotate')

def test_SLM():
    """Function to test SLM Algorithm."""
    testing_function('slm')

def test_SMEL():
    """Function to test SME Algorithm with linearr loss."""
    testing_function('sme', bilinear=False)

def test_SMEB():
    """Function to test SME Algorithmwith bilinear loss."""
    testing_function('sme', bilinear=True)

def test_transE():
    """Function to test TransE Algorithm."""
    testing_function('transe')

def test_transH():
    """Function to test TransH Algorithm."""
    testing_function('transh')

def test_transR():
    """Function to test TransR Algorithm."""
    testing_function('transr')

def test_TransD():
    """Function to test TransD Algorithm."""
    testing_function('transd')

def test_TransM():
    """Function to test TransM Algorithm."""
    testing_function('transm')

def test_HoLE():
    """Function to test HolE Algorithm."""
    testing_function('hole')

def test_Tucker():
    """Function to test Tucker Algorithm."""
    testing_function('tucker')

def test_Complex_args():
    """Function to test Complex Algorithm with arguments."""
    testing_function_with_args('complex')

def test_ConvE_args():
    """Function to test ConvE Algorithm with arguments."""
    testing_function_with_args('conve')
    
def test_DistMult_args():
    """Function to test DistMult Algorithm with arguments."""
    testing_function_with_args('distmult')

def test_KG2E_EL_args():
    """Function to test KG2E Algorithm with arguments."""
    testing_function_with_args('kg2e', distance_measure="expected_likelihood")

def test_KG2E_KL_args():
    """Function to test KG2E Algorithm with arguments."""
    testing_function_with_args('kg2e', distance_measure="kl_divergence")

def test_NTN_args():
    """Function to test NTN Algorithm with arguments."""
    testing_function_with_args('ntn')

def test_ProjE_args():
    """Function to test ProjE Algorithm with arguments."""
    testing_function_with_args('proje_pointwise')

def test_RESCAL_args():
    """Function to test Rescal Algorithm with arguments."""
    testing_function_with_args('rescal')

def test_RotatE_args():
    """Function to test RotatE Algorithm with arguments."""
    testing_function_with_args('rotate')

def test_SLM_args():
    """Function to test SLM Algorithm with arguments."""
    testing_function_with_args('slm')

def test_SMEL_args():
    """Function to test SME Algorithm with arguments."""
    testing_function_with_args('sme', bilinear=False)

def test_SMEB_args():
    """Function to test SME Algorithm with arguments."""
    testing_function_with_args('sme', bilinear=True)

def test_transE_args():
    """Function to test TransE Algorithm with arguments."""
    testing_function_with_args('transe')

def test_transH_args():
    """Function to test TransH Algorithm with arguments."""
    testing_function_with_args('transh')

def test_transR_args():
    """Function to test TransR Algorithm with arguments."""
    testing_function_with_args('transr')

def test_TransD_args():
    """Function to test TransD Algorithm with arguments."""
    testing_function_with_args('transd')

def test_TransM_args():
    """Function to test TransM Algorithm with arguments."""
    testing_function_with_args('transm')

def test_HoLE_args():
    """Function to test HolE Algorithm with arguments."""
    testing_function_with_args('hole')

def test_Tucker_args():
    """Function to test TuckER Algorithm with arguments."""
    testing_function_with_args('tucker')

def test_transE_display():
    """Function to test transE display."""
    testing_function('transe', display=True)

def test_transE_args_display():
    """Function to test TransE Algorithm with arguments for display."""
    testing_function_with_args('transe', display=True)