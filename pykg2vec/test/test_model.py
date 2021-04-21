#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of model
"""
import pytest

from pykg2vec.common import KGEArgParser, Importer
from pykg2vec.utils.trainer import Trainer
from pykg2vec.data.kgcontroller import KnowledgeGraph


@pytest.mark.parametrize("model_name", [
    'acre',
    'analogy',
    'complex',
    'complexn3',
    'conve',
    'convkb',
    'cp',
    'distmult',
    'hyper',
    'hole',
    'interacte',
    'kg2e',
    'murp',
    'ntn',
    'octonione',
    'proje_pointwise',
    'quate',
    'rescal',
    'rotate',
    'simple',
    'simple_ignr',
    'slm',
    'sme',
    'sme_bl',
    'transe',
    'transh',
    'transr',
    'transd',
    'transm',
])
def test_kge_methods(model_name):
    """Function to test a set of KGE algorithsm."""
    testing_function(model_name)

def test_error_on_importing_model():
    with pytest.raises(ValueError) as e:
        Importer().import_model_config("unknown")
    assert "unknown model has not been implemented. please select from" in str(e)

@pytest.mark.skip(reason="This is a functional method.")
def testing_function(name):
    """Function to test the models with arguments."""
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args(['-exp', 'True', '-mn', name])

    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(name)
    config = config_def(args)

    config.epochs = 1
    config.test_step = 1
    config.test_num = 10
    config.save_model = False
    config.debug = True
    config.ent_hidden_size = 10
    config.rel_hidden_size = 10
    config.channels = 2

    model = model_def(**config.__dict__)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()
