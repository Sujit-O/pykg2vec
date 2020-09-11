#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of model
"""
import os
import pytest

from pykg2vec.common import KGEArgParser, Importer
from pykg2vec.utils.trainer import Trainer
from pykg2vec.data.kgcontroller import KnowledgeGraph


@pytest.mark.skip(reason="This is a functional method.")
def testing_function_with_args(name, l1_flag, display=False):
    """Function to test the models with arguments."""
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args([])

    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(name)
    config = config_def(args)

    config.epochs = 1
    config.test_step = 1
    config.test_num = 10
    config.disp_result = display
    config.save_model = True
    config.l1_flag = l1_flag
    config.debug = True

    model = model_def(**config.__dict__)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()

    #can perform all the inference here after training the model
    trainer.enter_interactive_mode()

    #takes head, relation
    tails = trainer.infer_tails(1, 10, topk=5)
    assert len(tails) == 5

    #takes relation, tail
    heads = trainer.infer_heads(10, 20, topk=5)
    assert len(heads) == 5

    #takes head, tail
    if not name in ["conve", "proje_pointwise", "tucker"]:
        relations = trainer.infer_rels(1, 20, topk=5)
        assert len(relations) == 5

    trainer.exit_interactive_mode()

@pytest.mark.parametrize("model_name", [
    'analogy',
    'complex',
    'complexn3',
    'conve',
    'convkb',
    'cp',
    'distmult',
    'hole',
    'kg2e',
    'ntn',
    'proje_pointwise',
    'rotate',
    'rescal',
    'simple',
    'simple_ignr',
    'slm',
    'sme',
    'transd',
    'transe',
    'transh',
    'transm',
    'transr',
    'tucker',
])
def test_inference(model_name):
    """Function to test Algorithms with arguments."""
    testing_function_with_args(model_name, True)

def test_inference_on_pretrained_model():
    args = KGEArgParser().get_args([])
    config_def, model_def = Importer().import_model_config("transe")
    config = config_def(args)
    config.load_from_data = os.path.join(os.path.dirname(__file__), "resource", "pretrained", "TransE")

    model = model_def(**config.__dict__)

    # Create the model and load the trained weights.
    trainer = Trainer(model, config)
    trainer.build_model()

    #takes head, relation
    tails = trainer.infer_tails(1, 10, topk=5)
    assert len(tails) == 5

    #takes relation, tail
    heads = trainer.infer_heads(10, 20, topk=5)
    assert len(heads) == 5

    #takes head, tail
    relations = trainer.infer_rels(1, 20, topk=5)
    assert len(relations) == 5

def test_error_on_building_pretrained_model():
    with pytest.raises(ValueError) as e:
        args = KGEArgParser().get_args([])
        config_def, model_def = Importer().import_model_config("transe")
        config = config_def(args)
        config.load_from_data = "pretrained-model-does-not-exist"
        model = model_def(**config.__dict__)

        trainer = Trainer(model, config)
        trainer.build_model()

    assert "Cannot load model from %s" % config.load_from_data in str(e)
