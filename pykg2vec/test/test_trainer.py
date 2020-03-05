#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of training
"""
import os
import pytest
import tensorflow as tf

from pykg2vec.config.config import KGEArgParser, Importer
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.kgcontroller import KnowledgeGraph

@pytest.mark.skip(reason="This is a functional method.")
def get_model(result_path_dir, configured_epochs):
    tf.reset_default_graph()

    args = KGEArgParser().get_args([])

    knowledge_graph = KnowledgeGraph(dataset="Freebase15k")
    knowledge_graph.prepare_data()

    config_def, model_def = Importer().import_model_config("complex")
    config = config_def(args=args)

    config.epochs = configured_epochs
    config.test_step = 1
    config.test_num = 10
    config.disp_result = False
    config.save_model = False
    config.path_result = result_path_dir
    config.patience = 5
    config.early_stop_epoch = 1
    
    return model_def(config)

def test_full_epochs(tmpdir):
    result_path_dir = tmpdir.mkdir("result_path")
    configured_epochs = 5
    model = get_model(result_path_dir, configured_epochs)

    trainer = Trainer(model=model, debug=True)
    trainer.build_model()
    trainer.train_model()

    files = os.listdir(result_path_dir)
    training_result = [f for f in files if f.endswith(".csv")][0]
    with open(os.path.join(result_path_dir, training_result)) as file:
        actual_epochs = len(file.readlines()) - 1

    assert actual_epochs == configured_epochs

def test_early_stopping(tmpdir):
    result_path_dir = tmpdir.mkdir("result_path")
    configured_epochs = 5
    model = get_model(result_path_dir, configured_epochs)

    trainer = Trainer(model=model, debug=True)
    trainer.build_model()
    trainer.train_model()

    files = os.listdir(result_path_dir)
    training_result = [f for f in files if f.endswith(".csv")][0]
    with open(os.path.join(result_path_dir, training_result)) as file:
        actual_epochs = len(file.readlines()) - 1

    assert actual_epochs < configured_epochs