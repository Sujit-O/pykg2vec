#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of training
"""
import pytest

from pykg2vec.common import KGEArgParser, Importer, Monitor
from pykg2vec.utils.trainer import Trainer
from pykg2vec.data.kgcontroller import KnowledgeGraph

@pytest.mark.skip(reason="This is a functional method.")
def get_model(result_path_dir, configured_epochs, patience, config_key):
    args = KGEArgParser().get_args([])

    knowledge_graph = KnowledgeGraph(dataset="Freebase15k")
    knowledge_graph.prepare_data()

    config_def, model_def = Importer().import_model_config(config_key)
    config = config_def(args)

    config.epochs = configured_epochs
    config.test_step = 1
    config.test_num = 1
    config.disp_result = False
    config.save_model = False
    config.path_result = result_path_dir
    config.debug = True
    config.patience = patience

    return model_def(**config.__dict__), config

@pytest.mark.parametrize("config_key", list(Importer().modelMap.keys()))
def test_full_epochs(tmpdir, config_key):
    if config_key == "acre": # OOM
        return
    result_path_dir = tmpdir.mkdir("result_path")
    configured_epochs = 10
    model, config = get_model(result_path_dir, configured_epochs, -1, config_key)

    trainer = Trainer(model, config)
    trainer.build_model()
    actual_epochs = trainer.train_model()

    assert actual_epochs == configured_epochs - 1

@pytest.mark.parametrize("monitor", [
    Monitor.MEAN_RANK,
    Monitor.FILTERED_MEAN_RANK,
    Monitor.MEAN_RECIPROCAL_RANK,
    Monitor.FILTERED_MEAN_RECIPROCAL_RANK,
])
def test_early_stopping_on_ranks(tmpdir, monitor):
    result_path_dir = tmpdir.mkdir("result_path")
    configured_epochs = 10
    model, config = get_model(result_path_dir, configured_epochs, 0, "complex")

    trainer = Trainer(model, config)
    trainer.build_model(monitor=monitor)
    actual_epochs = trainer.train_model()

    assert actual_epochs < configured_epochs - 1
