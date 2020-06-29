#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for integration tests on visualization
"""
from os import listdir
from pykg2vec.common import KGEArgParser, Importer
from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.utils.trainer import Trainer


def test_visualization(tmpdir):
    result_path_dir = tmpdir.mkdir("result_path")

    args = KGEArgParser().get_args([])

    knowledge_graph = KnowledgeGraph(dataset="Freebase15k")
    knowledge_graph.prepare_data()

    config_def, model_def = Importer().import_model_config("analogy")
    config = config_def(args=args)

    config.epochs = 5
    config.test_step = 1
    config.test_num = 1
    config.disp_result = True
    config.save_model = False
    config.debug = True
    config.patience = -1
    config.plot_embedding = True
    config.plot_training_result = True
    config.plot_testing_result = True
    config.path_figures = result_path_dir
    config.path_result = result_path_dir

    trainer = Trainer(model_def(**config.__dict__), config)
    trainer.build_model()
    trainer.train_model()

    files = listdir(result_path_dir)
    assert any(map(lambda f: "_entity_plot" in f, files))
    assert any(map(lambda f: "_rel_plot" in f, files))
    assert any(map(lambda f: "_ent_n_rel_plot" in f, files))
    assert any(map(lambda f: "_training_loss_plot_" in f, files))
    assert any(map(lambda f: "_testing_hits_plot" in f, files))
    assert any(map(lambda f: "_testing_latex_table_" in f, files))
    assert any(map(lambda f: "_testing_table_" in f, files))
    assert any(map(lambda f: "_testing_rank_plot_" in f, files))
    assert any(map(lambda f: "_testing_hits_plot_" in f, files))
