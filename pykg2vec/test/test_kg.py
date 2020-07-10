#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for testing unit functions of KnowledgeGraph
"""

import os
import pytest
from pathlib import Path
from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.data.datasets import KnownDataset


@pytest.mark.parametrize("dataset_name", [
    "freebase15k",
    "deeplearning50a",
    "wordnet18",
    "wordnet18_rr",
    "yago3_10",
    "freebase15k_237",
    "kinship",
    "nations",
    "umls",
    "nell_995",
])
def test_known_datasets(dataset_name):
    """Function to test the the knowledge graph parse for Freebase."""
    knowledge_graph = KnowledgeGraph(dataset=dataset_name)
    knowledge_graph.force_prepare_data()

    assert len(knowledge_graph.dump()) == 9
    assert knowledge_graph.is_cache_exists()

    kg_metadata = knowledge_graph.dataset.read_metadata()

    assert kg_metadata.tot_triple > 0
    assert kg_metadata.tot_valid_triples > 0
    assert kg_metadata.tot_test_triples > 0
    assert kg_metadata.tot_train_triples > 0
    assert kg_metadata.tot_relation > 0
    assert kg_metadata.tot_entity > 0

    assert len(knowledge_graph.read_cache_data('triplets_train')) > 0
    assert len(knowledge_graph.read_cache_data('triplets_test')) > 0
    assert len(knowledge_graph.read_cache_data('triplets_valid')) > 0
    assert len(knowledge_graph.read_cache_data('hr_t')) > 0
    assert len(knowledge_graph.read_cache_data('tr_h')) > 0
    assert len(knowledge_graph.read_cache_data('idx2entity')) > 0
    assert len(knowledge_graph.read_cache_data('idx2relation')) > 0
    assert len(knowledge_graph.read_cache_data('entity2idx')) > 0
    assert len(knowledge_graph.read_cache_data('relation2idx')) > 0

def test_userdefined_dataset():
    custom_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource', 'custom_dataset')
    knowledge_graph = KnowledgeGraph(dataset="userdefineddataset", custom_dataset_path=custom_dataset_path)
    knowledge_graph.prepare_data()
    knowledge_graph.dump()

    knowledge_graph.read_cache_data('triplets_train')
    knowledge_graph.read_cache_data('triplets_test')
    knowledge_graph.read_cache_data('triplets_valid')
    knowledge_graph.read_cache_data('hr_t')
    knowledge_graph.read_cache_data('tr_h')
    knowledge_graph.read_cache_data('idx2entity')
    knowledge_graph.read_cache_data('idx2relation')
    knowledge_graph.read_cache_data('entity2idx')
    knowledge_graph.read_cache_data('relation2idx')

    knowledge_graph.dataset.read_metadata()
    knowledge_graph.dataset.dump()

    assert knowledge_graph.kg_meta.tot_train_triples == 1
    assert knowledge_graph.kg_meta.tot_test_triples == 1
    assert knowledge_graph.kg_meta.tot_valid_triples == 1
    assert knowledge_graph.kg_meta.tot_entity == 6
    assert knowledge_graph.kg_meta.tot_relation == 3


@pytest.mark.parametrize('file_name, new_ext', [
    ('dataset.tar.gz', 'tgz'),
    ('dataset.tgz', 'tgz'),
    ('dataset.zip', 'zip'),
])
def test_extract_compressed_dataset(file_name, new_ext):
    url = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource', file_name)).absolute().as_uri()
    dataset_name = 'test_dataset_%s' % file_name.replace('.', '_')
    dataset = KnownDataset(dataset_name, url, 'userdefineddataset-')
    dataset_dir = os.path.join(dataset.dataset_home_path, dataset_name)
    dataset_files = os.listdir(dataset_dir)

    assert len(dataset_files) == 4
    assert dataset_name + '.' + new_ext in dataset_files
    assert 'userdefineddataset-train.txt' in dataset_files
    assert 'userdefineddataset-test.txt' in dataset_files
    assert 'userdefineddataset-valid.txt' in dataset_files
