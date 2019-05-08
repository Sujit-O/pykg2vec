#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
We store the base configuration of the models here
"""
import tensorflow as tf
from pathlib import Path

import sys
sys.path.append("../")
from config.global_config import GlobalConfig
import pickle 

class BasicConfig:
    def __init__(self,
                 test_step=100,
                 test_num=300,
                 triple_num=20,
                 tmp=Path('..') / 'intermediate',
                 result=Path('..') / 'results',
                 figures=Path('..') / 'figures',
                 gpu_fraction=0.8,
                 hits=None,
                 gpu_allow_growth=True,
                 load_from_data=False,
                 save_model=False,
                 disp_summary=True,
                 disp_result=True,
                 plot_embedding=True,
                 log_device_placement=False,
                 plot_training_result=False,
                 plot_testing_result=False,
                 plot_entity_only=False):

        self.plot_entity_only = plot_entity_only
        self.test_step = test_step
        self.test_num = test_num
        self.disp_triple_num = triple_num

        self.tmp = tmp
        self.result = result
        self.figures = figures
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.result.mkdir(parents=True, exist_ok=True) 
        self.figures.mkdir(parents=True, exist_ok=True)

        if hits is None:
            hits = [10, 5]
        self.hits = hits

        self.gpu_fraction = gpu_fraction
        self.gpu_allow_growth = gpu_allow_growth
        self.gpu_config = tf.ConfigProto(log_device_placement=log_device_placement)
        self.gpu_config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.gpu_config.gpu_options.allow_growth = gpu_allow_growth

        self.loadFromData = load_from_data
        self.save_model = save_model

        self.disp_summary = disp_summary
        self.disp_result = disp_result
        self.plot_embedding = plot_embedding
        self.log_device_placement = log_device_placement
        self.plot_training_result = plot_training_result
        self.plot_testing_result = plot_testing_result
        self.knowledge_graph = None
        self.batch_size_testing = 16

    def set_dataset(self, dataset_name):
        self.knowledge_graph = GlobalConfig(dataset=dataset_name)
        with open(str(self.knowledge_graph.dataset.metadata_path), 'rb') as f:
            self.kg_meta = pickle.load(f)

    def read_hr_t(self, train_only=False):
        if not self.knowledge_graph:
            raise Exception("not initialized!")
        
        if train_only:
            path = self.knowledge_graph.dataset.hrt_train_path
        else:
            path = self.knowledge_graph.dataset.hrt_path

        with open(str(path), 'rb') as f:
            hr_t = pickle.load(f)
            return hr_t

    def read_tr_h(self, train_only=False):
        if not self.knowledge_graph:
            raise Exception("not initialized!")
        
        if train_only:
            path = self.knowledge_graph.dataset.trh_train_path
        else:
            path = self.knowledge_graph.dataset.trh_path

        with open(str(path), 'rb') as f:
            tr_h = pickle.load(f)
            return tr_h

    def read_train_triples_ids(self):
        if not self.knowledge_graph:
            raise Exception("not initialized!")
        
        with open(str(self.knowledge_graph.dataset.training_triples_id_path), 'rb') as f:
            train_triples_ids = pickle.load(f)
            return train_triples_ids

    def read_test_triples_ids(self):
        if not self.knowledge_graph:
            raise Exception("not initialized!")
                
        with open(str(self.knowledge_graph.dataset.testing_triples_id_path), 'rb') as f:
            test_triples_ids = pickle.load(f)
            return test_triples_ids

    def read_valid_triples_ids(self):
        if not self.knowledge_graph:
            raise Exception("not initialized!")

        with open(str(self.knowledge_graph.dataset.validating_triples_id_path), 'rb') as f:
            valid_triples_ids = pickle.load(f)
            return valid_triples_ids

    def read_train_data(self):
        if not self.knowledge_graph:
            raise Exception("not initialized!")

        with open(str(self.knowledge_graph.dataset.hrt_hr_rt_train), 'rb') as f:
            train_data = pickle.load(f)
            return train_data

    def read_relation_property(self):
        if not self.knowledge_graph:
            raise Exception("not initialized!")

        with open(str(self.knowledge_graph.dataset.relation_property_path), 'rb') as f:
            relation_property = pickle.load(f)
            return relation_property

    def read_idx2entity(self):
        if not self.knowledge_graph:
            raise Exception("not initialized!")

        with open(str(self.knowledge_graph.dataset.idx2entity_path), 'rb') as f:
            idx2entity = pickle.load(f)
            return idx2entity

    def read_idx2relation(self):
        if not self.knowledge_graph:
            raise Exception("not initialized!")

        with open(str(self.knowledge_graph.dataset.idx2relation_path), 'rb') as f:
            idx2relation = pickle.load(f)
            return idx2relation

class TransRConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.01,
                 l1_flag=True,
                 ent_hidden_size=64,
                 rel_hidden_size=32,
                 batch_size=128,
                 epochs=2,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.ent_hidden_size = ent_hidden_size
        self.rel_hidden_size = rel_hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling

class TransDConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.01,
                 l1_flag=True,
                 ent_hidden_size=64,
                 rel_hidden_size=32,
                 batch_size=128,
                 epochs=2,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.ent_hidden_size = ent_hidden_size
        self.rel_hidden_size = rel_hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling


class TransEConfig(BasicConfig):

    def __init__(self,
                 learning_rate=0.001,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=1000,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling



class TransMConfig(BasicConfig):

    def __init__(self,
                 learning_rate=0.001,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=1000,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling

class TransHConfig(BasicConfig):

    def __init__(self,
                 learning_rate=0.001,
                 l1_flag=True,
                 hidden_size=100,
                 batch_size=128,
                 epochs=1000,
                 margin=1.0,
                 C=0.123,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.C = C
        self.sampling = sampling


class RescalConfig(BasicConfig):

    def __init__(self,
                 learning_rate=0.001,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=1000,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling


class SMEConfig(BasicConfig):

    def __init__(self,
                 learning_rate=0.001,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=1000,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform", 
                 bilinear=False):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling
        self.bilinear = bilinear


class NTNConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.01,
                 l1_flag=True,
                 ent_hidden_size=64,
                 rel_hidden_size=32,
                 batch_size=128,
                 epochs=2,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.ent_hidden_size = ent_hidden_size
        self.rel_hidden_size = rel_hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling


class SLMConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.01,
                 l1_flag=True,
                 ent_hidden_size=64,
                 rel_hidden_size=32,
                 batch_size=128,
                 epochs=2,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.ent_hidden_size = ent_hidden_size
        self.rel_hidden_size = rel_hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling


class RotatEConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.01,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=2,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling


class ConvEConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.003,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=2,
                 input_dropout=0.2,
                 hidden_dropout=0.3,
                 feature_map_dropout=0.2,
                 lr_decay=0.995,
                 label_smoothing=0.1,
                 use_bias=True,
                 lmbda=0.1,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.lmbda = lmbda
        self.feature_map_dropout = feature_map_dropout
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.use_bias = use_bias
        self.label_smoothing = label_smoothing
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling


class ProjE_pointwiseConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.003,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=2,
                 input_dropout=0.2,
                 hidden_dropout=0.3,
                 feature_map_dropout=0.2,
                 lr_decay=0.995,
                 label_smoothing=0.1,
                 use_bias=True,
                 lmbda=0.1,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.lmbda = lmbda
        self.feature_map_dropout = feature_map_dropout
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.use_bias = use_bias
        self.label_smoothing = label_smoothing
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling


class KG2EConfig(BasicConfig):

     def __init__(self,
                 learning_rate=0.001,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=1000,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform", 
                 distance_measure="kl_divergence",
                 cmax=0.05,
                 cmin=5.00
                 ):
        BasicConfig.__init__(self)

        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling
        self.distance_measure = distance_measure
        self.cmax = cmax
        self.cmin = cmin


class ComplexConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.003,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=2,
                 input_dropout=0.2,
                 hidden_dropout=0.3,
                 feature_map_dropout=0.2,
                 lr_decay=0.995,
                 label_smoothing=0.1,
                 use_bias=True,
                 lmbda=0.1,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.lmbda = lmbda
        self.feature_map_dropout = feature_map_dropout
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.use_bias = use_bias
        self.label_smoothing = label_smoothing
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling


class DistMultConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.003,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=128,
                 epochs=2,
                 input_dropout=0.2,
                 hidden_dropout=0.3,
                 feature_map_dropout=0.2,
                 lr_decay=0.995,
                 lmbda = 0.1,
                 label_smoothing=0.1,
                 use_bias=True,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.lmbda = lmbda
        self.feature_map_dropout = feature_map_dropout
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.use_bias = use_bias
        self.label_smoothing = label_smoothing
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.L1_flag = l1_flag
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling


class TuckERConfig(BasicConfig):
    def __init__(self,
                 learning_rate=0.01,
                 l1_flag=True,
                 ent_hidden_size=50,
                 rel_hidden_size=50,
                 batch_size=128,
                 epochs=2,
                 input_dropout=0.2,
                 hidden_dropout1=0.3,
                 hidden_dropout2=0.3,
                 feature_map_dropout=0.2,
                 lr_decay=0.995,
                 label_smoothing=0.1,
                 use_bias=True,
                 lmbda=0.01,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 sampling="uniform"):
        BasicConfig.__init__(self)

        self.margin = margin
        self.hidden_dropout2 = hidden_dropout2
        self.hidden_dropout1 = hidden_dropout1
        self.epochs = epochs
        self.batch_size = batch_size
        self.l1_flag = l1_flag
        self.learning_rate = learning_rate
        self.rel_hidden_size = rel_hidden_size
        self.ent_hidden_size = ent_hidden_size
        self.lmbda = lmbda
        self.feature_map_dropout = feature_map_dropout
        self.input_dropout = input_dropout
        self.use_bias = use_bias
        self.label_smoothing = label_smoothing
        self.lr_decay = lr_decay
        self.data = data
        self.optimizer = optimizer
        self.sampling = sampling
