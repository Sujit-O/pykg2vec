#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
We store the base configuration of the models here
"""
import tensorflow as tf
from pathlib import Path

import sys

sys.path.append("../")
from config.global_config import KnowledgeGraph
import importlib

class Importer: 
    def __init__(self):
        self.model_path = "core"
        self.config_path = "config.config"

        self.modelMap = {"complex": "Complex",
                         "conve": "ConvE",
                         "distmult": "DistMult",
                         "distmult2": "DistMult2",
                         "kg2e": "KG2E",
                         "ntn": "NTN",
                         "proje_pointwise": "ProjE_pointwise",
                         "rescal": "Rescal",
                         "rotate": "RotatE",
                         "slm": "SLM",
                         "sme": "SME",
                         "transd": "TransD",
                         "transe": "TransE",
                         "transh": "TransH",
                         "transm": "TransM",
                         "transR": "TransR",
                         "tucker": "TuckER",
                         "tucker_v2": "TuckER_v2"}

        self.configMap = {"complex": "ComplexConfig",
                         "conve": "ConvEConfig",
                         "distmult": "DistMultConfig",
                         "distmult2": "DistMultConfig",
                         "kg2e": "KG2EConfig",
                         "ntn": "NTNConfig",
                         "proje_pointwise": "ProjE_pointwiseConfig",
                         "rescal": "RescalConfig",
                         "rotate": "RotatEConfig",
                         "slm": "SLMConfig",
                         "sme": "SMEConfig",
                         "transd": "TransDConfig",
                         "transe": "TransEConfig",
                         "transh": "TransHConfig",
                         "transm": "TransMConfig",
                         "transR": "TransRConfig",
                         "tucker": "TuckERConfig",
                         "tucker_v2": "TuckERConfig"}
    
    def import_model_config(self, name):
        config_obj = None
        model_obj = None
        try:
            config_obj = getattr(importlib.import_module(self.config_path), self.configMap[name])
            model_obj = getattr(importlib.import_module(self.model_path + ".%s" % self.modelMap[name]), self.modelMap[name])
        except ModuleNotFoundError:
            print("%s model  has not been implemented. please select from: %s" % (name, ' '.join(map(str, self.modelMap.values()))))
        
        return config_obj, model_obj

class BasicConfig:
    def __init__(self,
                 test_step=100,
                 test_num=600,
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
                 plot_entity_only=False,
                 full_test_flag=False):
        self.full_test_flag = full_test_flag
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
        self.batch_size_testing = 128

    def set_dataset(self, dataset_name):
        self.knowledge_graph = KnowledgeGraph(dataset=dataset_name)
        self.kg_meta = self.knowledge_graph.kg_meta


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
                 learning_rate=0.01,
                 l1_flag=True,
                 hidden_size=50,
                 batch_size=512,
                 epochs=500,
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
                 lmbda=0.1,
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
