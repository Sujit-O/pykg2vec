#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
We store the base configuration of the models here
"""
import tensorflow as tf
import os
from pathlib import Path


class BasicConfig:
    def __init__(self,
                 test_step=100,
                 test_num=300,
                 triple_num=20,
                 tmp=Path('..') / 'intermediate',
                 result=Path('..') / 'results',
                 figures=Path('..') / 'figures',
                 tmp_data=Path('..') / 'data',
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

        self.tmp_data = tmp_data
        self.plot_entity_only = plot_entity_only
        self.test_step = test_step
        self.test_num = test_num
        self.disp_triple_num = triple_num

        self.tmp = tmp
        self.result = result
        self.figures = figures
        if not os.path.exists(tmp):
            os.mkdir(tmp)
        if not os.path.exists(result):
            os.mkdir(result)
        if not os.path.exists(figures):
            os.mkdir(figures)

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


class SMEConfig(object):

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
                 learning_rate=0.003,
                 l1_flag=True,
                 ent_hidden_size=50,
                 rel_hidden_size=50,
                 batch_size=128,
                 epochs=2,
                 input_dropout=0.2,
                 ent_dropout=0.3,
                 rel_dropout=0.3,
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

        self.rel_dropout = rel_dropout
        self.ent_dropout = ent_dropout
        self.rel_hidden_size = rel_hidden_size
        self.ent_hidden_size = ent_hidden_size
        self.lmbda = lmbda
        self.feature_map_dropout = feature_map_dropout
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.use_bias = use_bias
        self.label_smoothing = label_smoothing
        self.lr_decay = lr_decay
