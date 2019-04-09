#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
We store the base configuration of the models here
"""
import tensorflow as tf

class TransRConfig(object):
    def __init__(self,
                 model_name = 'TransR',
                 learning_rate=0.001,
                 test_flag=False,
                 l1_flag=True,
                 ent_hidden_size=64,
                 rel_hidden_size=32,
                 load_from_data=False,
                 batch_size=128,
                 epochs=2,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 test_step=100,
                 test_num=300,
                 triple_num=20,
                 tmp='../intermediate',
                 gpu_fraction=0.4,
                 gpu_allow_growth=True,
                 save_model=True,
                 disp_summary=True,
                 disp_result=True,
                 log_device_placement=False):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.testFlag = test_flag
        self.L1_flag = l1_flag
        self.loadFromData = load_from_data
        self.ent_hidden_size = ent_hidden_size
        self.rel_hidden_size = rel_hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.test_step = test_step
        self.test_num = test_num
        self.disp_triple_num = triple_num
        self.tmp = tmp
        self.gpu_config = tf.ConfigProto(log_device_placement=log_device_placement)
        self.gpu_config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.gpu_config.gpu_options.allow_growth = gpu_allow_growth
        self.save_model = save_model
        self.disp_summary = disp_summary
        self.disp_result = disp_result


class TransEConfig(object):

    def __init__(self,
                 learning_rate=0.001,
                 test_flag=False,
                 l1_flag=True,
                 hidden_size=50,
                 load_from_data=False,
                 batch_size=128,
                 epochs=1000,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 test_step=100,
                 test_num=1000,
                 triple_num=5,
                 tmp='../intermediate',
                 result='../results',
                 gpu_fraction=0.4,
                 hits=None,
                 gpu_allow_growth=True,
                 save_model=False,
                 disp_summary=True,
                 disp_result=True,
                 log_device_placement=False):
        self.result = result
        if hits is None:
            hits = [10,5]
        self.learning_rate = learning_rate
        self.testFlag = test_flag
        self.L1_flag = l1_flag
        self.loadFromData = load_from_data
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.test_step = test_step
        self.test_num = test_num
        self.disp_triple_num = triple_num
        self.tmp = tmp
        self.hits = hits
        self.gpu_config = tf.ConfigProto(log_device_placement=log_device_placement)
        self.gpu_config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.gpu_config.gpu_options.allow_growth = gpu_allow_growth
        self.save_model = save_model
        self.disp_summary = disp_summary
        self.disp_result = disp_result


class TransHConfig(object):

    def __init__(self,
                 learning_rate=0.001,
                 test_flag=False,
                 l1_flag=True,
                 hidden_size=100,
                 load_from_data=False,
                 batch_size=128,
                 epochs=1000,
                 margin=1.0,
                 data='Freebase',
                 optimizer='adam',
                 test_step=100,
                 test_num=300,
                 triple_num=5,
                 tmp='./intermediate',
                 gpu_fraction=0.4,
                 hits=None,
                 gpu_allow_growth=True,
                 save_model=False,
                 disp_summary=True,
                 disp_result=True,
                 log_device_placement=False):
        self.learning_rate = learning_rate
        self.testFlag = test_flag
        self.L1_flag = l1_flag
        self.loadFromData = load_from_data
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.margin = margin
        self.data = data
        self.optimizer = optimizer
        self.test_step = test_step
        self.test_num = test_num
        self.disp_triple_num = triple_num
        self.tmp = tmp
        if hits is None:
            hits = [10,5]
        self.hits= hits
        self.gpu_config = tf.ConfigProto(log_device_placement=log_device_placement)
        self.gpu_config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.gpu_config.gpu_options.allow_growth = gpu_allow_growth
        self.save_model = save_model
        self.disp_summary = disp_summary
        self.disp_result = disp_result