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

# from pykg2vec.config.global_config import KnowledgeGraph
from argparse import ArgumentParser
import importlib


class Importer:
    def __init__(self):
        self.model_path = "core"
        # self.model_path = "pykg2vec.core"
        self.config_path = "config.config"
        # self.config_path = "pykg2vec.config.config"

        self.modelMap = {"complex": "Complex",
                         "conve": "ConvE",
                         "hole": "HoLE",
                         "distmult": "DistMult",
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
                         "transg": "TransG",
                         "transm": "TransM",
                         "transr": "TransR",
                         "tucker": "TuckER"}

        self.configMap = {"complex": "ComplexConfig",
                          "conve": "ConvEConfig",
                          "hole": "HoLEConfig",
                          "distmult": "DistMultConfig",
                          "kg2e": "KG2EConfig",
                          "ntn": "NTNConfig",
                          "proje_pointwise": "ProjE_pointwiseConfig",
                          "rescal": "RescalConfig",
                          "rotate": "RotatEConfig",
                          "slm": "SLMConfig",
                          "sme": "SMEConfig",
                          "transd": "TransDConfig",
                          "transe": "TransEConfig",
                          "transg": "TransGConfig",
                          "transh": "TransHConfig",
                          "transm": "TransMConfig",
                          "transr": "TransRConfig",
                          "tucker": "TuckERConfig"}

    def import_model_config(self, name):
        config_obj = None
        model_obj = None
        try:
            config_obj = getattr(importlib.import_module(self.config_path), self.configMap[name])
            model_obj = getattr(importlib.import_module(self.model_path + ".%s" % self.modelMap[name]),
                                self.modelMap[name])
        except ModuleNotFoundError:
            print("%s model  has not been implemented. please select from: %s" % (
            name, ' '.join(map(str, self.modelMap.values()))))

        return config_obj, model_obj


class KGEArgParser:

    def __init__(self):
        self.parser = ArgumentParser(description='Knowledge Graph Embedding tunable configs.')

        ''' basic configs '''
        self.general_group = self.parser.add_argument_group('Generic')

        self.general_group.add_argument('-mn', dest='model_name', default='TransE', type=str, help='Name of model')
        self.general_group.add_argument('-db', dest='debug', default=False, type=lambda x: (str(x).lower() == 'true'),
                                        help='To use debug mode or not.')
        self.general_group.add_argument('-ghp', dest='golden', default=False, type=lambda x: (str(x).lower() == 'true'),
                                        help='Use Golden Hyper parameters!')
        self.general_group.add_argument('-ds', dest='dataset_name', default='Freebase15k', type=str,
                                        help='The name of dataset.')
        self.general_group.add_argument('-ld', dest='load_from_data', default=False,
                                        type=lambda x: (str(x).lower() == 'true'), help='load_from_data!')
        self.general_group.add_argument('-sv', dest='save_model', default=True,
                                        type=lambda x: (str(x).lower() == 'true'), help='Save the model!')

        ''' arguments regarding hyperparameters '''
        self.general_hyper_group = self.parser.add_argument_group('Generic Hyperparameters')
        self.general_hyper_group.add_argument('-b', dest='batch_training', default=128, type=int,
                                              help='training batch size')
        self.general_hyper_group.add_argument('-bt', dest='batch_testing', default=16, type=int,
                                              help='testing batch size')
        self.general_hyper_group.add_argument('-mg', dest='margin', default=0.8, type=float, help='Margin to take')
        self.general_hyper_group.add_argument('-opt', dest='optimizer', default='adam', type=str,
                                              help='optimizer to be used in training.')
        self.general_hyper_group.add_argument('-s', dest='sampling', default='uniform', type=str,
                                              help='strategy to do negative sampling.')
        self.general_hyper_group.add_argument('-l', dest='epochs', default=100, type=int,
                                              help='The total number of Epochs')
        self.general_hyper_group.add_argument('-tn', dest='test_num', default=1000, type=int,
                                              help='The total number of test triples')
        self.general_hyper_group.add_argument('-ts', dest='test_step', default=10, type=int, help='Test every _ epochs')
        self.general_hyper_group.add_argument('-lr', dest='learning_rate', default=0.01, type=float,
                                              help='learning rate')
        self.general_hyper_group.add_argument('-k', dest='hidden_size', default=50, type=int,
                                              help='Hidden embedding size.')
        self.general_hyper_group.add_argument('-km', dest='ent_hidden_size', default=50, type=int,
                                              help="Hidden embedding size for entities.")
        self.general_hyper_group.add_argument('-kr', dest='rel_hidden_size', default=50, type=int,
                                              help="Hidden embedding size for relations.")
        self.general_hyper_group.add_argument('-l1', dest='l1_flag', default=True,
                                              type=lambda x: (str(x).lower() == 'true'),
                                              help='The flag of using L1 or L2 norm.')
        self.general_hyper_group.add_argument('-c', dest='C', default=0.0125, type=float,
                                              help='The parameter C used in transH.')

        ''' arguments regarding SME and KG2E '''
        self.SME_group = self.parser.add_argument_group('SME KG2E function selection')
        self.SME_group.add_argument('-func', dest='function', default='bilinear', type=str,
                                    help="The name of function used in SME model.")
        self.SME_group.add_argument('-cmax', dest='cmax', default=0.05, type=float,
                                    help="The parameter for clipping values for KG2E.")
        self.SME_group.add_argument('-cmin', dest='cmin', default=5.00, type=float,
                                    help="The parameter for clipping values for KG2E.")

        ''' arguments regarding TransG '''
        self.TransG_group = self.parser.add_argument_group('TransG function selection')
        self.TransG_group.add_argument('-th', dest='training_threshold', default=3.5, type=float,
                                    help="Training Threshold for updateing the clusters.")
        self.TransG_group.add_argument('-nc', dest='ncluster', default=4, type=int,
                                       help="Number of clusters")
        self.TransG_group.add_argument('-crp', dest='crp_factor', default=0.01, type=float,
                                       help="Chinese Restaurant Process Factor.")
        self.TransG_group.add_argument('-stb', dest='step_before', default=10, type=int,
                                       help="Steps before")
        self.TransG_group.add_argument('-wn', dest='weight_norm', default=False,
                                              type=lambda x: (str(x).lower() == 'true'),
                                       help="normalize the weights!")

        ''' for conve '''
        self.conv_group = self.parser.add_argument_group('ConvE specific Hyperparameters')
        self.conv_group.add_argument('-lmda', dest='lmbda', default=0.1, type=float, help='The lmbda used in ConvE.')
        self.conv_group.add_argument('-fmd', dest='feature_map_dropout', default=0.2, type=float,
                                     help="feature map dropout value used in ConvE.")
        self.conv_group.add_argument('-idt', dest="input_dropout", default=0.3, type=float,
                                     help="input dropout value used in ConvE.")
        self.conv_group.add_argument('-hdt', dest="hidden_dropout", default=0.3, type=float,
                                     help="hidden dropout value used in ConvE.")
        self.conv_group.add_argument('-hdt2', dest="hidden_dropout2", default=0.3, type=float,
                                     help="hidden dropout value used in ConvE.")
        self.conv_group.add_argument('-ubs', dest='use_bias', default=True, type=lambda x: (str(x).lower() == 'true'),
                                     help='The boolean indicating whether use biases or not in ConvE.')
        self.conv_group.add_argument('-lbs', dest='label_smoothing', default=0.1, type=float,
                                     help="The parameter used in label smoothing.")
        self.conv_group.add_argument('-lrd', dest='lr_decay', default=0.995, type=float,
                                     help="The parameter for learning_rate decay used in ConvE.")

        ''' others '''
        self.misc_group = self.parser.add_argument_group('MISC')
        self.misc_group.add_argument('-t', dest='tmp', default='../intermediate', type=str,
                                     help='The folder name to store trained parameters.')
        self.misc_group.add_argument('-r', dest='result', default='../results', type=str,
                                     help="The folder name to save the results.")
        self.misc_group.add_argument('-fig', dest='figures', default='../figures', type=str,
                                     help="The folder name to save the figures.")
        self.misc_group.add_argument('-plote', dest='plot_embedding', default=False,
                                     type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.misc_group.add_argument('-plot', dest='plot_entity_only', default=True,
                                     type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.misc_group.add_argument('-gp', dest='gpu_frac', default=0.8, type=float, help='GPU fraction to use')

    def get_args(self):
        return self.parser.parse_args()


class BasicConfig:
    def __init__(self, args=None):

        if args is None:
            self.test_step = 100
            self.test_num = 600
            self.triple_num = 20
            self.tmp = Path('..') / 'intermediate'
            self.result = Path('..') / 'results'
            self.figures = Path('..') / 'figures'
            self.gpu_fraction = 0.8
            self.gpu_allow_growth = True
            self.loadFromData = False
            self.save_model = False
            self.disp_summary = True
            self.disp_result = True
            self.plot_embedding = True
            self.log_device_placement = False
            self.plot_training_result = False
            self.plot_testing_result = False
            self.plot_entity_only = False
            self.full_test_flag = False
            self.batch_size_testing = 16

        else:
            self.tmp = Path(args.tmp)
            self.result = Path(args.result)
            self.figures = Path(args.figures)
            self.full_test_flag = (args.test_step == 0)
            self.plot_entity_only = args.plot_entity_only
            self.test_step = args.test_step
            self.test_num = args.test_num
            self.disp_triple_num = 20
            self.log_device_placement = False
            self.gpu_fraction = args.gpu_frac
            self.gpu_allow_growth = True
            self.loadFromData = args.load_from_data
            self.save_model = args.save_model
            self.disp_summary = True
            self.disp_result = True
            self.plot_embedding = args.plot_embedding
            self.plot_training_result = True
            self.plot_testing_result = True

            self.batch_size_testing = args.batch_testing

        self.tmp.mkdir(parents=True, exist_ok=True)
        self.result.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.hits = [10, 5]
        self.gpu_config = tf.ConfigProto(log_device_placement=self.log_device_placement)
        self.gpu_config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
        self.gpu_config.gpu_options.allow_growth = self.gpu_allow_growth
        self.knowledge_graph = KnowledgeGraph(dataset=self.data, negative_sample=self.sampling)
        self.kg_meta = self.knowledge_graph.kg_meta


class TransGConfig(BasicConfig):

    def __init__(self, args=None):
        if args is None or args.golden is True:
            # the golden setting for TransE (only for Freebase15k now)
            self.learning_rate = 0.0015
            self.L1_flag = True
            self.hidden_size = 400
            self.batch_size = 512
            self.epochs = 500
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"
            self.training_threshold = 3.0
            self.ncluster = 4
            self.CRP_factor = 0.01
            self.weight_norm = True
            self.step_before = 10



        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling
            self.training_threshold = args.training_threshold
            self.ncluster = args.ncluster
            self.CRP_factor = args.crp_factor
            self.weight_norm = args.weight_norm
            self.step_before = args.step_before

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'threshold': self.training_threshold,
            'cluster':self.ncluster,
            'crp_factor':self.CRP_factor,
            'weight_norm':self.weight_norm

        }
        BasicConfig.__init__(self, args)

class TransEConfig(BasicConfig):

    def __init__(self, args=None):
        if args is None or args.golden is True:
            # the golden setting for TransE (only for Freebase15k now)
            self.learning_rate = 0.01
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 512
            self.epochs = 500
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling
        }
        BasicConfig.__init__(self, args)


class HoLEConfig(BasicConfig):

    def __init__(self, args=None):
        if args is None or args.golden is True:
            # the golden setting for TransE (only for Freebase15k now)
            self.learning_rate = 0.01
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 512
            self.epochs = 500
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling
        }
        BasicConfig.__init__(self, args)


class TransRConfig(BasicConfig):
    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.learning_rate = 0.01
            self.L1_flag = True
            self.ent_hidden_size = 128
            self.rel_hidden_size = 128
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.ent_hidden_size = args.ent_hidden_size
            self.rel_hidden_size = args.rel_hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'ent_hidden_size': self.ent_hidden_size,
            'rel_hidden_size': self.rel_hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling
        }

        BasicConfig.__init__(self, args)


class TransDConfig(BasicConfig):
    def __init__(self, args=None):

        if args == None or args.golden is True:
            self.learning_rate = 0.01
            self.L1_flag = True
            self.ent_hidden_size = 100
            self.rel_hidden_size = 100
            self.batch_size = 4800
            self.epochs = 2
            self.margin = 2.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.ent_hidden_size = args.ent_hidden_size
            self.rel_hidden_size = args.rel_hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'ent_hidden_size': self.ent_hidden_size,
            'rel_hidden_size': self.rel_hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling
        }

        BasicConfig.__init__(self, args)


class TransMConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.learning_rate = 0.001
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling
        }

        BasicConfig.__init__(self, args)


class TransHConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.learning_rate = 0.005
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 1200
            self.epochs = 500
            self.margin = 0.5
            self.C = 0.015625
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling
            self.C = args.C

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'C': self.C
        }

        BasicConfig.__init__(self, args)


class RescalConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.learning_rate = 0.001
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling
        }

        BasicConfig.__init__(self, args)


class SMEConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.learning_rate = 0.001
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"
            self.bilinear = False

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling
            self.bilinear = True if args.function == 'bilinear' else False

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'bilinear': self.bilinear
        }

        BasicConfig.__init__(self, args)


class NTNConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.learning_rate = 0.01
            self.L1_flag = True
            self.ent_hidden_size = 64
            self.rel_hidden_size = 32
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.ent_hidden_size = args.ent_hidden_size
            self.rel_hidden_size = args.rel_hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'ent_hidden_size': self.ent_hidden_size,
            'rel_hidden_size': self.rel_hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling
        }

        BasicConfig.__init__(self, args)


class SLMConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.learning_rate = 0.01
            self.L1_flag = True
            self.ent_hidden_size = 64
            self.rel_hidden_size = 32
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.ent_hidden_size = args.ent_hidden_size
            self.rel_hidden_size = args.rel_hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'ent_hidden_size': self.ent_hidden_size,
            'rel_hidden_size': self.rel_hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling
        }

        BasicConfig.__init__(self, args)


class RotatEConfig(BasicConfig):
    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.learning_rate = 0.01
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling
        }

        BasicConfig.__init__(self, args)


class ConvEConfig(BasicConfig):
    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.lmbda = 0.1
            self.feature_map_dropout = 0.2
            self.input_dropout = 0.2
            self.hidden_dropout = 0.3
            self.use_bias = True
            self.label_smoothing = 0.1
            self.lr_decay = 0.995

            self.learning_rate = 0.003
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.lmbda = args.lmbda
            self.feature_map_dropout = args.feature_map_dropout
            self.input_dropout = args.input_dropout
            self.hidden_dropout = args.hidden_dropout
            self.use_bias = args.use_bias
            self.label_smoothing = args.label_smoothing
            self.lr_decay = args.lr_decay

            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            # TODO: Currently conve can only have k=50, 100, or 200
            self.hidden_size = 50  # args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'feature_map_dropout': self.feature_map_dropout,
            'input_dropout': self.input_dropout,
            'hidden_dropout': self.hidden_dropout,
            'use_bias': self.use_bias,
            'label_smoothing': self.label_smoothing,
            'lr_decay': self.lr_decay,

            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
        }

        BasicConfig.__init__(self, args)


class ProjE_pointwiseConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.lmbda = 0.1
            self.feature_map_dropout = 0.2
            self.input_dropout = 0.2
            self.hidden_dropout = 0.3
            self.use_bias = True
            self.label_smoothing = 0.1
            self.lr_decay = 0.995

            self.learning_rate = 0.003
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.lmbda = args.lmbda
            self.feature_map_dropout = args.feature_map_dropout
            self.input_dropout = args.input_dropout
            self.hidden_dropout = args.hidden_dropout
            self.use_bias = args.use_bias
            self.label_smoothing = args.label_smoothing
            self.lr_decay = args.lr_decay

            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'feature_map_dropout': self.feature_map_dropout,
            'input_dropout': self.input_dropout,
            'hidden_dropout': self.hidden_dropout,
            'use_bias': self.use_bias,
            'label_smoothing': self.label_smoothing,
            'lr_decay': self.lr_decay,

            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
        }

        BasicConfig.__init__(self, args)


class KG2EConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.learning_rate = 0.001
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"
            self.bilinear = False
            self.distance_measure = "kl_divergence"
            self.cmax = 0.05
            self.cmin = 5.00

        else:
            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling
            self.distance_measure = args.function
            self.cmax = args.cmax
            self.cmin = args.cmin

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'distance_measure': self.distance_measure,
            'cmax': self.cmax,
            'cmin': self.cmin
        }

        BasicConfig.__init__(self, args)


class ComplexConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.lmbda = 0.1
            self.feature_map_dropout = 0.2
            self.input_dropout = 0.2
            self.hidden_dropout = 0.3
            self.use_bias = True
            self.label_smoothing = 0.1
            self.lr_decay = 0.995

            self.learning_rate = 0.003
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.lmbda = args.lmbda
            self.feature_map_dropout = args.feature_map_dropout
            self.input_dropout = args.input_dropout
            self.hidden_dropout = args.hidden_dropout
            self.use_bias = args.use_bias
            self.label_smoothing = args.label_smoothing
            self.lr_decay = args.lr_decay

            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'feature_map_dropout': self.feature_map_dropout,
            'input_dropout': self.input_dropout,
            'hidden_dropout': self.hidden_dropout,
            'use_bias': self.use_bias,
            'label_smoothing': self.label_smoothing,
            'lr_decay': self.lr_decay,

            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
        }

        BasicConfig.__init__(self, args)


class DistMultConfig(BasicConfig):

    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.lmbda = 0.1
            self.feature_map_dropout = 0.2
            self.input_dropout = 0.2
            self.hidden_dropout = 0.3
            self.use_bias = True
            self.label_smoothing = 0.1
            self.lr_decay = 0.995

            self.learning_rate = 0.003
            self.L1_flag = True
            self.hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.lmbda = args.lmbda
            self.feature_map_dropout = args.feature_map_dropout
            self.input_dropout = args.input_dropout
            self.hidden_dropout = args.hidden_dropout
            self.use_bias = args.use_bias
            self.label_smoothing = args.label_smoothing
            self.lr_decay = args.lr_decay

            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.hidden_size = args.hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'feature_map_dropout': self.feature_map_dropout,
            'input_dropout': self.input_dropout,
            'hidden_dropout': self.hidden_dropout,
            'use_bias': self.use_bias,
            'label_smoothing': self.label_smoothing,
            'lr_decay': self.lr_decay,

            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
        }

        BasicConfig.__init__(self, args)


class TuckERConfig(BasicConfig):
    def __init__(self, args=None):

        if args is None or args.golden is True:
            self.lmbda = 0.1
            self.feature_map_dropout = 0.2
            self.input_dropout = 0.2
            self.hidden_dropout2 = 0.3
            self.hidden_dropout1 = 0.3
            self.use_bias = True
            self.label_smoothing = 0.1
            self.lr_decay = 0.995

            self.learning_rate = 0.003
            self.L1_flag = True
            self.hidden_size = 50
            self.rel_hidden_size = 50
            self.ent_hidden_size = 50
            self.batch_size = 128
            self.epochs = 2
            self.margin = 1.0
            self.data = 'Freebase15k'
            self.optimizer = 'adam'
            self.sampling = "uniform"

        else:
            self.lmbda = args.lmbda
            self.feature_map_dropout = args.feature_map_dropout
            self.input_dropout = args.input_dropout
            self.hidden_dropout1 = args.hidden_dropout
            self.hidden_dropout2 = args.hidden_dropout2
            self.use_bias = args.use_bias
            self.label_smoothing = args.label_smoothing
            self.lr_decay = args.lr_decay

            self.learning_rate = args.learning_rate
            self.L1_flag = args.l1_flag
            self.rel_hidden_size = args.rel_hidden_size
            self.ent_hidden_size = args.ent_hidden_size
            self.batch_size = args.batch_training
            self.epochs = args.epochs
            self.margin = args.margin
            self.data = args.dataset_name
            self.optimizer = args.optimizer
            self.sampling = args.sampling

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'feature_map_dropout': self.feature_map_dropout,
            'input_dropout': self.input_dropout,
            'hidden_dropout1': self.hidden_dropout1,
            'hidden_dropout2': self.hidden_dropout2,
            'use_bias': self.use_bias,
            'label_smoothing': self.label_smoothing,
            'lr_decay': self.lr_decay,

            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'rel_hidden_size': self.rel_hidden_size,
            'ent_hidden_size': self.ent_hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'margin': self.margin,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
        }

        BasicConfig.__init__(self, args)
