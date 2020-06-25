"""
hyperparams.py
====================================
It provides configuration for the tunable hyper-parameter ranges for all the algorithms.
"""

from argparse import ArgumentParser
from hyperopt import hp
from hyperopt.pyll.base import scope
import numpy as np


class HyperparamterLoader:
    def __init__(self):
      # This hyperparameter setting aims to reproduce the experimental setup in its original papers. 
      self.hyperparams_paper = {
        'freebase15k': 
        {
          'transe'  : {'learning_rate':  0.01,'L1_flag': True,'hidden_size':50,'batch_size': 128,'epochs':1000,'margin':1.00,'optimizer': 'sgd','sampling':"uniform",'neg_rate':1},
          'transh'  : {'learning_rate': 0.005,'L1_flag':False,'hidden_size':50,'batch_size':1200,'epochs':1000,'margin': 0.5,'optimizer': 'sgd','sampling':"uniform",'neg_rate':1},
          'hole'    : {'learning_rate':   0.1,'hidden_size':150,'batch_size': 5000,'epochs':1000,'margin': 0.2,'optimizer':'sgd','sampling':"uniform",'neg_rate':1},
          'transm'  : {'learning_rate': 0.001,'L1_flag': True,'hidden_size':50,'batch_size': 128,'epochs':1000,'margin': 1.0,'optimizer':'adam','sampling':"uniform",'neg_rate':1},
          'rescal'  : {'learning_rate': 0.001,'L1_flag': True,'hidden_size':50,'batch_size': 128,'epochs':1000,'margin': 1.0,'optimizer':'adam','sampling':"uniform",'neg_rate':1},
          'rotate'  : {'learning_rate':0.0001,'hidden_size':1000,'batch_size': 1024,'epochs':1000,'margin': 24.0,'optimizer':'adam','sampling':"adversarial_negative_sampling",'alpha': 1.0, 'neg_rate':16},
          'sme'     : {'learning_rate': 0.1,'L1_flag': True,'hidden_size':50,'batch_size':50000,'epochs':1000,'optimizer':'adam'},
          'transr'  : {'learning_rate': 0.001,'L1_flag': True,'ent_hidden_size':50,'rel_hidden_size':50,'batch_size': 4800,'epochs': 1000,'margin': 1.0,'optimizer': 'sgd','sampling':   "bern",'neg_rate':1},
          'transd'  : {'learning_rate': 0.001,'L1_flag':False,'ent_hidden_size':50,'rel_hidden_size':50,'batch_size':  200,'epochs': 1000,'margin': 1.0,'optimizer': 'sgd','sampling':"uniform",'neg_rate':1},
          'ntn'     : {'learning_rate':  0.01,'ent_hidden_size':100,'rel_hidden_size':100,'batch_size':  128,'epochs': 1000,'margin': 1.0,'optimizer':'adam','sampling':"uniform",'neg_rate':1, 'lmbda':0.0001}, # problematic
          'slm'     : {'learning_rate':  0.01,'L1_flag': True,'ent_hidden_size':64,'rel_hidden_size':32,'batch_size':  128,'epochs': 1000,'margin': 1.0,'optimizer':'adam','sampling':"uniform",'neg_rate':1},
          'kg2e'    : {'learning_rate':  0.01,'L1_flag': True,'hidden_size':50,'batch_size':1440,'epochs':1000,'margin': 4.0,'optimizer': 'sgd','sampling':"uniform",'distance_measure': "kl_divergence",'cmax': 0.05,'cmin': 5.00,'neg_rate': 1},
          'complex' : {'learning_rate':  0.05,'hidden_size':200,'batch_size':5000,'epochs':1000,'optimizer':'adagrad','sampling':"uniform",'neg_rate':1,'lmbda':0.0001},
          'distmult': {'learning_rate':   0.1,'hidden_size':100,'batch_size':50000,'epochs':1000,'optimizer':'adagrad','sampling':"uniform",'neg_rate':1,'lmbda':0.0001},
          'proje_po': {'learning_rate':  0.01,'hidden_dropout': 0.5, 'hidden_size':200,'batch_size':200,' epochs':100, 'optimizer':'adam','lmbda':0.00001},
          'conve'   : {'learning_rate': 0.003,'optimizer':'adam', 'label_smoothing':0.1, 'batch_size':128, 'hidden_size':200, 'hidden_size_1':20, 'input_dropout':0.2, 'feature_map_dropout':0.2, 'hidden_dropout':0.3,'neg_rate':0},
          'convkb'  : {'lmbda': 0.001,'filter_sizes':[1,2],'num_filters':50,'learning_rate': 0.0001,'optimizer':'adam','hidden_size': 100,'batch_size': 128,'epochs':200,'neg_rate':1},
          'cp': {'learning_rate': 0.01, 'hidden_size': 50, 'batch_size': 128, 'epochs': 50, 'optimizer': 'adagrad', 'sampling': "uniform", 'neg_rate': 1, 'lmbda': 0.0001},
          'analogy': {'learning_rate': 0.1, 'hidden_size': 200, 'batch_size': 128, 'epochs': 500, 'optimizer': 'adagrad', 'sampling': "uniform", 'neg_rate': 1, 'lmbda': 0.0001},
          'simple': {'learning_rate': 0.05, 'hidden_size': 100, 'batch_size': 128, 'epochs': 1000, 'optimizer': 'adagrad', 'sampling': "uniform", 'neg_rate': 1, 'lmbda': 0.1}
        }
      }

      self.hyperparams_paper['fb15k'] = self.hyperparams_paper['freebase15k']
    
    def load_hyperparameter(self, dataset_name, algorithm):
      d_name = dataset_name.lower()
      a_name = algorithm.lower()

      if d_name in self.hyperparams_paper and a_name in self.hyperparams_paper[d_name]:
        params = self.hyperparams_paper[d_name][a_name]
        return params
      else:
        raise Exception("We have not explored this experimental setting! (%s, %s)"%(dataset_name, algorithm))


class KGETuneArgParser:
    """The class defines the arguements accepted for the bayesian optimizer.

      KGETuneArgParser utilizes the ArgumentParser module and add the arguments
      accepted for tuning the model.

      Args:
         model (str): Name of the model/algorithm to be tuned.
         debug (bool): If True, tunes the model in debugging mode.

      Examples:
          >>> from pykg2vec.config.hyperparams import KGETuneArgParser
          >>> from pykg2vec.utils.bayesian_optimizer import BaysOptimizer
          >>> args = KGETuneArgParser().get_args()
          >>> bays_opt = BaysOptimizer(args=args)

       Todo:
         * Add more arguments!.
    """

    def __init__(self):
        self.parser = ArgumentParser(description='Knowledge Graph Embedding tunable configs.')

        ''' basic configs '''
        self.parser.add_argument('-mn', dest='model', default='TransE', type=str, help='Model to tune')
        self.parser.add_argument('-db', dest='debug', default=False, type=lambda x: (str(x).lower() == 'true'),
                                 help='To use debug mode or not.')
        self.parser.add_argument('-ds', dest='dataset_name', default='Freebase15k', type=str, help='The dataset name (choice: fb15k/wn18/wn18_rr/yago/fb15k_237/ks/nations/umls)')
        self.parser.add_argument('-dsp', dest='dataset_path', default=None, type=str, help='The path to custom dataset.')
        self.parser.add_argument('-mt', dest='max_number_trials', default=100, type=int, help='The maximum times of trials for bayesian optimizer.')

    def get_args(self, args):
        """Gets the arguments from the console and parses it."""
        return self.parser.parse_args(args)



class TransEParams:
    """This class defines the hyperameters and its ranges for tuning TranE algorithm.

    TransEParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 10.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [500]) # always choose 10 training epochs.
        }


class TransHParams:
    """This class defines the hyperameters and its ranges for tuning TranH algorithm.

    TransHParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class TransMParams:
    """This class defines the hyperameters and its ranges for tuning TranM algorithm.

    TransMParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class RescalParams:
    """This class defines the hyperameters and its ranges for tuning Rescal algorithm.

    Rescal defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(128),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class SMEParams:
    """This class defines the hyperameters and its ranges for tuning SME algorithm.

    SME defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.
      bilinear (bool): List of boolean values.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'bilinear': hp.choice('bilinear', [True, False]),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }

class TransDParams:
    """This class defines the hyperameters and its ranges for tuning TranD algorithm.

    TransDParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class TransRParams:
    """This class defines the hyperameters and its ranges for tuning TranR algorithm.

    TransRParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      ent_hidden_size (list): List of integer values.
      rel_hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'ent_hidden_size': scope.int(hp.qloguniform('ent_hidden_size', np.log(8), np.log(256),1)),
          'rel_hidden_size': scope.int(hp.qloguniform('rel_hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class NTNParams:
    """This class defines the hyperameters and its ranges for tuning NTN algorithm.

    NTNParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      ent_hidden_size (list): List of integer values.
      rel_hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'ent_hidden_size': scope.int(hp.qloguniform('ent_hidden_size', np.log(8), np.log(64),1)),
          'rel_hidden_size': scope.int(hp.qloguniform('rel_hidden_size', np.log(8), np.log(64),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class SLMParams:
    """This class defines the hyperameters and its ranges for tuning SLM algorithm.

    SLMParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      ent_hidden_size (list): List of integer values.
      rel_hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'ent_hidden_size': scope.int(hp.qloguniform('ent_hidden_size', np.log(8), np.log(256),1)),
          'rel_hidden_size': scope.int(hp.qloguniform('rel_hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class HoLEParams:
    """This class defines the hyperameters and its ranges for tuning HoLE algorithm.

    HoLEParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class RotatEParams:
    """This class defines the hyperameters and its ranges for tuning RotatE algorithm.

    RotatEParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'margin': hp.uniform('margin', 0.0, 2.0),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class ConvEParams:
    """This class defines the hyperameters and its ranges for tuning ConvE algorithm.

    ConvEParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      lambda (list) : List of floating point values.
      feature_map_dropout (list) :List of floating point values.
      input_dropout (list) : List of floating point values.
      hidden_dropout (list) : List of floating point values.
      use_bias (list) :List of boolean values.
      label_smoothing (list) : List of floating point values.
      lr_decay (float) : List of floating point values.
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.


    """

    def __init__(self):
        self.lmbda = [0.1, 0.2]
        self.feature_map_dropout = [0.1, 0.2, 0.5]
        self.input_dropout = [0.1, 0.2, 0.5]
        self.hidden_dropout = [0.1, 0.2, 0.5]
        self.use_bias = [True, False]
        self.label_smoothing = [0.1, 0.2, 0.5]
        self.lr_decay = [0.95, 0.9, 0.8]
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [50]
        self.batch_size = [200, 400, 600]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


class ProjE_pointwiseParams:
    """This class defines the hyperameters and its ranges for tuning ProjE_pointwise algorithm.

    ProjE_pointwise defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      lambda (list) : List of floating point values.
      feature_map_dropout (list) :List of floating point values.
      input_dropout (list) : List of floating point values.
      hidden_dropout (list) : List of floating point values.
      use_bias (list) :List of boolean values.
      label_smoothing (list) : List of floating point values.
      lr_decay (float) : List of floating point values.
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.lmbda = [0.1, 0.2]
        self.feature_map_dropout = [0.1, 0.2, 0.5]
        self.input_dropout = [0.1, 0.2, 0.5]
        self.hidden_dropout = [0.1, 0.2, 0.5]
        self.use_bias = [True, False]
        self.label_smoothing = [0.1, 0.2, 0.5]
        self.lr_decay = [0.95, 0.9, 0.8]
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16]
        self.batch_size = [256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


class KG2EParams:
    """This class defines the hyperameters and its ranges for tuning KG2E algorithm.

    KG2E defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.
      bilinear (list): List of boolean values.
      distance_measure (list): [kl_divergence or expected_likelihood]
      cmax (list):  List of floating point values.
      cmin (list):  List of floating point values.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'L1_flag': hp.choice('L1_flag', [True, False]),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'margin': hp.uniform('margin', 0.5, 8.0),
          'distance_measure': hp.choice('distance_measure', ["kl_divergence", "expected_likelihood"]),
          'cmax': hp.loguniform('cmax', np.log(0.05), np.log(0.2)),
          'cmin': hp.loguniform('cmin', np.log(1), np.log(5)),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }



class ComplexParams:
    """This class defines the hyperameters and its ranges for tuning Complex algorithm.

    Complex defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      lambda (list) : List of floating point values.
      feature_map_dropout (list) :List of floating point values.
      input_dropout (list) : List of floating point values.
      hidden_dropout (list) : List of floating point values.
      use_bias (list) :List of boolean values.
      label_smoothing (list) : List of floating point values.
      lr_decay (float) : List of floating point values.
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class DistMultParams:
    """This class defines the hyperameters and its ranges for tuning DistMult algorithm.

    DistMultParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      lambda (list) : List of floating point values.
      feature_map_dropout (list) :List of floating point values.
      input_dropout (list) : List of floating point values.
      hidden_dropout (list) : List of floating point values.
      use_bias (list) :List of boolean values.
      label_smoothing (list) : List of floating point values.
      lr_decay (float) : List of floating point values.
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }


class TuckERParams:
    """This class defines the hyperameters and its ranges for tuning TuckER algorithm.

    TuckERParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      lambda (list) : List of floating point values.
      feature_map_dropout (list) :List of floating point values.
      input_dropout (list) : List of floating point values.
      hidden_dropout (list) : List of floating point values.
      use_bias (list) :List of boolean values.
      label_smoothing (list) : List of floating point values.
      lr_decay (float) : List of floating point values.
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.lmbda = [0.1, 0.2]
        self.feature_map_dropout = [0.1, 0.2, 0.5]
        self.input_dropout = [0.1, 0.2, 0.5]
        self.hidden_dropout = [0.1, 0.2, 0.5]
        self.use_bias = [True, False]
        self.label_smoothing = [0.1, 0.2, 0.5]
        self.lr_decay = [0.95, 0.9, 0.8]
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


class TransGParams:
    """This class defines the hyperameters and its ranges for tuning TransG algorithm.

    TransGParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.
      training_threshold (float): List of floating point values.
      ncluster (int): List of integer values.
      CRP_factor (float): List of floating point values.
      weight_norm (bool): List of boolean values.

    """

    def __init__(self):
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]
        self.training_threshold = [1.0, 2.0, 3.0]
        self.ncluster = [3, 4, 5, 6, 7]
        self.CRP_factor = [0.01, 0.05, 0.1]
        self.weight_norm = [True, False]


class CPParams:
    """This class defines the hyperameters and its ranges for tuning Canonical Tensor Decomposition algorithm.

    CPParams defines all the possibel values to be tuned for the algorithm. User may

    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      lambda (list) : List of floating point values.
      feature_map_dropout (list) :List of floating point values.
      input_dropout (list) : List of floating point values.
      hidden_dropout (list) : List of floating point values.
      use_bias (list) :List of boolean values.
      label_smoothing (list) : List of floating point values.
      lr_decay (float) : List of floating point values.
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }

class ANALOGYParams:
    """This class defines the hyperameters and its ranges for tuning ANALOGY algorithm.
    ANALOGYParams defines all the possibel values to be tuned for the algorithm. User may
    change these values directly for performing the bayesian optimization of the hyper-parameters
    Args:
      lambda (list) : List of floating point values.
      feature_map_dropout (list) :List of floating point values.
      input_dropout (list) : List of floating point values.
      hidden_dropout (list) : List of floating point values.
      use_bias (list) :List of boolean values.
      label_smoothing (list) : List of floating point values.
      lr_decay (float) : List of floating point values.
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.
    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }

class SimplEParams:
    """This class defines the hyperameters and its ranges for tuning SimplE algorithm.

    SimplEParams defines all the possibel values to be tuned for the algorithm. User may

    change these values directly for performing the bayesian optimization of the hyper-parameters

    Args:
      lambda (list) : List of floating point values.
      feature_map_dropout (list) :List of floating point values.
      input_dropout (list) : List of floating point values.
      hidden_dropout (list) : List of floating point values.
      use_bias (list) :List of boolean values.
      label_smoothing (list) : List of floating point values.
      lr_decay (float) : List of floating point values.
      learning_rate (list): List of floating point values.
      L1_flag (list): List of boolean values.
      hidden_size (list): List of integer values.
      batch_size (list): List of integer values.
      epochs (list): List of integer values.
      margin (list): List of floating point values.
      optimizer (list): List of strings defining the optimization algorithm to be used.
      sampling (list): List of string defining the sampling to be used for generating negative examples.

    """

    def __init__(self):
        self.search_space = {
          'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
          'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
          'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
          'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
          'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
          'epochs': hp.choice('epochs', [10]) # always choose 10 training epochs.
        }
