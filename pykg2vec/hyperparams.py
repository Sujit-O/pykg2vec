"""
hyperparams.py
====================================
It provides configuration for the tunable hyper-parameter ranges for all the algorithms.
"""

from hyperopt import hp
from hyperopt.pyll.base import scope
import numpy as np


class TransEParams:
    """ This class defines the hyperameters and its ranges for tuning TranE algorithm.

        TransEParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 10.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [500])
        }


class TransHParams:
    """ This class defines the hyperameters and its ranges for tuning TranH algorithm.

        TransHParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class TransMParams:
    """ This class defines the hyperameters and its ranges for tuning TranM algorithm.

        TransMParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class RescalParams:
    """ This class defines the hyperameters and its ranges for tuning Rescal algorithm.

        Rescal defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(128), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class SMEParams:
    """ This class defines the hyperameters and its ranges for tuning SME algorithm.

        SME defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            'bilinear': hp.choice('bilinear', [True, False]),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class TransDParams:
    """ This class defines the hyperameters and its ranges for tuning TranD algorithm.

        TransDParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class TransRParams:
    """ This class defines the hyperameters and its ranges for tuning TranR algorithm.

        TransRParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'ent_hidden_size': scope.int(hp.qloguniform('ent_hidden_size', np.log(8), np.log(256), 1)),
            'rel_hidden_size': scope.int(hp.qloguniform('rel_hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class NTNParams:
    """ This class defines the hyperameters and its ranges for tuning NTN algorithm.

        NTNParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'ent_hidden_size': scope.int(hp.qloguniform('ent_hidden_size', np.log(8), np.log(64), 1)),
            'rel_hidden_size': scope.int(hp.qloguniform('rel_hidden_size', np.log(8), np.log(64), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class SLMParams:
    """ This class defines the hyperameters and its ranges for tuning SLM algorithm.

        SLMParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'ent_hidden_size': scope.int(hp.qloguniform('ent_hidden_size', np.log(8), np.log(256), 1)),
            'rel_hidden_size': scope.int(hp.qloguniform('rel_hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class HoLEParams:
    """ This class defines the hyperameters and its ranges for tuning HoLE algorithm.

        HoLEParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class RotatEParams:
    """ This class defines the hyperameters and its ranges for tuning RotatE algorithm.

        RotatEParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters


    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'margin': hp.uniform('margin', 0.0, 2.0),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class ConvEParams:
    """ This class defines the hyperameters and its ranges for tuning ConvE algorithm.

        ConvEParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

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
    """ This class defines the hyperameters and its ranges for tuning ProjE_pointwise algorithm.

        ProjE_pointwise defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

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
    """ This class defines the hyperameters and its ranges for tuning KG2E algorithm.

        KG2E defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'L1_flag': hp.choice('L1_flag', [True, False]),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            'margin': hp.uniform('margin', 0.5, 8.0),
            'distance_measure': hp.choice('distance_measure', ["kl_divergence", "expected_likelihood"]),
            'cmax': hp.loguniform('cmax', np.log(0.05), np.log(0.2)),
            'cmin': hp.loguniform('cmin', np.log(1), np.log(5)),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class ComplexParams:
    """ This class defines the hyperameters and its ranges for tuning Complex algorithm.

        Complex defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class DistMultParams:
    """ This class defines the hyperameters and its ranges for tuning DistMult algorithm.

        DistMultParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class TuckERParams:
    """ This class defines the hyperameters and its ranges for tuning TuckER algorithm.

        TuckERParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

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
    """ This class defines the hyperameters and its ranges for tuning TransG algorithm.

        TransGParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

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
    """ This class defines the hyperameters and its ranges for tuning Canonical Tensor Decomposition algorithm.

        CPParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class ANALOGYParams:
    """ This class defines the hyperameters and its ranges for tuning ANALOGY algorithm.

        ANALOGYParams defines all the possibel values to be tuned for the algorithm.
        User may change these values directly for performing2 the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }


class SimplEParams:
    """ This class defines the hyperameters and its ranges for tuning SimplE algorithm.

        SimplEParams defines all the possibel values to be tuned for the algorithm. User may
        change these values directly for performing the bayesian optimization of the hyper-parameters

    """

    def __init__(self):
        self.search_space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
            'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256), 1)),
            'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096), 1)),
            'lmbda': hp.loguniform('lmbda', np.log(0.00001), np.log(0.001)),
            'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
            # always choose 10 training epochs.
            'epochs': hp.choice('epochs', [10])
        }
