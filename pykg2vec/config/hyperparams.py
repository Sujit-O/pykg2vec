"""
hyperparams.py
====================================
It provides configuration for the tunable hyper-parameter ranges for all the algorithms.
"""

from argparse import ArgumentParser


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]
        self.bilinear = [True, False]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.ent_hidden_size = [8, 16, 32, 64, 128, 256]
        self.rel_hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.ent_hidden_size = [8, 16, 32]
        self.rel_hidden_size = [8, 16, 32]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.ent_hidden_size = [8, 16, 32, 64, 128, 256]
        self.rel_hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]


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
        self.learning_rate = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        self.L1_flag = [True, False]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.batch_size = [128, 256, 512]
        self.epochs = [2, 5, 10]
        self.margin = [0.4, 1.0, 2.0]
        self.optimizer = ["adam", "sgd", 'rms']
        self.sampling = ["uniform", "bern"]
        self.bilinear = [True, False]
        self.distance_measure = ["kl_divergence", "expected_likelihood"]
        self.cmax = [0.05, 0.1, 0.2]
        self.cmin = [5.00, 3.00, 2.00, 1.00]


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
