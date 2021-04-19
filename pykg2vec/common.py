import os
import yaml
import importlib
import numpy as np
from enum import Enum
from argparse import ArgumentParser
from pykg2vec.utils.logger import Logger
from hyperopt import hp
from hyperopt.pyll.base import scope
from pathlib import Path

class Monitor(Enum):
    """Training monitor enums"""
    MEAN_RANK = "mr"
    FILTERED_MEAN_RANK = "fmr"
    MEAN_RECIPROCAL_RANK = "mrr"
    FILTERED_MEAN_RECIPROCAL_RANK = "fmrr"


class TrainingStrategy(Enum):
    """Training strategy enums"""
    PROJECTION_BASED = "projection_based"               # matching models with neural network
    PAIRWISE_BASED = "pairwise_based"                   # translational distance models
    POINTWISE_BASED = "pointwise_based"                 # semantic matching models


class KGEArgParser:
    """The class implements the argument parser for the pykg2vec.

    KGEArgParser defines all the necessary arguments for the global and local
    configuration of all the modules.

    Attributes:
        general_group (object): It parses the general arguements used by most of the modules.
        general_hyper_group (object): It parses the arguments for the hyper-parameter tuning.

    Examples:
        >>> from pykg2vec.config import KGEArgParser
        >>> args = KGEArgParser().get_args()
    """

    def __init__(self):
        self.parser = ArgumentParser(description='Knowledge Graph Embedding tunable configs.')

        ''' argument group for hyperparameters '''
        self.general_hyper_group = self.parser.add_argument_group('Generic Hyperparameters')
        self.general_hyper_group.add_argument('-lmda', dest='lmbda', default=0.1, type=float, help='The lmbda for regularization.')
        self.general_hyper_group.add_argument('-b', dest='batch_size', default=128, type=int, help='training batch size')
        self.general_hyper_group.add_argument('-mg', dest='margin', default=0.8, type=float, help='Margin to take')
        self.general_hyper_group.add_argument('-opt', dest='optimizer', default='adam', type=str, help='optimizer to be used in training.')
        self.general_hyper_group.add_argument('-s', dest='sampling', default='uniform', type=str, help='strategy to do negative sampling.')
        self.general_hyper_group.add_argument('-ngr', dest='neg_rate', default=1, type=int, help='The number of negative samples generated per positve one.')
        self.general_hyper_group.add_argument('-l', dest='epochs', default=100, type=int, help='The total number of Epochs')
        self.general_hyper_group.add_argument('-lr', dest='learning_rate', default=0.01, type=float, help='learning rate')
        self.general_hyper_group.add_argument('-k', dest='hidden_size', default=50, type=int, help='Hidden embedding size.')
        self.general_hyper_group.add_argument('-km', dest='ent_hidden_size', default=50, type=int, help="Hidden embedding size for entities.")
        self.general_hyper_group.add_argument('-kr', dest='rel_hidden_size', default=50, type=int, help="Hidden embedding size for relations.")
        self.general_hyper_group.add_argument('-k2', dest='hidden_size_1', default=10, type=int, help="Hidden embedding size for relations.")
        self.general_hyper_group.add_argument('-l1', dest='l1_flag', default=True, type=lambda x: (str(x).lower() == 'true'), help='The flag of using L1 or L2 norm.')
        self.general_hyper_group.add_argument('-al', dest='alpha', default=0.1, type=float, help='The alpha used in self-adversarial negative sampling.')
        self.general_hyper_group.add_argument('-fsize', dest='filter_sizes', default=[1, 2, 3], nargs='+', type=int, help='Filter sizes to be used in convKB which acts as the widths of the kernals')
        self.general_hyper_group.add_argument('-fnum', dest='num_filters', default=50, type=int, help='Filter numbers to be used in convKB and InteractE.')
        self.general_hyper_group.add_argument('-fmd', dest='feature_map_dropout', default=0.2, type=float, help='feature map dropout value used in ConvE and InteractE.')
        self.general_hyper_group.add_argument('-idt', dest='input_dropout', default=0.3, type=float, help='input dropout value used in ConvE and InteractE.')
        self.general_hyper_group.add_argument('-hdt', dest='hidden_dropout', default=0.3, type=float, help='hidden dropout value used in ConvE.')
        self.general_hyper_group.add_argument('-hdt1', dest='hidden_dropout1', default=0.4, type=float, help='hidden dropout value used in TuckER.')
        self.general_hyper_group.add_argument('-hdt2', dest='hidden_dropout2', default=0.5, type=float, help='hidden dropout value used in TuckER.')
        self.general_hyper_group.add_argument('-lbs', dest='label_smoothing', default=0.1, type=float, help='The parameter used in label smoothing.')
        self.general_hyper_group.add_argument('-cmax', dest='cmax', default=0.05, type=float, help='The parameter for clipping values for KG2E.')
        self.general_hyper_group.add_argument('-cmin', dest='cmin', default=5.00, type=float, help='The parameter for clipping values for KG2E.')
        self.general_hyper_group.add_argument('-fp', dest='feature_permutation', default=1, type=int, help='The number of feature permutations for InteractE.')
        self.general_hyper_group.add_argument('-rh', dest='reshape_height', default=20, type=int, help='The height of the reshaped matrix for InteractE.')
        self.general_hyper_group.add_argument('-rw', dest='reshape_width', default=10, type=int, help='The width of the reshaped matrix for InteractE.')
        self.general_hyper_group.add_argument('-ks', dest='kernel_size', default=9, type=int, help='The kernel size to use for InteractE.')
        self.general_hyper_group.add_argument('-ic', dest='in_channels', default=9, type=int, help='The kernel size to use for InteractE.')
        self.general_hyper_group.add_argument('-w', dest='way', default="parallel", type=str, choices=["serial", "parallel"], help='The way used by AcrE to organize standard convolution and atrous convolutions.')
        self.general_hyper_group.add_argument('-fa', dest='first_atrous', default=1, type=int, help='The first layer expansion coefficient to use for Acre')
        self.general_hyper_group.add_argument('-sa', dest='second_atrous', default=2, type=int, help='The second layer expansion coefficient to use for Acre')
        self.general_hyper_group.add_argument('-ta', dest='third_atrous', default=2, type=int, help='The third layer expansion coefficient to use for Acre')
        self.general_hyper_group.add_argument('-ab', dest='acre_bias', default=True, action='store_true', help='Whether to use bias in the Acre model')

        # basic configs
        self.general_group = self.parser.add_argument_group('Generic')
        self.general_group.add_argument('-mn', dest='model_name', default='TransE', type=str, help='Name of model')
        self.general_group.add_argument('-db', dest='debug', default=False, type=lambda x: (str(x).lower() == 'true'), help='To use debug mode or not.')
        self.general_group.add_argument('-exp', dest='exp', default=False, type=lambda x: (str(x).lower() == 'true'), help='Use Experimental setting extracted from original paper. (use Freebase15k by default)')
        self.general_group.add_argument('-ds', dest='dataset_name', default='Freebase15k', type=str, help='The dataset name (choice: fb15k/wn18/wn18_rr/yago/fb15k_237/ks/nations/umls)')
        self.general_group.add_argument('-dsp', dest='dataset_path', default=None, type=str, help='The path to custom dataset.')
        self.general_group.add_argument('-ld', dest='load_from_data', default=None, type=str, help='The path to the pretrained model.')
        self.general_group.add_argument('-sv', dest='save_model', default=True, type=lambda x: (str(x).lower() == 'true'), help='Save the model!')
        self.general_group.add_argument('-tn', dest='test_num', default=1000, type=int, help='The total number of test triples')
        self.general_group.add_argument('-ts', dest='test_step', default=10, type=int, help='Test every _ epochs')
        self.general_group.add_argument('-t', dest='tmp', default='../intermediate', type=str, help='The folder name to store trained parameters.')
        self.general_group.add_argument('-r', dest='result', default='../results', type=str, help='The folder name to save the results.')
        self.general_group.add_argument('-fig', dest='figures', default='../figures', type=str, help='The folder name to save the figures.')
        self.general_group.add_argument('-plote', dest='plot_embedding', default=False, type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.general_group.add_argument('-plot', dest='plot_entity_only', default=False, type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.general_group.add_argument('-device', dest='device', default='cpu', type=str, choices=['cpu', 'cuda'], help='Device to run pykg2vec (cpu or cuda).')
        self.general_group.add_argument('-npg', dest='num_process_gen', default=2, type=int, help='number of processes used in the Generator.')
        self.general_group.add_argument('-hpf', dest='hp_abs_file', default=None, type=str, help='The path to the hyperparameter configuration YAML file.')
        self.general_group.add_argument('-ssf', dest='ss_abs_file', default=None, type=str, help='The path to the search space configuration YAML file.')
        self.general_group.add_argument('-mt', dest='max_number_trials', default=100, type=int, help='The maximum times of trials for bayesian optimizer.')

    def get_args(self, args):
        """This function parses the necessary arguments.

        This function is called to parse all the necessary arguments.

        Returns:
          object: ArgumentParser object.
        """
        return self.parser.parse_args(args)


class HyperparameterLoader:
    """Hyper parameters loading based datasets and embedding algorithms"""

    _logger = Logger().get_logger(__name__)

    def __init__(self, args):
        self.hyperparams = {}
        self.search_space = {}

        # load hyperparameters from options (file, dir or with pkg.)
        default_search_space_dir = (Path(__file__).resolve().parent)/"searchspaces"
        for config_file in default_search_space_dir.glob('**/*.yaml'):
            self.search_space = self._load_ss_yaml(config_file, self.search_space)
        default_hyperparam_dir = (Path(__file__).resolve().parent)/"hyperparams"
        for config_file in default_hyperparam_dir.glob('**/*.yaml'):
            self.hyperparams = self._load_hp_yaml(config_file, self.hyperparams)

        # load search spaces from options (file, dir or with pkg.)
        if hasattr(args, "hp_abs_file") and args.hp_abs_file is not None:
            self.hyperparams = self._load_hp_yaml(args.hp_abs_file, self.hyperparams)
        if hasattr(args, "ss_abs_file") and args.ss_abs_file is not None:
            self.search_space = self._load_ss_yaml(args.ss_abs_file, self.search_space)

    def load_hyperparameter(self, dataset_name, algorithm):
        d_name = dataset_name.lower()
        a_name = algorithm.lower()

        if d_name in self.hyperparams and a_name in self.hyperparams[d_name]:
            params = self.hyperparams[d_name][a_name]
            return params

        raise Exception("This experimental setting for (%s, %s) has not been configured" % (dataset_name, algorithm))

    def load_search_space(self, algorithm):
        if algorithm in self.search_space:
            return self.search_space[algorithm]
        raise ValueError("Hyperparameter search space is not configured for %s" % algorithm)

    @staticmethod
    def _load_hp_yaml(config_file, hyperparams):
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Cannot find configuration file %s" % config_file)
        if str(config_file).endswith("yaml") or str(config_file).endswith("yml"):
            with open(os.path.abspath(config_file), "r") as file:
                try:
                    config = yaml.safe_load(file)
                    algorithm = config["model_name"].lower()
                    for dataset in config["datasets"]:
                        if dataset["dataset"] in hyperparams:
                            hyperparams[dataset["dataset"]][algorithm] = dataset["parameters"]
                        else:
                            hyperparams = {**hyperparams, **{dataset["dataset"]: {algorithm: dataset["parameters"]}}}
                except yaml.YAMLError:
                    HyperparameterLoader._logger.error("Cannot load configuration: %s" % config_file)
                    raise
        else:
            raise ValueError("Configuration file must have .yaml or .yml extension: %s" % config_file)
        return hyperparams

    @staticmethod
    def _load_ss_yaml(config_file, search_space):
        ''' loading search space configurationfrom yaml file'''
        if not os.path.isfile(config_file):
            raise FileNotFoundError("Cannot find configuration file %s" % config_file)
        if str(config_file).endswith("yaml") or str(config_file).endswith("yml"):
            with open(os.path.abspath(config_file), "r") as file:
                try:
                    config = yaml.safe_load(file)
                    algorithm = config["model_name"].lower()
                    search_space = {**search_space, **{algorithm: HyperparameterLoader._config_tuning_space(config["search_space"])}}
                except yaml.YAMLError:
                    HyperparameterLoader._logger.error("Cannot load configuration: %s" % config_file)
                    raise
        else:
            raise ValueError("Configuration file must have .yaml or .yml extension: %s" % config_file)
        return search_space

    @staticmethod
    def _config_tuning_space(tuning_space_raw):
        if tuning_space_raw is None:
            return None

        hyper_obj = {}
        if "learning_rate" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"learning_rate": hp.loguniform('learning_rate', np.log(tuning_space_raw['learning_rate']['min']), np.log(tuning_space_raw['learning_rate']['max']))}}
        if "hidden_size" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"hidden_size": scope.int(hp.qloguniform('hidden_size', np.log(tuning_space_raw['hidden_size']['min']), np.log(tuning_space_raw['hidden_size']['max']), 1))}}
        if "ent_hidden_size" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"ent_hidden_size": scope.int(hp.qloguniform("ent_hidden_size", np.log(tuning_space_raw['ent_hidden_size']['min']), np.log(tuning_space_raw['ent_hidden_size']['max']), 1))}}
        if "rel_hidden_size" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"rel_hidden_size": scope.int(hp.qloguniform("rel_hidden_size", np.log(tuning_space_raw['rel_hidden_size']['min']), np.log(tuning_space_raw['rel_hidden_size']['max']), 1))}}
        if "batch_size" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"batch_size": scope.int(hp.qloguniform("batch_size", np.log(tuning_space_raw['batch_size']['min']), np.log(tuning_space_raw['batch_size']['max']), 1))}}
        if "margin" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"margin": hp.uniform("margin", tuning_space_raw["margin"]["min"], tuning_space_raw["margin"]["max"])}}
        if "lmbda" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"lmbda": hp.loguniform('lmbda', np.log(tuning_space_raw["lmbda"]["min"]), np.log(tuning_space_raw["lmbda"]["max"]))}}
        if "distance_measure" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"distance_measure": hp.choice('distance_measure', tuning_space_raw["distance_measure"])}}
        if "cmax" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"cmax": hp.loguniform('cmax', np.log(tuning_space_raw["cmax"]["min"]), np.log(tuning_space_raw["cmax"]["max"]))}}
        if "cmin" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"cmin": hp.loguniform('cmin', np.log(tuning_space_raw["cmin"]["min"]), np.log(tuning_space_raw["cmin"]["max"]))}}
        if "optimizer" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"optimizer": hp.choice("optimizer", tuning_space_raw["optimizer"])}}
        if "bilinear" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"bilinear": hp.choice('bilinear', tuning_space_raw["bilinear"])}}
        if "epochs" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"epochs": hp.choice("epochs", tuning_space_raw["epochs"])}}
        if "feature_map_dropout" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"feature_map_dropout": hp.choice('feature_map_dropout', tuning_space_raw["feature_map_dropout"])}}
        if "input_dropout" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"input_dropout": hp.choice('input_dropout', tuning_space_raw["input_dropout"])}}
        if "hidden_dropout" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"hidden_dropout": hp.choice('hidden_dropout', tuning_space_raw["hidden_dropout"])}}
        if "use_bias" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"use_bias": hp.choice('use_bias', tuning_space_raw["use_bias"])}}
        if "label_smoothing" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"label_smoothing": hp.choice('label_smoothing', tuning_space_raw["label_smoothing"])}}
        if "lr_decay" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"lr_decay": hp.choice('lr_decay', tuning_space_raw["lr_decay"])}}
        if "l1_flag" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"l1_flag": hp.choice('l1_flag', tuning_space_raw["l1_flag"])}}
        if "sampling" in tuning_space_raw:
            hyper_obj = {**hyper_obj, **{"sampling": hp.choice('sampling', tuning_space_raw["sampling"])}}

        return hyper_obj

class Importer:
    """The class defines methods for importing pykg2vec modules.

    Importer is used to defines the maps for the algorithm names and
    provides methods for loading configuration and models.

    Attributes:
        model_path (str): Path where the models are defined.
        config_path (str): Path where the configuration for each models are defineds.
        modelMap (dict): This map transforms the names of model to the actual class names.
        configMap (dict): This map transforms the input config names to the actuall config class names.

    Examples:
        >>> from pykg2vec import Importer
        >>> config_def, model_def = Importer().import_model_config('transe')
        >>> config = config_def()
        >>> model = model_def(config)

    """
    _logger = Logger().get_logger(__name__)

    def __init__(self):
        self.model_path = "pykg2vec.models"
        self.modelMap = {
                         "acre": "projection.AcrE",
                         "analogy": "pointwise.ANALOGY",
                         "complex": "pointwise.Complex",
                         "complexn3": "pointwise.ComplexN3",
                         "conve": "projection.ConvE",
                         "convkb": "pointwise.ConvKB",
                         "cp": "pointwise.CP",
                         "distmult": "pointwise.DistMult",
                         "hole": "pairwise.HoLE",
                         "hyper": "projection.HypER",
                         "interacte": "projection.InteractE",
                         "kg2e": "pairwise.KG2E",
                         "murp": "pointwise.MuRP",
                         "ntn": "pairwise.NTN",
                         "octonione": "pointwise.OctonionE",
                         "proje_pointwise": "projection.ProjE_pointwise",
                         "quate": "pointwise.QuatE",
                         "rescal": "pairwise.Rescal",
                         "rotate": "pairwise.RotatE",
                         "simple": "pointwise.SimplE",
                         "simple_ignr": "pointwise.SimplE_ignr",
                         "slm": "pairwise.SLM",
                         "sme": "pairwise.SME",
                         "sme_bl": "pairwise.SME_BL",
                         "transd": "pairwise.TransD",
                         "transe": "pairwise.TransE",
                         "transh": "pairwise.TransH",
                         "transm": "pairwise.TransM",
                         "transr": "pairwise.TransR",
                         "tucker": "projection.TuckER",
                         }
        self.config_path = "pykg2vec.config"

    def import_model_config(self, name):
        """This function imports models and configuration.

        This function is used to dynamically import the modules within
        pykg2vec.

        Args:
          name (str): The input to the module is either name of the model or the configuration file. The strings are converted to lowercase to makesure the user inputs can easily be matched to the names of the models and the configuration class.

        Returns:
          object: Configuration and model object after it is successfully loaded.

          `config_obj` (object): Returns the configuration class object of the corresponding algorithm.
          `model_obj` (object): Returns the model class object of the corresponding algorithm.

        Raises:
          ModuleNotFoundError: It raises a module not found error if the configuration or the model cannot be found.
        """
        try:
            config_obj = getattr(importlib.import_module(self.config_path), "Config")
            if name in self.modelMap:
                splited_path = self.modelMap[name].split('.')
            else:
                raise ValueError("%s model has not been implemented. please select from: %s" % (name, ' '.join(map(lambda x: str(x).split(".")[1], self.modelMap.values()))))
            model_obj = getattr(importlib.import_module(self.model_path + ".%s" % splited_path[0]), splited_path[1])
            return config_obj, model_obj
        except ModuleNotFoundError:
            self._logger.error("%s model has not been implemented. please select from: %s" % (name, ' '.join(map(str.split(".")[1], self.modelMap.values()))))
            raise ValueError("%s model has not been implemented. please select from: %s" % (name, ' '.join(map(str.split(".")[1], self.modelMap.values()))))
