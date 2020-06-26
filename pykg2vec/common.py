import importlib
from enum import Enum
from argparse import ArgumentParser
from pykg2vec.utils.logger import Logger

class Monitor(Enum):
    MEAN_RANK = "mr"
    FILTERED_MEAN_RANK = "fmr"
    MEAN_RECIPROCAL_RANK = "mrr"
    FILTERED_MEAN_RECIPROCAL_RANK = "fmrr"


class TrainingStrategy(Enum):
    PROJECTION_BASED = "projection_based"   # matching models with neural network
    PAIRWISE_BASED = "pairwise_based"       # translational distance models
    POINTWISE_BASED = "pointwise_based"     # semantic matching models


class KGETuneArgParser:
    """The class defines the arguements accepted for the bayesian optimizer.

      KGETuneArgParser utilizes the ArgumentParser module and add the arguments
      accepted for tuning the model.

      Args:
         model (str): Name of the model/algorithm to be tuned.
         debug (bool): If True, tunes the model in debugging mode.

      Examples:
          >>> from pykg2vec.hyperparams import KGETuneArgParser
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


class KGEArgParser:
    """The class implements the argument parser for the pykg2vec.

    KGEArgParser defines all the necessary arguements for the global and local 
    configuration of all the modules.

    Attributes:
        general_group (object): It parses the general arguements used by most of the modules.
        general_hyper_group (object): It parses the arguments for the hyper-parameter tuning.
        SME_group (object): It parses the arguments for SME and KG2E algorithms.
        conv_group (object): It parses the arguments for convE algorithms.
        misc_group (object): It prases other necessary arguments.
    
    Examples:
        >>> from pykg2vec.config import KGEArgParser
        >>> args = KGEArgParser().get_args()
    """

    def __init__(self):
        self.parser = ArgumentParser(description='Knowledge Graph Embedding tunable configs.')

        
        ''' arguments regarding TransG '''
        self.TransG_group = self.parser.add_argument_group('TransG function selection')
        self.TransG_group.add_argument('-th', dest='training_threshold', default=3.5, type=float, help="Training Threshold for updateing the clusters.")
        self.TransG_group.add_argument('-nc', dest='ncluster', default=4, type=int, help="Number of clusters")
        self.TransG_group.add_argument('-crp', dest='crp_factor', default=0.01, type=float, help="Chinese Restaurant Process Factor.")
        self.TransG_group.add_argument('-stb', dest='step_before', default=10, type=int, help="Steps before")
        self.TransG_group.add_argument('-wn', dest='weight_norm', default=False, type=lambda x: (str(x).lower() == 'true'), help="normalize the weights!")

        ''' arguments regarding SME and KG2E '''
        self.SME_group = self.parser.add_argument_group('SME KG2E function selection')
        self.SME_group.add_argument('-func', dest='function', default='bilinear', type=str, help="The name of function used in SME model.")
        self.SME_group.add_argument('-cmax', dest='cmax', default=0.05, type=float, help="The parameter for clipping values for KG2E.")
        self.SME_group.add_argument('-cmin', dest='cmin', default=5.00, type=float, help="The parameter for clipping values for KG2E.")

        ''' for conve '''
        self.conv_group = self.parser.add_argument_group('ConvE specific Hyperparameters')
        self.conv_group.add_argument('-fmd', dest='feature_map_dropout', default=0.2, type=float, help="feature map dropout value used in ConvE.")
        self.conv_group.add_argument('-idt', dest="input_dropout", default=0.3, type=float, help="input dropout value used in ConvE.")
        self.conv_group.add_argument('-hdt', dest="hidden_dropout", default=0.3, type=float, help="hidden dropout value used in ConvE.")
        self.conv_group.add_argument('-hdt1', dest="hidden_dropout1", default=0.4, type=float, help="hidden dropout value used in TuckER.")
        self.conv_group.add_argument('-hdt2', dest="hidden_dropout2", default=0.5, type=float, help="hidden dropout value used in TuckER.")
        self.conv_group.add_argument('-lbs', dest='label_smoothing', default=0.1, type=float, help="The parameter used in label smoothing.")

        '''for convKB'''
        self.convkb_group = self.parser.add_argument_group('ConvKB specific Hyperparameters')
        self.convkb_group.add_argument('-fsize', dest='filter_sizes', default=[1,2,3],nargs='+', type=int, help='Filter sizes to be used in convKB which acts as the widths of the kernals')
        self.convkb_group.add_argument('-fnum', dest='num_filters', default=50, type=int, help='Filter numbers to be used in convKB')

        '''for RotatE'''
        self.rotate_group = self.parser.add_argument_group('RotatE specific Hyperparameters')
        self.rotate_group.add_argument('-al', dest='alpha', default=0.1, type=float, help='The alpha used in self-adversarial negative sampling.')

        ''' arguments regarding hyperparameters '''
        self.general_hyper_group = self.parser.add_argument_group('Generic Hyperparameters')
        self.general_hyper_group.add_argument('-lmda', dest='lmbda', default=0.1, type=float, help='The lmbda for regularization.')
        self.general_hyper_group.add_argument('-b',   dest='batch_size', default=128, type=int, help='training batch size')
        self.general_hyper_group.add_argument('-mg',  dest='margin', default=0.8, type=float, help='Margin to take')
        self.general_hyper_group.add_argument('-opt', dest='optimizer', default='adam', type=str, help='optimizer to be used in training.')
        self.general_hyper_group.add_argument('-s',   dest='sampling', default='uniform', type=str, help='strategy to do negative sampling.')
        self.general_hyper_group.add_argument('-ngr', dest='neg_rate', default=1, type=int, help='The number of negative samples generated per positve one.')
        self.general_hyper_group.add_argument('-l',   dest='epochs', default=100, type=int, help='The total number of Epochs')
        self.general_hyper_group.add_argument('-lr',  dest='learning_rate', default=0.01, type=float,help='learning rate')
        self.general_hyper_group.add_argument('-k',   dest='hidden_size', default=50, type=int,help='Hidden embedding size.')
        self.general_hyper_group.add_argument('-km',  dest='ent_hidden_size', default=50, type=int, help="Hidden embedding size for entities.")
        self.general_hyper_group.add_argument('-kr',  dest='rel_hidden_size', default=50, type=int, help="Hidden embedding size for relations.")
        self.general_hyper_group.add_argument('-k2',  dest='hidden_size_1', default=10, type=int, help="Hidden embedding size for relations.")

        self.general_hyper_group.add_argument('-l1',  dest='l1_flag', default=True, type=lambda x: (str(x).lower() == 'true'),help='The flag of using L1 or L2 norm.')

        ''' working environments '''
        self.environment_group = self.parser.add_argument_group('Working Environments')
        self.environment_group.add_argument('-gp',  dest='gpu_frac', default=0.8, type=float, help='GPU fraction to use')
        self.environment_group.add_argument('-npg', dest='num_process_gen', default=2, type=int, help='number of processes used in the Generator.')

        ''' basic configs '''
        self.general_group = self.parser.add_argument_group('Generic')
        self.general_group.add_argument('-mn',    dest='model_name', default='TransE', type=str, help='Name of model')
        self.general_group.add_argument('-db',    dest='debug',      default=False, type=lambda x: (str(x).lower() == 'true'), help='To use debug mode or not.')
        self.general_group.add_argument('-exp',   dest='exp', default=False, type=lambda x: (str(x).lower() == 'true'), help='Use Experimental setting extracted from original paper. (use with -ds or FB15k in default)')
        self.general_group.add_argument('-ds',    dest='dataset_name', default='Freebase15k', type=str, help='The dataset name (choice: fb15k/wn18/wn18_rr/yago/fb15k_237/ks/nations/umls)')
        self.general_group.add_argument('-dsp',   dest='dataset_path', default=None, type=str, help='The path to custom dataset.')
        self.general_group.add_argument('-ld',    dest='load_from_data', default=False, type=lambda x: (str(x).lower() == 'true'), help='load from tensroflow saved data!')
        self.general_group.add_argument('-sv',    dest='save_model', default=True, type=lambda x: (str(x).lower() == 'true'), help='Save the model!')
        self.general_group.add_argument('-tn',    dest='test_num', default=1000, type=int, help='The total number of test triples')
        self.general_group.add_argument('-ts',    dest='test_step', default=10, type=int, help='Test every _ epochs')
        self.general_group.add_argument('-t',     dest='tmp', default='../intermediate', type=str,help='The folder name to store trained parameters.')
        self.general_group.add_argument('-r',     dest='result', default='../results', type=str,help="The folder name to save the results.")
        self.general_group.add_argument('-fig',   dest='figures', default='../figures', type=str,help="The folder name to save the figures.")
        self.general_group.add_argument('-plote', dest='plot_embedding', default=False,type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.general_group.add_argument('-plot',  dest='plot_entity_only', default=False,type=lambda x: (str(x).lower() == 'true'), help='Plot the entity only!')
        self.general_group.add_argument('-device',dest='device', default='cpu', type=str, choices=['cpu', 'cuda'], help="Device to run pykg2vec (cpu or cuda).")

    def get_args(self, args):
      """This function parses the necessary arguments.

      This function is called to parse all the necessary arguments. 

      Returns:
          object: ArgumentParser object.
      """
      return self.parser.parse_args(args)


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
        self.config_path = "pykg2vec.config"
        self.hyper_path = "pykg2vec.hyperparams"

        self.modelMap = {"analogy": "pointwise.ANALOGY",
                         "complex": "pointwise.Complex",
                         "complexn3": "pointwise.ComplexN3",
                         "conve": "projection.ConvE",
                         "convkb": "pointwise.ConvKB",
                         "cp": "pointwise.CP",
                         "hole": "pairwise.HoLE",
                         "distmult": "pointwise.DistMult",
                         "kg2e": "pairwise.KG2E",
                         "kg2e_el": "pairwise.KG2E_EL",
                         "ntn": "pairwise.NTN",
                         "proje_pointwise": "projection.ProjE_pointwise",
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
                         "tucker": "projection.TuckER"}

        self.configMap = {"analogy": "ANALOGYConfig",
                          "complex": "ComplexConfig",
                          "complexn3": "ComplexConfig",
                          "conve": "ConvEConfig",
                          "convkb": "ConvKBConfig",
                          "cp": "CPConfig",
                          "hole": "HoLEConfig",
                          "distmult": "DistMultConfig",
                          "kg2e": "KG2EConfig",
                          "kg2e_el": "KG2EConfig",
                          "ntn": "NTNConfig",
                          "proje_pointwise": "ProjE_pointwiseConfig",
                          "rescal": "RescalConfig",
                          "rotate": "RotatEConfig",
                          "simple": "SimplEConfig",
                          "simple_ignr": "SimplEConfig",
                          "slm": "SLMConfig",
                          "sme": "SMEConfig",
                          "sme_bl": "SMEConfig",
                          "transd": "TransDConfig",
                          "transe": "TransEConfig",
                          "transg": "TransGConfig",
                          "transh": "TransHConfig",
                          "transm": "TransMConfig",
                          "transr": "TransRConfig",
                          "tucker": "TuckERConfig"}
        
        self.hyperparamMap = {"analogy": "ANALOGYParams",
                              "complex": "ComplexParams",
                              "complexn3": "ComplexParams",
                              "conve": "ConvEParams",
                              "cp": "CPParams",
                              "hole": "HoLEParams",
                              "distmult": "DistMultParams",
                              "kg2e": "KG2EParams",
                              "kg2e_el": "KG2EParams",
                              "ntn": "NTNParams",
                              "proje_pointwise": "ProjE_pointwiseParams",
                              "rescal": "RescalParams",
                              "rotate": "RotatEParams",
                              "simple": "SimplEParams",
                              "simple_ignr": "SimplEParams",
                              "slm": "SLMParams",
                              "sme": "SMEParams",
                              "sme_bl": "SMEParams",
                              "transd": "TransDParams",
                              "transe": "TransEParams",
                              "transg": "TransGParams",
                              "transh": "TransHParams",
                              "transm": "TransMParams",
                              "transr": "TransRParams",
                              "tucker": "TuckERParams"}

    def import_hyperparam_config(self, name):
        hyper_obj = None 

        try:
            hyper_obj = getattr(importlib.import_module(self.hyper_path), self.hyperparamMap[name])
          
        except ModuleNotFoundError:
            self._logger.error("%s model  has not been implemented. please select from: %s" % (
            name, ' '.join(map(str, self.hyperparamMap.values()))))

        return hyper_obj

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
        config_obj = None
        model_obj = None
        try:
            config_obj = getattr(importlib.import_module(self.config_path), self.configMap[name])
            splited_path = self.modelMap[name].split('.')
            model_obj  = getattr(importlib.import_module(self.model_path + ".%s" % splited_path[0]), splited_path[1])

        except ModuleNotFoundError:
            self._logger.error("%s model  has not been implemented. please select from: %s" % (
            name, ' '.join(map(str, self.modelMap.values()))))

        return config_obj, model_obj
