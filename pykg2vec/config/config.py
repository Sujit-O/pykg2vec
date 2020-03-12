"""
config.py
==========

This module consists of definition of the necessary configuration parameters for all the 
core algorithms. The parameters are seprated into global parameters which are common
across all the algorithms, and local parameters which are specific to the algorithms.

"""

import tensorflow as tf
from argparse import ArgumentParser
import importlib

from pykg2vec.utils.kgcontroller import KnowledgeGraph, KGMetaData
from pykg2vec.config.hyperparams import HyperparamterLoader

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
        >>> from pykg2vec.config.config import Importer
        >>> config_def, model_def = Importer().import_model_config('transe')
        >>> config = config_def()
        >>> model = model_def(config)

    """
    def __init__(self):
        self.model_path = "pykg2vec.core"
        self.config_path = "pykg2vec.config.config"

        self.modelMap = {"complex": "Complex",
                         "conve": "ConvE",
                         "convkb": "ConvKB",
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
                          "convkb": "ConvKBConfig",
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
          model_obj  = getattr(importlib.import_module(self.model_path + ".%s" % self.modelMap[name]),
                              self.modelMap[name])
      except ModuleNotFoundError:
          print("%s model  has not been implemented. please select from: %s" % (
          name, ' '.join(map(str, self.modelMap.values()))))

      return config_obj, model_obj


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
        >>> from pykg2vec.config.config import KGEArgParser
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
        self.conv_group.add_argument('-lbs', dest='label_smoothing', default=0.1, type=float, help="The parameter used in label smoothing.")

        '''for convKB'''
        self.convkb_group = self.parser.add_argument_group('ConvKB specific Hyperparameters')
        self.convkb_group.add_argument('-fsize', dest='filter_sizes', default=[1,2,3],nargs='+', type=int, help='Filter sizes to be used in convKB which acts as the widths of the kernals')
        self.convkb_group.add_argument('-fnum', dest='num_filters', default=50, type=int, help='Filter numbers to be used in convKB')

        ''' arguments regarding hyperparameters '''
        self.general_hyper_group = self.parser.add_argument_group('Generic Hyperparameters')
        self.general_hyper_group.add_argument('-lmda', dest='lmbda', default=0.1, type=float, help='The lmbda for regularization.')
        self.general_hyper_group.add_argument('-b',   dest='batch_training', default=128, type=int, help='training batch size')
        self.general_hyper_group.add_argument('-mg',  dest='margin', default=0.8, type=float, help='Margin to take')
        self.general_hyper_group.add_argument('-opt', dest='optimizer', default='adam', type=str, help='optimizer to be used in training.')
        self.general_hyper_group.add_argument('-s',   dest='sampling', default='uniform', type=str, help='strategy to do negative sampling.')
        self.general_hyper_group.add_argument('-ngr', dest='negrate', default=1, type=int, help='The number of negative samples generated per positve one.')
        self.general_hyper_group.add_argument('-l',   dest='epochs', default=100, type=int, help='The total number of Epochs')
        self.general_hyper_group.add_argument('-lr',  dest='learning_rate', default=0.01, type=float,help='learning rate')
        self.general_hyper_group.add_argument('-k',   dest='hidden_size', default=50, type=int,help='Hidden embedding size.')
        self.general_hyper_group.add_argument('-km',  dest='ent_hidden_size', default=50, type=int, help="Hidden embedding size for entities.")
        self.general_hyper_group.add_argument('-kr',  dest='rel_hidden_size', default=50, type=int, help="Hidden embedding size for relations.")
        self.general_hyper_group.add_argument('-l1',  dest='l1_flag', default=True, type=lambda x: (str(x).lower() == 'true'),help='The flag of using L1 or L2 norm.')
        self.general_hyper_group.add_argument('-c',   dest='C', default=0.0125, type=float, help='The parameter C used in transH.')

        ''' working environments '''
        self.environment_group = self.parser.add_argument_group('Working Environments')
        self.environment_group.add_argument('-gp',  dest='gpu_frac', default=0.8, type=float, help='GPU fraction to use')
        self.environment_group.add_argument('-npg', dest='num_process_gen', default=2, type=int, help='number of processes used in the Generator.')
        self.environment_group.add_argument('-npe', dest='num_process_evl', default=1, type=int, help='number of processes used in the Evaluator.')

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
        self.general_group.add_argument('-els',   dest='early_stop_epoch', default=50, type=int, help='Interval of performing early stop check. ')

    def get_args(self, args):
      """This function parses the necessary arguments.

      This function is called to parse all the necessary arguments. 

      Returns:
          object: ArgumentParser object.
      """
      return self.parser.parse_args(args)


class BasicConfig:
    """The class defines the basic configuration for the pykg2vec.

    BasicConfig consists of the necessary parameter description used by all the 
    modules including the algorithms and utility functions.

    Args:
      test_step (int): Testing is carried out every test_step.
      test_num (int): Number of triples that will be tested during evaluation.
      triple_num (int): Number of triples that will be used for plotting the embedding.
      tmp (Path Object): Path where temporary model information is stored.
      result (Path Object): Gives the path where the result will be saved.
      figures (Path Object): Gives the path where the figures will be saved.
      gpu_fraction (float): Amount of GPU fraction that will be made available for training and inference.
      gpu_allow_growth (bool): If True, allocates only necessary GPU memory and grows as required later.
      loadFromData (bool): If True, loads the model parameters if available from memory.
      save_model (True): If True, store the trained model parameters.
      disp_summary (bool): If True, display the summary before and after training the algorithm.
      disp_result (bool): If True, displays result while training.
      plot_embedding (bool): If True, will plot the embedding after performing t-SNE based dimensionality reduction.
      log_training_placement (bool): If True, allows us to find out which devices the operations and tensors are assigned to.
      plot_training_result (bool): If True, plots the loss values stored during training.
      plot_testing_result (bool): If True, it will plot all the testing result such as mean rank, hit ratio, etc.
      plot_entity_only (bool): If True, plots the t-SNE reduced embdding of the entities in a figure.
      full_test_flag (bool): It True, performs a full test after completing the training for full epochs.
      hits (List): Gives the list of integer for calculating hits.
      knowledge_graph (Object): It prepares and holds the instance of the knowledge graph dataset.
      kg_meta (object): Stores the statistics metadata of the knowledge graph.
    
    """
    def __init__(self, args):
        # Training and evaluating related variables
        self.test_step = args.test_step
        self.full_test_flag = (self.test_step == 0)
        self.test_num = args.test_num
        self.hits = [1, 3, 5, 10]
        self.loadFromData = args.load_from_data
        self.save_model = args.save_model
        self.disp_summary = True
        self.disp_result = False
        
        self.early_stop_epoch = args.early_stop_epoch # the interval of early stop checking. 
        self.patience = 3 # should make this configurable as well.
        
        # Visualization related, 
        # p.s. the visualizer is disable for most of the KGE methods for now. 
        self.disp_triple_num = 20
        self.plot_embedding = args.plot_embedding
        self.plot_training_result = True
        self.plot_testing_result = True
        self.plot_entity_only = args.plot_entity_only
        
        # Working environment variables.
        self.num_process_gen = args.num_process_gen
        self.num_process_evl = args.num_process_evl
        self.log_device_placement = False
        self.gpu_fraction = args.gpu_frac
        self.gpu_allow_growth = True
        # self.gpu_config = tf.ConfigProto(log_device_placement=self.log_device_placement)
        # self.gpu_config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction
        # self.gpu_config.gpu_options.allow_growth = self.gpu_allow_growth

        # Knowledge Graph Information
        self.custom_dataset_path = args.dataset_path
        self.knowledge_graph = KnowledgeGraph(dataset=self.data, custom_dataset_path=self.custom_dataset_path)
        self.kg_meta = self.knowledge_graph.kg_meta
        
        # The results of training will be stored in the following folders 
        # which are relative to the parent folder (the path of the dataset).
        dataset_path = self.knowledge_graph.dataset.dataset_path
        self.path_tmp =  dataset_path / 'intermediate'
        self.path_tmp.mkdir(parents=True, exist_ok=True)
        self.path_result = dataset_path / 'results'
        self.path_result.mkdir(parents=True, exist_ok=True)
        self.path_figures = dataset_path / 'figures'
        self.path_figures.mkdir(parents=True, exist_ok=True)
        self.path_embeddings = dataset_path / 'embeddings'
        self.path_embeddings.mkdir(parents=True, exist_ok=True)

        # debugging information 
        self.debug = args.debug

    def summary(self):
        """Function to print the summary."""
        print("\n------------------Global Setting--------------------")
        # Acquire the max length and add four more spaces
        maxspace = len(max([k for k in self.__dict__.keys()])) +20
        for key, val in self.__dict__.items():
            if key in self.__dict__['hyperparameters']:
                continue

            if isinstance(val, (KGMetaData, KnowledgeGraph)) or key.startswith('gpu') or key.startswith('hyperparameters'):
                continue

            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            print("%s : %s"%(key, val))
        print("---------------------------------------------------")

    def summary_hyperparameter(self, model_name):
        """Function to print the hyperparameter summary."""
        print("\n-----------%s Hyperparameter Setting-------------"%(model_name))
        maxspace = len(max([k for k in self.hyperparameters.keys()])) + 15
        for key,val in self.hyperparameters.items():
            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            print("%s : %s" % (key, val))
        print("---------------------------------------------------")


class TransEConfig(BasicConfig):
    """This class defines the configuration for the TransE Algorithm.

    TransEConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for both entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.L1_flag = args.l1_flag
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.margin = args.margin
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'transe')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class HoLEConfig(BasicConfig):
    """This class defines the configuration for the HoLE Algorithm.

    HoLEConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for both entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args=None):
        self.learning_rate = args.learning_rate
        self.L1_flag = args.l1_flag
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.margin = args.margin
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'hole')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class TransRConfig(BasicConfig):
    """This class defines the configuration for the TransR Algorithm.

    TransRConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      ent_hidden_size (int): Defines the size of the latent dimension for entities.
      rel_hidden_size (int): Defines the size of the latent dimension for relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """

    def __init__(self, args=None):
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
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'transr')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class TransDConfig(BasicConfig):
    """This class defines the configuration for the TransD Algorithm.

    TransDConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      ent_hidden_size (int): Defines the size of the latent dimension for entities.
      rel_hidden_size (int): Defines the size of the latent dimension for relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args=None):
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
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'transd')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class TransMConfig(BasicConfig):
    """This class defines the configuration for the TransM Algorithm.

    TransMConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """

    def __init__(self, args=None):
        self.learning_rate = args.learning_rate
        self.L1_flag = args.l1_flag
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.margin = args.margin
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'transm')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class TransHConfig(BasicConfig):
    """This class defines the configuration for the TransH Algorithm.

    TransHConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      C (float) : It is used to weigh the importance of soft-constraints.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """

    def __init__(self, args=None):
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
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'transh')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'C': self.C,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class RescalConfig(BasicConfig):
    """This class defines the configuration for the Rescal Algorithm.

    RescalConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args=None):
        self.learning_rate = args.learning_rate
        self.L1_flag = args.l1_flag
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.margin = args.margin
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'rescal')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class SMEConfig(BasicConfig):
    """This class defines the configuration for the SME Algorithm.

    SMEConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      ent_hidden_size (int): Defines the size of the latent dimension for entities.
      rel_hidden_size (int): Defines the size of the latent dimension for relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
      bilinear (bool): If true uses bilnear transformation for loss else uses linear.
    
    """
    def __init__(self, args=None):
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
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'sme')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'bilinear': self.bilinear,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class NTNConfig(BasicConfig):
    """This class defines the configuration for the NTN Algorithm.

    NTNConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      ent_hidden_size (int): Defines the size of the latent dimension for entities.
      rel_hidden_size (int): Defines the size of the latent dimension for relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args=None):
        self.lmbda = args.lmbda
        self.learning_rate = args.learning_rate
        self.L1_flag = args.l1_flag
        self.ent_hidden_size = args.ent_hidden_size
        self.rel_hidden_size = args.rel_hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'ntn')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'ent_hidden_size': self.ent_hidden_size,
            'rel_hidden_size': self.rel_hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class SLMConfig(BasicConfig):
    """This class defines the configuration for the SLM Algorithm.

    SLMConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      ent_hidden_size (int): Defines the size of the latent dimension for entities.
      rel_hidden_size (int): Defines the size of the latent dimension for relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """

    def __init__(self, args=None):
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
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'slm')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class RotatEConfig(BasicConfig):
    """This class defines the configuration for the RotatE Algorithm.

    RotatEDConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args=None):
        self.learning_rate = args.learning_rate
        self.L1_flag = args.l1_flag
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.margin = args.margin
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'rotate')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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


class KG2EConfig(BasicConfig):
    """This class defines the configuration for the KG2E Algorithm.

    KG2EConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
      bilinear (bool): If True, sets the transformaton to be bilinear.
      distance_measure (str): Uses either kl_divergence or expected_likelihood as distance measure.
      cmax (float): Sets the upper clipping range for the embedding.
      cmin (float): Sets the lower clipping range for the embedding.
    
    """
    def __init__(self, args=None):
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
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'kg2e')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

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
            'cmin': self.cmin,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class ComplexConfig(BasicConfig):
    """This class defines the configuration for the Complex Algorithm.

    ComplexConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      lambda (float) : Weigth applied to the regularization in the loss function.
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args=None):
        self.lmbda = args.lmbda
        self.learning_rate = args.learning_rate
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'complex')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class DistMultConfig(BasicConfig):
    """This class defines the configuration for the DistMult Algorithm.

    DistMultConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      lambda (float) : Weigth applied to the regularization in the loss function.
      feature_map_dropout (float) : Sets the dropout for the feature layer.
      input_dropout (float) : Sets the dropout rate for the input layer.
      hidden_dropout (float) : Sets the dropout rate for the hidden layer.
      use_bias (bool) : If true, adds bias in the end before the activation.
      label_smoothing (float) : Smoothens the label from 0 and 1 by adding it on the 0 and subtracting it from 1. 
      lr_decay (float) : Sets the learning decay rate for optimization.
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """

    def __init__(self, args=None):
        self.lmbda = args.lmbda
        self.learning_rate = args.learning_rate
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'distmult')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class ProjE_pointwiseConfig(BasicConfig):
    """This class defines the configuration for the ProjE Algorithm.

    ProjE_pointwiseConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      lambda (float) : Weigth applied to the regularization in the loss function.
      feature_map_dropout (float) : Sets the dropout for the feature layer.
      input_dropout (float) : Sets the dropout rate for the input layer.
      hidden_dropout (float) : Sets the dropout rate for the hidden layer.
      use_bias (bool) : If true, adds bias in the end before the activation.
      label_smoothing (float) : Smoothens the label from 0 and 1 by adding it on the 0 and subtracting it from 1. 
      lr_decay (float) : Sets the learning decay rate for optimization.
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """

    def __init__(self, args=None):
        self.lmbda = args.lmbda
        self.hidden_dropout = args.hidden_dropout
        self.learning_rate = args.learning_rate
        self.L1_flag = args.l1_flag
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.margin = args.margin
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'proje_po')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'hidden_dropout': self.hidden_dropout,
            'learning_rate': self.learning_rate,
            'L1_flag': self.L1_flag,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class ConvKBConfig(BasicConfig):
    """This class defines the configuration for the ConvKB Algorithm.

    ConvKBConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      lambda (float) : Weigth applied to the regularization in the loss function.
      feature_map_dropout (float) : Sets the dropout for the feature layer.
      input_dropout (float) : Sets the dropout rate for the input layer.
      hidden_dropout (float) : Sets the dropout rate for the hidden layer.
      use_bias (bool) : If true, adds bias in the end before the activation.
      label_smoothing (float) : Smoothens the label from 0 and 1 by adding it on the 0 and subtracting it from 1. 
      lr_decay (float) : Sets the learning decay rate for optimization.
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args=None):
        self.lmbda = args.lmbda
        self.filter_sizes =args.filter_sizes
        self.num_filters = args.num_filters
        self.learning_rate = args.learning_rate
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'convkb')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'filter_sizes' : self.filter_sizes,
            'num_filters' :self.num_filters,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)


class ConvEConfig(BasicConfig):
    """This class defines the configuration for the ConvE Algorithm.

    ConvEConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      lambda (float) : Weigth applied to the regularization in the loss function.
      feature_map_dropout (float) : Sets the dropout for the feature layer.
      input_dropout (float) : Sets the dropout rate for the input layer.
      hidden_dropout (float) : Sets the dropout rate for the hidden layer.
      use_bias (bool) : If true, adds bias in the end before the activation.
      label_smoothing (float) : Smoothens the label from 0 and 1 by adding it on the 0 and subtracting it from 1. 
      lr_decay (float) : Sets the learning decay rate for optimization.
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args=None):

        self.feature_map_dropout = args.feature_map_dropout
        self.input_dropout = args.input_dropout
        self.hidden_dropout = args.hidden_dropout
        self.label_smoothing = args.label_smoothing
        self.learning_rate = args.learning_rate
        # TODO: Currently conve can only have k=50, 100, or 200
        # self.hidden_size = args.hidden_size
        self.hidden_size = 200
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = args.negrate

        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'conve')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

        self.hyperparameters = {
            'feature_map_dropout': self.feature_map_dropout,
            'input_dropout': self.input_dropout,
            'hidden_dropout': self.hidden_dropout,
            'label_smoothing': self.label_smoothing,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'data': self.data,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)

class TuckERConfig(BasicConfig):
    """This class defines the configuration for the TuckER Algorithm.

    TuckERConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      lambda (float) : Weigth applied to the regularization in the loss function.
      feature_map_dropout (float) : Sets the dropout for the feature layer.
      input_dropout (float) : Sets the dropout rate for the input layer.
      hidden_dropout (float) : Sets the dropout rate for the hidden layer.
      use_bias (bool) : If true, adds bias in the end before the activation.
      label_smoothing (float) : Smoothens the label from 0 and 1 by adding it on the 0 and subtracting it from 1. 
      lr_decay (float) : Sets the learning decay rate for optimization.
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
    
    """
    def __init__(self, args=None):

        if args is None or args.exp is True:
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
            self.neg_rate = 1

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
            self.neg_rate = args.negrate

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
            'neg_rate': self.neg_rate,
        }

        BasicConfig.__init__(self, args)

class TransGConfig(BasicConfig):
    """This class defines the configuration for the TransG Algorithm.

    TransGConfig inherits the BasicConfig and defines the local arguements used in the
    algorithm.

    Attributes:
      hyperparameters (dict): Defines the dictionary of hyperparameters to be used by bayesian optimizer for tuning.

    Args:
      learning_rate (float): Defines the learning rate for the optimization.
      L1_flag (bool): If True, perform L1 regularization on the model parameters.
      hidden_size (int): Defines the size of the latent dimension for both entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      margin (float): Defines the margin used between the positive and negative triple loss.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
      training_threshold (float): Defines the threshold to be used to update the clusters for TransG.
      ncluster (int): Defines the initial cluster for the relation.
      CRP_factor (float): Chinese Restaurant Process Factor.
      weight_norm (bool): If True, normalizes the weights.
      step_before (int): Defines the number of steps before which the update is cluster is not performed.
    
    """
    def __init__(self, args=None):
        if args is None or args.exp is True:
            # the exp setting for TransG (only for Freebase15k now)
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
            self.CRP_factor = 0.1
            self.weight_norm = True
            self.step_before = 10
            self.neg_rate = 1

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
            self.neg_rate = args.negrate

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
            'weight_norm':self.weight_norm,
            'neg_rate': self.neg_rate,

        }
        BasicConfig.__init__(self, args)