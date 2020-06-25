"""
config.py
==========

This module consists of definition of the necessary configuration parameters for all the 
core algorithms. The parameters are seprated into global parameters which are common
across all the algorithms, and local parameters which are specific to the algorithms.

"""


from pykg2vec.data.kgcontroller import KnowledgeGraph, KGMetaData
from pykg2vec.utils.logger import Logger
from pykg2vec.common import HyperparamterLoader


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
    _logger = Logger().get_logger(__name__)

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
        self.device = args.device

    def summary(self):
        """Function to print the summary."""
        summary = []
        summary.append("")
        summary.append("------------------Global Setting--------------------")
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
            summary.append("%s : %s"%(key, val))
        summary.append("---------------------------------------------------")
        summary.append("")
        self._logger.info("\n".join(summary))

    def summary_hyperparameter(self, model_name):
        """Function to print the hyperparameter summary."""
        summary_hyperparameter = []
        summary_hyperparameter.append("")
        summary_hyperparameter.append("-----------%s Hyperparameter Setting-------------"%(model_name))
        maxspace = len(max([k for k in self.hyperparameters.keys()])) + 15
        for key,val in self.hyperparameters.items():
            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            summary_hyperparameter.append("%s : %s" % (key, val))
        summary_hyperparameter.append("---------------------------------------------------")
        self._logger.info("\n".join(summary_hyperparameter))


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

    def __init__(self, args):
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
    def __init__(self, args):
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
      hidden_size (int): Defines the size of the latent dimension for entities and relations.
      batch_size (int): Defines the batch size for training the algorithm.
      epochs (int): Defines the total number of epochs for training the algorithm.
      data (str): Defines the knowledge base dataset to be used for training the algorithm.
      optimizer (str): Defines the optimization algorithm such as adam, sgd, adagrad, etc.
      sampling (str): Defines the sampling (bern or uniform) for corrupting the triples.
      bilinear (bool): If true uses bilnear transformation for loss else uses linear.
    
    """
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = 1
        self.margin = 1.0
        
        if args.exp is True:
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'sme')
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

        self.hyperparameters = {
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'sampling': self.sampling,
            'margin': self.margin,
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
    def __init__(self, args):
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
        self.margin = 1.0

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

    def __init__(self, args):
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
        self.alpha = args.alpha

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
            'sampling': self.sampling,
            'neg_rate': self.neg_rate,
            'alpha': self.alpha
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
    def __init__(self, args):
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

    def __init__(self, args):
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

    def __init__(self, args):
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
    def __init__(self, args):
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
    def __init__(self, args):

        self.feature_map_dropout = args.feature_map_dropout
        self.input_dropout = args.input_dropout
        self.hidden_dropout = args.hidden_dropout
        self.label_smoothing = args.label_smoothing
        self.learning_rate = args.learning_rate
        self.hidden_size = args.hidden_size
        self.hidden_size_1 = args.hidden_size_1
        
        self.batch_size = args.batch_training
        self.epochs = args.epochs
        self.data = args.dataset_name
        self.optimizer = args.optimizer
        self.sampling = args.sampling
        self.neg_rate = 0

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
            'hidden_size_1': self.hidden_size_1,
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
    def __init__(self, args):
        self.lmbda = 0.1
        self.input_dropout = 0.3
        self.hidden_dropout2 = 0.5
        self.hidden_dropout1 = 0.4
        self.label_smoothing = 0

        self.learning_rate = 0.0005
        self.rel_hidden_size = 200
        self.ent_hidden_size = 200
        self.batch_size = 128
        self.epochs = 500
        self.data = 'Freebase15k'
        self.optimizer = 'adam'
        self.sampling = "uniform"
        self.neg_rate = 0

        # else:
        #     self.lmbda = args.lmbda
        #     self.feature_map_dropout = args.feature_map_dropout
        #     self.input_dropout = args.input_dropout
        #     self.hidden_dropout1 = args.hidden_dropout
        #     self.hidden_dropout2 = args.hidden_dropout2
        #     self.use_bias = args.use_bias
        #     self.label_smoothing = args.label_smoothing
        #     self.lr_decay = args.lr_decay

        #     self.learning_rate = args.learning_rate
        #     self.L1_flag = args.l1_flag
        #     self.rel_hidden_size = args.rel_hidden_size
        #     self.ent_hidden_size = args.ent_hidden_size
        #     self.batch_size = args.batch_training
        #     self.epochs = args.epochs
        #     self.margin = args.margin
        #     self.data = args.dataset_name
        #     self.optimizer = args.optimizer
        #     self.sampling = args.sampling
        #     self.neg_rate = args.negrate

        self.hyperparameters = {
            'lmbda': self.lmbda,
            'input_dropout': self.input_dropout,
            'hidden_dropout1': self.hidden_dropout1,
            'hidden_dropout2': self.hidden_dropout2,
            'label_smoothing': self.label_smoothing,
            'learning_rate': self.learning_rate,
            'rel_hidden_size': self.rel_hidden_size,
            'ent_hidden_size': self.ent_hidden_size,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer': self.optimizer,
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
    def __init__(self, args):
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


class CPConfig(BasicConfig):
    """This class defines the configuration for the Canonical Tensor Decomposition Algorithm.

    CPConfig inherits the BasicConfig and defines the local arguements used in the
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

    def __init__(self, args):
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
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'cp')
            for key, value in paper_params.items():
                self.__dict__[key] = value  # copy all the setting from the paper.

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


class ANALOGYConfig(BasicConfig):
    """This class defines the configuration for the ANALOGY Algorithm.

    ANALOGYConfig inherits the BasicConfig and defines the local arguements used in the
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

    def __init__(self, args):
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
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'analogy')
            for key, value in paper_params.items():
                self.__dict__[key] = value  # copy all the setting from the paper.

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


class SimplEConfig(BasicConfig):
    """This class defines the configuration for the SimplE Algorithm.

    SimplEConfig inherits the BasicConfig and defines the local arguements used in the
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

    def __init__(self, args):
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
            paper_params = HyperparamterLoader().load_hyperparameter(args.dataset_name, 'simple')
            for key, value in paper_params.items():
                self.__dict__[key] = value  # copy all the setting from the paper.

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
