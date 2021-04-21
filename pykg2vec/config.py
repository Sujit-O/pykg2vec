"""
config.py
==========

This module consists of definition of the necessary configuration parameters for all the
core algorithms. The parameters are seprated into global parameters which are common
across all the algorithms, and local parameters which are specific to the algorithms.

"""


from pykg2vec.data.kgcontroller import KnowledgeGraph, KGMetaData
from pykg2vec.utils.logger import Logger
from pykg2vec.common import HyperparameterLoader


class Config:
    """ The class defines the basic configuration for the pykg2vec.

        Config consists of the necessary parameter description used by all the
        modules including the algorithms and utility functions.

        Args:
            test_step (int): Testing is carried out every test_step.
            test_num (int): Number of triples that will be tested during evaluation.
            triple_num (int): Number of triples that will be used for plotting the embedding.
            tmp (Path Object): Path where temporary model information is stored.
            result (Path Object): Gives the path where the result will be saved.
            figures (Path Object): Gives the path where the figures will be saved.
            load_from_data (string): If set, loads the model parameters if available from disk.
            save_model (True): If True, store the trained model parameters.
            disp_summary (bool): If True, display the summary before and after training the algorithm.
            disp_result (bool): If True, displays result while training.
            plot_embedding (bool): If True, will plot the embedding after performing t-SNE based dimensionality reduction.
            log_training_placement (bool): If True, allows us to find out which devices the operations and tensors are assigned to.
            plot_training_result (bool): If True, plots the loss values stored during training.
            plot_testing_result (bool): If True, it will plot all the testing result such as mean rank, hit ratio, etc.
            plot_entity_only (bool): If True, plots the t-SNE reduced embdding of the entities in a figure.
            hits (List): Gives the list of integer for calculating hits.
            knowledge_graph (Object): It prepares and holds the instance of the knowledge graph dataset.
            kg_meta (object): Stores the statistics metadata of the knowledge graph.

    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, args):
        for arg_name in vars(args):
            self.__dict__[arg_name] = getattr(args, arg_name)
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name

        # Training and evaluating related variables
        self.hits = [1, 3, 5, 10]
        self.disp_result = False
        self.patience = 3 # should make this configurable as well.

        # Visualization related,
        # p.s. the visualizer is disable for most of the KGE methods for now.
        self.disp_triple_num = 20
        self.plot_training_result = True
        self.plot_testing_result = True

        # Knowledge Graph Information
        self.knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, custom_dataset_path=args.dataset_path)
        for key in self.knowledge_graph.kg_meta.__dict__:
            self.__dict__[key] = self.knowledge_graph.kg_meta.__dict__[key]

        # The results of training will be stored in the following folders
        # which are relative to the parent folder (the path of the dataset).
        dataset_path = self.knowledge_graph.dataset.dataset_path
        self.path_tmp = dataset_path / 'intermediate'
        self.path_tmp.mkdir(parents=True, exist_ok=True)
        self.path_result = dataset_path / 'results'
        self.path_result.mkdir(parents=True, exist_ok=True)
        self.path_figures = dataset_path / 'figures'
        self.path_figures.mkdir(parents=True, exist_ok=True)
        self.path_embeddings = dataset_path / 'embeddings'
        self.path_embeddings.mkdir(parents=True, exist_ok=True)

        if args.exp is True:
            paper_params = HyperparameterLoader(args).load_hyperparameter(args.dataset_name, args.model_name)
            for key, value in paper_params.items():
                self.__dict__[key] = value # copy all the setting from the paper.

    def summary(self):
        """Function to print the summary."""
        summary = ["", "------------------Global Setting--------------------"]
        # Acquire the max length and add four more spaces
        maxspace = len(max(self.__dict__.keys())) + 20
        for key, val in self.__dict__.items():
            if isinstance(val, (KGMetaData, KnowledgeGraph)):
                continue

            if len(key) < maxspace:
                for _ in range(maxspace - len(key)):
                    key = ' ' + key
            summary.append("%s : %s"%(key, val))
        summary.append("---------------------------------------------------")
        summary.append("")
        self._logger.info("\n".join(summary))
