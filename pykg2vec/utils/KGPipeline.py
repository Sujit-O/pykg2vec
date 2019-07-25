#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for training, tuning and evaluation.
"""
import tensorflow as tf
import timeit
import sys

import numpy as np

from pykg2vec.utils.evaluation import Evaluation
from pykg2vec.utils.visualization import Visualization
from pykg2vec.utils.generator import Generator
from pykg2vec.config.config import Importer, KGEArgParser
from pykg2vec.config.global_config import GeneratorConfig
from pykg2vec.utils.kgcontroller import KGMetaData, KnowledgeGraph
from pykg2vec.config.hyperparams import KGETuneArgParser
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer
from pykg2vec.utils.trainer import Trainer



class KGPipeline:
    """Class for handling the entire pipeline for KG embedding.

        Args:
            model (str): Name of the model
            dataset (str): Dataset name
            debug (bool): Run in debug mode for faster simulation

        Examples:
            >>> from pykg2vec.utils.KGPipeline import KGPipeline 
            >>> kg_pipeline = KGPipeline(model="transe", dataset ="Freebase15k")
            >>> kg_pipeline.tune()
            >>> kg_pipeline.plot_tune_result()
            >>> kg_pipeline.test()
            >>> kg_pipeline.plot_test_result()
    """

    def __init__(self, model="TransE", dataset="Freebase15k", debug=False):
        self.model = model
        self.dataset= dataset
        self.debug = debug
        self.best = None
    
    def tune(self): 
        """Fuction to tune the hyper-parameters for the model 
        using training and validation set."""   
        
        # getting the customized configurations from the command-line arguments.
        args = KGETuneArgParser().get_args([])
        args.model = self.model
        args.dataset_name = self.dataset
        args.debug = self.debug
        # initializing bayesian optimizer and prepare data.
        bays_opt = BaysOptimizer(args=args)

        # perform the golden hyperparameter tuning. 
        bays_opt.optimize()

        self.best = bays_opt.return_best()

    def plot_tune_result(self):    
        """Function to plot the tuning result."""
        pass

    def test(self):
        """Function to evaluate final model on testing set while training the model using 
        best hyper-paramters on merged training and validation set."""
        args = KGEArgParser().get_args([])
        args.model = self.model
        args.dataset_name = self.dataset
        args.debug = self.debug
        # Preparing data and cache the data for later usage
        knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, negative_sample=args.sampling)
        knowledge_graph.prepare_data()

        # Extracting the corresponding model config and definition from Importer().
        config_def, model_def = Importer().import_model_config(args.model_name.lower())
        config = config_def(args=args)
        
        #Update the config params with the golden hyperparameter
        for k,v in self.best.items():
            config.__dict__[k]=v
        model = model_def(config)

        # Create, Compile and Train the model. While training, several evaluation will be performed.
        trainer = Trainer(model=model,trainon='train_and_valid', teston='test',debug=args.debug)
        trainer.build_model()
        trainer.train_model()

    def plot_test_result(self):    
        """Function to plot the final test result."""
        pass    

        
if __name__ == "__main__":
    kg_pipeline = KGPipeline(model="transe", dataset ="Freebase15k", debug=True)
    kg_pipeline.tune()
    # kg_pipeline.plot_tune_result()
    kg_pipeline.test()
    # kg_pipeline.plot_test_result()


        