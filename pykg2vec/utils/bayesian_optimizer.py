#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for performing bayesian optimization on algorithms
"""
import importlib

from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
import pandas as pd


from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.logger import Logger
from pykg2vec.config import KGEArgParser

model_path = "pykg2vec.models"
config_path = "pykg2vec.config"
hyper_param_path = "pykg2vec.hyperparams"

moduleMap = {"analogy": "pointwise",
             "complex": "pointwise",
             "complexn3": "pointwise",
             "conve": "projection",
             "cp": "pointwise",
             "hole": "pairwise",
             "distmult": "pointwise",
             "kg2e": "pairwise",
             "kg2e_el": "pairwise",
             "ntn": "pairwise",
             "proje_pointwise": "projection",
             "rescal": "pairwise",
             "rotate": "pairwise",
             "simple": "pointwise",
             "simple_ignr": "pointwise",
             "slm": "pairwise",
             "sme": "pairwise",
             "sme_bl": "pairwise",
             "transd": "pairwise",
             "transe": "pairwise",
             "transg": "pairwise",
             "transh": "pairwise",
             "transm": "pairwise",
             "transr": "pairwise",
             "tucker": "projection"}


modelMap = {"analogy": "ANALOGY",
            "complex": "Complex",
            "complexn3": "ComplexN3",
            "conve": "ConvE",
            "cp": "CP",
            "hole": "HoLE",
            "distmult": "DistMult",
            "kg2e": "KG2E",
            "kg2e_el": "KG2E_EL",
            "ntn": "NTN",
            "proje_pointwise": "ProjE_pointwise",
            "rescal": "Rescal",
            "rotate": "RotatE",
            "simple": "SimplE",
            "simple_ignr": "SimplE_ignr",
            "slm": "SLM",
            "sme": "SME",
            "sme_bl": "SME_BL",
            "transd": "TransD",
            "transe": "TransE",
            "transg": "TransG",
            "transh": "TransH",
            "transm": "TransM",
            "transr": "TransR",
            "tucker": "TuckER"}

configMap = {"analogy": "ANALOGYConfig",
             "complex": "ComplexConfig",
             "complexn3": "ComplexConfig",
             "conve": "ConvEConfig",
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

hypMap = {"analogy": "ANALOGYParams",
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


class BaysOptimizer(object):
    """Bayesian optimizer class for tuning hyperparameter.

      This class implements the Bayesian Optimizer for tuning the 
      hyper-parameter.

      Args:
        args (object): The Argument Parser object providing arguments.
        name_dataset (str): The name of the dataset.
        sampling (str): sampling to be used for generating negative triples


      Examples:
        >>> from pykg2vec.hyperparams import KGETuneArgParser
        >>> from pykg2vec.utils.bayesian_optimizer import BaysOptimizer
        >>> model = Complex()
        >>> args = KGETuneArgParser().get_args(sys.argv[1:])
        >>> bays_opt = BaysOptimizer(args=args)
        >>> bays_opt.optimize()
    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, args=None):
        """store the information of database"""
        if args.model.lower() in ["tucker", "tucker_v2", "conve", "convkb", "proje_pointwise"]:
          raise Exception("Model %s has not been supported in tuning hyperparameters!" % args.model)

        self.model_name = args.model
        self.knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, custom_dataset_path=args.dataset_path)
        self.kge_args = KGEArgParser().get_args([])
        self.kge_args.dataset_name = args.dataset_name
        self.kge_args.debug = args.debug
        self.max_evals = args.max_number_trials if not args.debug else 3
        try:
            self.model_obj = getattr(importlib.import_module(model_path + ".%s" % moduleMap[self.model_name.lower()]),
                                     modelMap[self.model_name.lower()])
            self.config_obj = getattr(importlib.import_module(config_path), configMap[self.model_name.lower()])
            self.config_local = self.config_obj(self.kge_args)
            hyper_params = getattr(importlib.import_module(hyper_param_path), hypMap[self.model_name.lower()])()
            self.search_space = hyper_params.search_space
        except ModuleNotFoundError:
            self._logger.error("%s not implemented! Select from: %s" % \
                               (self.model_name.lower(), ' '.join(map(str, modelMap.values()))))

    def optimize(self):
        """Function that performs bayesian optimization"""
        trials = Trials()

        self._best_result = fmin(fn=self._get_loss, space=self.search_space, trials=trials,
                                 algo=tpe.suggest, max_evals=self.max_evals)
        
        columns = list(self.search_space.keys())   
        results = pd.DataFrame(columns=['iteration'] + columns + ['loss'])
        
        for idx, trial in enumerate(trials.trials):
            row = [idx]
            translated_eval = space_eval(self.search_space, {k: v[0] for k, v in trial['misc']['vals'].items()})
            for k in columns:
                row.append(translated_eval[k])
            row.append(trial['result']['loss'])
            results.loc[idx] = row

        path = self.config_local.path_result / self.model_name
        path.mkdir(parents=True, exist_ok=True)
        results.to_csv(str(path / "trials.csv"), index=False)
        
        self._logger.info(results)
        self._logger.info('Found golden setting:')
        self._logger.info(space_eval(self.search_space, self._best_result))

    def return_best(self):
        """Function to return the best hyper-parameters"""
        assert hasattr(self, '_best_result') is True, 'Cannot find golden setting. Has optimize() been called?'
        return space_eval(self.search_space, self._best_result)

    def _get_loss(self, params):
        """Function that defines and acquires the loss"""
        
        # copy the hyperparameters to trainer config and hyperparameter set. 
        for key, value in params.items():
          self.config_local.__dict__[key] = value
          self.config_local.hyperparameters[key] = value

        model = self.model_obj(self.config_local)

        self.trainer = Trainer(model)

        # configure common setting for a tuning training. 
        self.config_local.disp_result = False
        self.config_local.disp_summary = False
        self.config_local.save_model = False

        # do not overwrite test numbers if set
        if self.config_local.test_num is None:
            self.config_local.test_num = 1000

        if self.kge_args.debug:
          self.config_local.epochs = 1
          self.config_local.hyperparameters['epochs'] = 1

        # start the trial.
        self.trainer.build_model()
        loss = self.trainer.tune_model()

        return {'loss': loss, 'status': STATUS_OK}
