#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for performing bayesian optimization on algorithms
"""
from hyperopt import fmin, tpe, Trials, STATUS_OK, space_eval
import pandas as pd

from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.utils.trainer import Trainer
from pykg2vec.utils.logger import Logger
from pykg2vec.common import Importer, HyperparameterLoader


class BaysOptimizer:
    """Bayesian optimizer class for tuning hyperparameter.

      This class implements the Bayesian Optimizer for tuning the
      hyper-parameter.

      Args:
        args (object): The Argument Parser object providing arguments.
        name_dataset (str): The name of the dataset.
        sampling (str): sampling to be used for generating negative triples


      Examples:
        >>> from pykg2vec.common import KGEArgParser
        >>> from pykg2vec.utils.bayesian_optimizer import BaysOptimizer
        >>> model = Complex()
        >>> args = KGEArgParser().get_args(sys.argv[1:])
        >>> bays_opt = BaysOptimizer(args=args)
        >>> bays_opt.optimize()
    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, args):
        """store the information of database"""
        if args.model_name.lower() in ["conve", "convkb", "proje_pointwise", "interacte", "hyper", "acre"]:
            raise Exception("Model %s has not been supported in tuning hyperparameters!" % args.model)

        self.model_name = args.model_name
        self.knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, custom_dataset_path=args.dataset_path)
        self.kge_args = args
        self.max_evals = args.max_number_trials if not args.debug else 3

        self.config_obj, self.model_obj = Importer().import_model_config(self.model_name.lower())
        self.config_local = self.config_obj(self.kge_args)
        self.search_space = HyperparameterLoader(args).load_search_space(self.model_name.lower())
        self._best_result = None
        self.trainer = None

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
        assert self._best_result is not None, 'Cannot find golden setting. Has optimize() been called?'
        return space_eval(self.search_space, self._best_result)

    def _get_loss(self, params):
        """Function that defines and acquires the loss"""

        # copy the hyperparameters to trainer config and hyperparameter set.
        for key, value in params.items():
            self.config_local.__dict__[key] = value
        self.config_local.__dict__['device'] = self.kge_args.device
        model = self.model_obj(**self.config_local.__dict__)

        self.trainer = Trainer(model, self.config_local)

        # configure common setting for a tuning training.
        self.config_local.disp_result = False
        self.config_local.disp_summary = False
        self.config_local.save_model = False

        # do not overwrite test numbers if set
        if self.config_local.test_num is None:
            self.config_local.test_num = 1000

        if self.kge_args.debug:
            self.config_local.epochs = 1

        # start the trial.
        self.trainer.build_model()
        loss = self.trainer.tune_model()

        return {'loss': loss, 'status': STATUS_OK}
