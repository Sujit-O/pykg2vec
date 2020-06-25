'''
==================================================
Automatic Hyperparameter Discovery (tune_model.py)
==================================================
With tune_model.py we can train the existed model using command:

- check all tunnable parameters. ::

    $ python tune_model.py -h 

- We are still improving the interfaces to make them more convenient to use. For now, please refer to hyperparams.py_ to manually adjust the search space of hyperparameters. ::

    # in hyperparams.py#xxxParams
    self.search_space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
        'L1_flag': hp.choice('L1_flag', [True, False]),
        'hidden_size': scope.int(hp.qloguniform('hidden_size', np.log(8), np.log(256),1)),
        'batch_size': scope.int(hp.qloguniform('batch_size', np.log(8), np.log(4096),1)),
        'margin': hp.uniform('margin', 0.0, 2.0),
        'optimizer': hp.choice('optimizer', ["adam", "sgd", 'rms']),
        'epochs': hp.choice('epochs', [10])
    }

- Tune TransE using the benchmark dataset fb15k. ::

    $ python tune_model.py -mn TransE -ds fb15k


.. _hyperparams.py: https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/config/hyperparams.py

====

We also attached the source code of tune_model.py below for your reference.


'''
# Author: Sujit Rokka Chhetri
# License: MIT

import sys


from pykg2vec.hyperparams import KGETuneArgParser
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer


def main():
   	# getting the customized configurations from the command-line arguments.
    args = KGETuneArgParser().get_args(sys.argv[1:])

    # initializing bayesian optimizer and prepare data.
    bays_opt = BaysOptimizer(args=args)

    # perform the golden hyperparameter tuning. 
    bays_opt.optimize()


if __name__ == "__main__":
    main()
