'''
=================================================
Automatic hyperparameter discovery
=================================================
In this example, we will domonstrate how to 
use the pykg2vec to tune a single algorithm.

With tune_model.py we then can train the existed model using command: ::

    # check all tunnable parameters.
    $ python tune_model.py -h 

    # Tune [TransE model] using the [benchmark dataset].
    $ python tune_model.py -mn [TransE] -ds [dataset name] 

We are still working on more convenient interfaces.
Right now, please have a look over [hyperparams.py](https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/config/hyperparams.py) to adjust the ranges to be searched through other than the default ranges.
Besides, you can tune the hyperparameter on your own dataset as well by following the previous instructions.

'''
# Author: Sujit Rokka Chhetri
# License: MIT

import sys


from pykg2vec.config.hyperparams import KGETuneArgParser
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
