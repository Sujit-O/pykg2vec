'''
=======================
Tuning an Algorithm
=======================
In this example, we will domonstrate how to 
use the pykg3vec algorithm to tune a single algorithm.
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
