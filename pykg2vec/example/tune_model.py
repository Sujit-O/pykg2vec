# import sys
# sys.path.append("../")
# from config.hyperparams import KGETuneArgParser
from pykg2vec.config.hyperparams import KGETuneArgParser
# from utils.bayesian_optimizer import BaysOptimizer
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer


def main():
   	# getting the customized configurations from the command-line arguments.
    args = KGETuneArgParser().get_args()

    # initializing bayesian optimizer and prepare data.
    bays_opt = BaysOptimizer(args=args)

    # perform the golden hyperparameter tuning. 
    bays_opt.optimize()


if __name__ == "__main__":
    main()
