import tensorflow as tf
from argparse import ArgumentParser
import importlib
import sys

sys.path.append("../")
model_path = "core"
config_path = "config.config"

from utils.bayesian_optimizer import BaysOptimizer


def main():
    parser = ArgumentParser(description='Bayesian HyperParameter Optimizer')
    parser.add_argument('-m', '--model', default='TransE', type=str, help='Model to tune')
    args = parser.parse_args()

    bays_opt = BaysOptimizer(args=args)

    bays_opt.optimize()
    
if __name__ == "__main__":
    main()
