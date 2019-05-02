import tensorflow as tf
from argparse import ArgumentParser

import sys

sys.path.append("../")
from core.DistMult2 import DistMult2
from config.config import DistMultConfig
from utils.dataprep import DataPrep
from utils.trainer import Trainer


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with distmult2')
    parser.add_argument('-b', '--batch', default=200, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=100, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=1000, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=2, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.8, type=float, help='GPU fraction to use')
    parser.add_argument('-k', '--embed', default=100, type=int, help='Hidden embedding size')
    args = parser.parse_args()

    data_handler = DataPrep(name_dataset=args.dataset, sampling="uniform", algo='distmult2')

    config = DistMultConfig(learning_rate=args.learn_rate,
                         batch_size=args.batch,
                         epochs=args.epochs,
                         hidden_size=args.embed)

    config.test_step = args.test_step
    config.test_num = args.test_num
    config.gpu_fraction = args.gpu_frac
    # config.plot_entity_only = True
    config.save_model = True
    config.optimizer = "adagrad"
    
    model = DistMult2(config)

    trainer = Trainer(model=model)
    trainer.build_model()
    trainer.train_model()


if __name__ == "__main__":
    tf.app.run()
