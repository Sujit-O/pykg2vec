import tensorflow as tf
from argparse import ArgumentParser

import sys
sys.path.append("../")
from core.KG2E import KG2E
from config.config import KG2EConfig
from utils.dataprep import DataPrep
from utils.trainer import Trainer


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with KG2E')
    parser.add_argument('-b', '--batch', default=1440, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=500, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=1000, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=10, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.8, type=float, help='GPU fraction to use')
    parser.add_argument('-k', '--embed', default=50, type=int, help='Hidden embedding size')
    args = parser.parse_args()

    data_handler = DataPrep(name_dataset=args.dataset, sampling="uniform", algo='KG2E')
    # args.test_num = min(len(data_handler.test_triples_ids), args.test_num)
    
    config = KG2EConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          hidden_size=args.embed)

    config.test_step = args.test_step
    config.test_num  = args.test_num
    config.gpu_fraction = args.gpu_frac
    config.save_model = True
    config.margin = 1.0

    model = KG2E(config, data_handler)
    
    trainer = Trainer(model=model)
    trainer.build_model()
    trainer.train_model()

if __name__ == "__main__":
    tf.app.run()