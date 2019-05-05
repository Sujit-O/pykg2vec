import tensorflow as tf
from argparse import ArgumentParser

import sys
sys.path.append("../")
from core.TransD import TransD
from config.config import TransDConfig
from utils.dataprep import DataPrep
from utils.trainer import Trainer


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with TransD')
    parser.add_argument('-b', '--batch', default=200, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=1000, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=60000, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=10, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')
    parser.add_argument('-ke', '--embed_ent', default=50, type=int, help='Hidden entity embedding size')
    parser.add_argument('-kl', '--embed_rel', default=50, type=int, help='Hidden relation embedding size')
    parser.add_argument('-db', '--debug', default=0, type=int, help='debug')

    args = parser.parse_args()

    data_handler = DataPrep(name_dataset=args.dataset, sampling="uniform", algo='TransD')
    data_handler.prepare_data()
    
    config = TransDConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs)
    
    config.set_dataset(args.dataset)

    config.ent_hidden_size = args.embed_ent
    config.rel_hidden_size = args.embed_rel

    config.test_step = args.test_step
    config.test_num  = args.test_num
    config.gpu_fraction = args.gpu_frac
    config.save_model = True
    config.L1_flag = False

    model = TransD(config)

    trainer = Trainer(model=model, debug=True if args.debug == 1 else False)
    trainer.build_model()
    trainer.train_model()

if __name__ == "__main__":
    tf.app.run()