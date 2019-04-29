import tensorflow as tf
from argparse import ArgumentParser

import sys
sys.path.append("../")
from core.ProjE_pointwise import ProjE_pointwise
from config.config import ProjE_pointwiseConfig
from utils.dataprep import DataPrep
from utils.trainer import Trainer


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with ProjE_pointwise')
    parser.add_argument('-b', '--batch', default=200, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=100, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=60000, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=1, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')
    parser.add_argument('-k', '--embed', default=200, type=int, help='Hidden embedding size')
    args = parser.parse_args()

    data_handler = DataPrep(name_dataset=args.dataset, sampling="uniform", algo='proje')
    # args.test_num = min(len(data_handler.test_triples_ids), args.test_num)
    
    config = ProjE_pointwiseConfig(learning_rate=args.learn_rate,
                       batch_size=args.batch,
                       epochs=args.epochs)
    
    config.hidden_size = args.embed
    config.test_step = args.test_step
    config.test_num  = args.test_num
    config.gpu_fraction = args.gpu_frac
    config.plot_entity_only = True
    config.save_model = True
    config.sampling = "proje"

    model = ProjE_pointwise(config)
    
    trainer = Trainer(model=model)
    trainer.build_model()
    trainer.train_model()

if __name__ == "__main__":
    tf.app.run()