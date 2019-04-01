#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example code for using pykg2vec
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import sys
sys.path.append("D:\dev\pykg2vec\pykg2vec")
from config.config import TransEConfig
from utils.dataprep import DataPrep
from argparse import ArgumentParser
from core.TransE import TransE
import os


# import tensorflow as tf
# from pykg2vec.config.config import TransEConfig
# from pykg2vec.utils.dataprep import DataPrep
# from argparse import ArgumentParser
# from pykg2vec.core.TransE import TransE
# import os

def main():
    parser = ArgumentParser(description='Knowledge Graph Embedding with TransE')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='./intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=10, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=5, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=5, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')

    args = parser.parse_args()

    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    data_handler = DataPrep(args.dataset)

    config = TransEConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          test_step=args.test_step,
                          test_num=args.test_num,
                          gpu_fraction=args.gpu_frac)

    model = TransE(config=config,data_handler=data_handler)
    model.summary()
    model.train()


if __name__ == "__main__":
    main()

