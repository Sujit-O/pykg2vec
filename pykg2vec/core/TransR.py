#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
------------------Paper Title-----------------------------
Knowledge Graph Embedding by Translating on Hyperplanes
------------------Paper Authors---------------------------
Zhen Wang1,Jianwen Zhang2, Jianlin Feng1, Zheng Chen2
1Department of Information Science and Technology, Sun Yat-sen University, Guangzhou, China
2Microsoft Research, Beijing, China
1{wangzh56@mail2, fengjlin@mail}.sysu.edu.cn
2{jiazhan, zhengc}@microsoft.com
------------------Summary---------------------------------
TransH  models a relation as a hyperplane together with a translation operation on it.
By doint this, it aims to preserve the mapping properties of relations such as reflexive,
one-to-many, many-to-one, and many-to-many with almost the same model complexity of TransE.

Portion of Code Based on https://github.com/thunlp/OpenKE/blob/master/models/TransE.py
 and https://github.com/wencolani/TransE.git
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("D:\dev\pykg2vec\pykg2vec")
from core.KGMeta import KGMeta
from utils.visualization import Visualization
from utils.evaluation import EvaluationTransE
from utils.evaluation import EvaluationTransE
from config.config import TransEConfig
from utils.dataprep import DataPrep

import tensorflow as tf
import timeit
from argparse import ArgumentParser
import os


class TransR(KGMeta):
    @property
    def variables(self):
        return self.__variables

    def __init__(self, config=None, data_handler=None):

        if not config:
            self.config = TransEConfig()
        else:
            self.config = config

        self.data_handler = data_handler
        pass

    def train_model(self):
        """function to train the model"""
        pass

    def train(self):
        pass

    def test(self):
        pass

    def embed(self, h, r, t):
        """function to get the embedding value"""
        pass

    def predict_embed(self, h, r, t, sess=None):
        """function to get the embedding value in numpy"""
        if not sess:
            raise NotImplementedError('No session found for predicting embedding!')
        pass

    def display(self, triples=None, sess=None):
        """function to display embedding"""
        viz = Visualization(triples=triples,
                            idx2entity=self.data_handler.idx2entity,
                            idx2relation=self.data_handler.idx2relation)

        viz.get_idx_n_emb(model=self, sess=sess)
        viz.reduce_dim()
        viz.draw_figure()

    def save_model(self, sess):
        """function to save the model"""
        if not os.path.exists(self.config.tmp):
            os.mkdir('../intermediate')
        saver = tf.train.Saver()
        saver.save(sess, '../intermediate/TransEModel.vec')

    def load_model(self, sess):
        """function to load the model"""
        saver = tf.train.Saver()
        saver.restore(sess, self.config.tmp + '/TransEModel.vec')

    def summary(self):
        """function to print the summary"""
        print("\n----------SUMMARY----------")
        # Acquire the max length and add four more spaces
        maxspace = len(max([k for k in self.config.__dict__.keys()])) + 4
        for key, val in self.config.__dict__.items():
            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            print(key, ":", val)
        print("---------------------------")
    # TODO: Save summary
    # with open('../intermediate/TransEModel_summary.json', 'wb') as fp:
    # 	json.dump(self.config.__dict__, fp)


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with TransE')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-l', '--epochs', default=10, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=5, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=5, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')

    args = parser.parse_args()

    config = TransEConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          test_step=args.test_step,
                          test_num=args.test_num,
                          gpu_fraction=args.gpu_frac)

    model = TransR(config=config)
    model.summary()
    model.train()


if __name__ == "__main__":
    tf.app.run()
