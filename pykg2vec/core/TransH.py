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
TransH models a relation as a hyperplane together with a translation operation on it.
By doint this, it aims to preserve the mapping properties of relations such as reflexive,
one-to-many, many-to-one, and many-to-many with almost the same model complexity of TransE.

Portion of Code Based on https://github.com/thunlp/OpenKE/blob/master/models/TransH.py
 and https://github.com/thunlp/TensorFlow-TransX/blob/master/transH.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("D:\louis\Dropbox\louis_research\pyKG2Vec\pykg2vec")

from core.KGMeta import KGMeta
import tensorflow as tf
from config.config import TransHConfig
from utils.dataprep import DataPrep
import timeit
from argparse import ArgumentParser
from utils.visualization import Visualization
from utils.evaluation import EvaluationTransH
import os


class TransH(KGMeta):
    
    def __init__(self, config=None, data_handler=None):

        if not config:
            self.config = TransHConfig()
        else:
            self.config = config

        self.data_handler = data_handler
        self.loss_regularize = None

        self.pos_h = tf.placeholder(tf.int32, [None])
        self.pos_t = tf.placeholder(tf.int32, [None])
        self.pos_r = tf.placeholder(tf.int32, [None])

        self.neg_h = tf.placeholder(tf.int32, [None])
        self.neg_t = tf.placeholder(tf.int32, [None])
        self.neg_r = tf.placeholder(tf.int32, [None])

        # self.test_h = tf.placeholder(tf.int32, [1])
        # self.test_t = tf.placeholder(tf.int32, [1])
        # self.test_r = tf.placeholder(tf.int32, [1])

        self.loss = None
        self.op_train = None
        self.loss_every = None
        self.norm_entity = None
        self.head_rank = None
        self.tail_rank = None
        self.__variables = []

        self.norm_head_rank = None
        self.norm_tail_rank = None

        self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                              shape=[self.data_handler.tot_entity, self.config.hidden_size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        self.rel_embeddings = tf.get_variable(name="rel_embedding",
                                              shape=[self.data_handler.tot_relation, self.config.hidden_size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        
        self.__variables.append(self.ent_embeddings)
        self.__variables.append(self.rel_embeddings)

    def train_model(self):
        """function to define the model"""

        self.norm_entity = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        self.norm_relation = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
        norm_entity_l2sum = tf.sqrt(tf.reduce_sum(self.norm_entity ** 2, axis=1))

        emb_ph = tf.nn.embedding_lookup(self.norm_entity, self.pos_h)
        emb_pt = tf.nn.embedding_lookup(self.norm_entity, self.pos_t)
        emb_pr = tf.nn.embedding_lookup(self.norm_relation, self.pos_r)

        emb_nh = tf.nn.embedding_lookup(self.norm_entity, self.neg_h)
        emb_nt = tf.nn.embedding_lookup(self.norm_entity, self.neg_t)
        emb_nr = tf.nn.embedding_lookup(self.norm_relation, self.neg_r)
        
        # getting the required normal vectors of planes to transfer entity embedding 
        # pos_norm = tf.nn.embedding_lookup(''' projection matrix''', self.pos_r)
        # neg_norm = tf.nn.embedding_lookup(''' projection matrix''', self.neg_r)

        score_pos = tf.reduce_sum(tf.abs(emb_ph + emb_pr - emb_pt), axis=1)
        score_neg = tf.reduce_sum(tf.abs(emb_nh + emb_nr - emb_nt), axis=1)

        self.loss_every = tf.maximum(0., score_pos + self.config.margin - score_neg)
        self.loss = tf.reduce_sum(tf.maximum(0., score_pos + self.config.margin - score_neg))
        self.loss_regularize = tf.reduce_sum(tf.abs(self.rel_embeddings)) \
                               + tf.reduce_sum(tf.abs(self.ent_embeddings))

        if self.config.optimizer == 'gradient':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        else:
            raise NotImplementedError("No support for %s optimizer" % self.config.optimizer)

        grads = optimizer.compute_gradients(self.loss, self.variables)
        self.op_train = optimizer.apply_gradients(grads)
        return self.loss, self.op_train, self.loss_every, self.norm_entity

    def train(self):
        """function to train the model"""

        with tf.Session(config=self.config.gpu_config) as sess:
            evaluate = EvaluationTransH(self, 'test')
            loss, op_train, loss_every, norm_entity = self.train_model()
            sess.run(tf.global_variables_initializer())

            norm_rel = sess.run(tf.nn.l2_normalize(self.rel_embeddings, axis=1))
            sess.run(tf.assign(self.rel_embeddings, norm_rel))

            norm_ent = sess.run(tf.nn.l2_normalize(self.ent_embeddings, axis=1))
            sess.run(tf.assign(self.ent_embeddings, norm_ent))

            gen_train = self.data_handler.batch_generator_train(batch=self.config.batch_size)

            if self.config.loadFromData:
                saver = tf.train.Saver()
                saver.restore(sess, self.config.tmp + '/TransEModel.vec')

            for n_iter in range(self.config.epochs):
                acc_loss = 0
                batch = 0
                num_batch = len(self.data_handler.train_triples_ids) // self.config.batch_size
                start_time = timeit.default_timer()

                for i in range(num_batch):
                    ph, pt, pr, nh, nt, nr = list(next(gen_train))

                    feed_dict = {
                        self.pos_h: ph,
                        self.pos_t: pt,
                        self.pos_r: pr,
                        self.neg_h: nh,
                        self.neg_t: nt,
                        self.neg_r: nr
                    }

                    l_val, _, l_every, n_entity = sess.run([loss, op_train, loss_every, norm_entity],
                                                           feed_dict)

                    acc_loss += l_val
                    batch += 1
                    print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time,
                                                                batch,
                                                                num_batch,
                                                                l_val), end='\r')
                print('iter[%d] ---Train Loss: %.5f ---time: %.2f' % (
                    n_iter, acc_loss, timeit.default_timer() - start_time))

                if n_iter % self.config.test_step == 0 or n_iter == 0 or n_iter == self.config.epochs - 1:
                    evaluate.test(sess, n_iter)
                    evaluate.print_test_summary(n_iter)

            if self.config.save_model:
                self.save_model(sess)
            if self.config.disp_summary:
                self.summary()
            if self.config.disp_result:
                triples = self.data_handler.validation_triples_ids[:self.config.disp_triple_num]
                self.display(triples, sess)

    def test(self):
        head_vec, rel_vec, tail_vec = self.embed(self.test_h, self.test_r, self.test_t)

        norm_embedding_entity = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_embedding_relation = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        norm_head_vec = tf.nn.embedding_lookup(norm_embedding_entity, self.test_h)
        norm_rel_vec = tf.nn.embedding_lookup(norm_embedding_relation, self.test_r)
        norm_tail_vec = tf.nn.embedding_lookup(norm_embedding_entity, self.test_t)

        _, self.head_rank = tf.nn.top_k(tf.reduce_sum(tf.abs(self.ent_embeddings
                                                             + rel_vec - tail_vec),
                                                      axis=1),
                                        k=self.data_handler.tot_entity)
        _, self.tail_rank = tf.nn.top_k(tf.reduce_sum(tf.abs(head_vec
                                                             + rel_vec - self.ent_embeddings),
                                                      axis=1),
                                        k=self.data_handler.tot_entity)

        _, self.norm_head_rank = tf.nn.top_k(
            tf.reduce_sum(tf.abs(norm_embedding_entity + norm_rel_vec - norm_tail_vec),
                          axis=1), k=self.data_handler.tot_entity)
        _, self.norm_tail_rank = tf.nn.top_k(
            tf.reduce_sum(tf.abs(norm_head_vec + norm_rel_vec - norm_embedding_entity),
                          axis=1), k=self.data_handler.tot_entity)
        return self.head_rank, self.tail_rank, self.norm_head_rank, self.norm_tail_rank

    def embed(self, h, r, t):
        """function to get the embedding value"""
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
        return emb_h, emb_r, emb_t

    def predict_embed(self, h, r, t, sess=None):
        """function to get the embedding value in numpy"""
        if not sess:
            raise NotImplementedError('No session found for predicting embedding!')
        emb_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        emb_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        emb_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
        h, r, t = sess.run([emb_h, emb_r, emb_t])
        return h, r, t

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
    @property
    def variables(self):
        return self.__variables

def main(_):
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

    config = TransHConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          test_step=args.test_step,
                          test_num=args.test_num,
                          gpu_fraction=args.gpu_frac)

    model = TransH(config=config, data_handler=data_handler)
    model.summary()
    model.train()


if __name__ == "__main__":
    tf.app.run()
