#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
------------------Paper Title-----------------------------
Learning Entity and Relation Embeddings for Knowledge Graph Completion
------------------Paper Authors---------------------------
Yankai Lin1, Zhiyuan Liu1∗, Maosong Sun 1,2, Yang Liu3, Xuan Zhu 3
1 Department of Computer Science and Technology, State Key Lab on Intelligent Technology and Systems,
National Lab for Information Science and Technology, Tsinghua University, Beijing, China
2 Jiangsu Collaborative Innovation Center for Language Competence, Jiangsu, China
3 Samsung R&D Institute of China, Beijing, China
------------------Summary---------------------------------
TranR is a translation based knowledge graph embedding method. Similar to TransE and TransH, it also
builds entity and relation embeddings by regarding a relation as translation from head entity to tail
entity. However, compared to them, it builds the entity and relation embeddings in a separate entity
and relation spaces.

Portion of Code Based on https://github.com/thunlp/TensorFlow-TransX/blob/master/transR.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("D:\dev\pykg2vec\pykg2vec")
from core.KGMeta import KGMeta
from utils.visualization import Visualization
from config.config import TransRConfig
from utils.dataprep import DataPrep

import tensorflow as tf
import timeit
from argparse import ArgumentParser
import os
import numpy as np
from sklearn.manifold import TSNE


class TransR(KGMeta):
    @property
    def variables(self):
        return self.__variables

    def __init__(self, config=None, data_handler=None, load_entity=None, load_rel=None):

        if config is None:
            self.config = TransRConfig()
        else:
            self.config = config

        self.data_handler = data_handler
        self.model_name = 'TransR'
        with tf.name_scope("read_inputs"):
            self.pos_h = tf.placeholder(tf.int32, [self.config.batch_size])
            self.pos_t = tf.placeholder(tf.int32, [self.config.batch_size])
            self.pos_r = tf.placeholder(tf.int32, [self.config.batch_size])
            self.neg_h = tf.placeholder(tf.int32, [self.config.batch_size])
            self.neg_t = tf.placeholder(tf.int32, [self.config.batch_size])
            self.neg_r = tf.placeholder(tf.int32, [self.config.batch_size])

        with tf.name_scope("embedding"):
            if load_entity is not None:
                self.ent_embeddings = tf.Variable(np.loadtxt(load_entity),
                                                  name="ent_embedding",
                                                  dtype=np.float32)
            else:
                self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                                      shape=[self.data_handler.tot_entity,
                                                             self.config.ent_hidden_size],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            if load_rel is not None:
                self.rel_embeddings = tf.Variable(np.loadtxt(load_rel),
                                                  name="rel_embedding",
                                                  dtype=np.float32)
            else:
                self.rel_embeddings = tf.get_variable(name="rel_embedding",
                                                      shape=[self.data_handler.tot_relation,
                                                             self.config.rel_hidden_size],
                                                      initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            rel_matrix = np.zeros([self.data_handler.tot_relation,
                                   self.config.rel_hidden_size * self.config.ent_hidden_size],
                                  dtype=np.float32)
            for i in range(self.data_handler.tot_relation):
                for j in range(self.config.rel_hidden_size):
                    for k in range(self.config.ent_hidden_size):
                        if j == k:
                            rel_matrix[i][j * self.config.ent_hidden_size + k] = 1.0
            self.rel_matrix = tf.Variable(rel_matrix, name="rel_matrix")

        with tf.name_scope('lookup_embeddings'):
            pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                        self.pos_h),
                                 [-1, self.config.ent_hidden_size, 1])
            pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                        self.pos_t),
                                 [-1, self.config.ent_hidden_size, 1])
            pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings,
                                                        self.pos_r),
                                 [-1, self.config.rel_hidden_size])
            neg_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                        self.neg_h),
                                 [-1, self.config.ent_hidden_size, 1])
            neg_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                        self.neg_t),
                                 [-1, self.config.ent_hidden_size, 1])
            neg_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings,
                                                        self.neg_r),
                                 [-1, self.config.rel_hidden_size])
            pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix,
                                                           self.pos_r),
                                    [-1, self.config.rel_hidden_size, self.config.ent_hidden_size])
            neg_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix,
                                                           self.neg_r),
                                    [-1, self.config.rel_hidden_size, self.config.ent_hidden_size])

            pos_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_h_e),
                                                    [-1, self.config.rel_hidden_size]), 1)
            pos_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_t_e),
                                                    [-1, self.config.rel_hidden_size]), 1)
            neg_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_h_e),
                                                    [-1, self.config.rel_hidden_size]), 1)
            neg_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(neg_matrix, neg_t_e),
                                                    [-1, self.config.rel_hidden_size]), 1)

        if self.config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
            self.predict = pos

        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.config.margin, 0))

    def train(self):
        with tf.Session(config=self.config.gpu_config) as sess:
            gen_train = self.data_handler.batch_generator_train(batch=self.config.batch_size)
            # if self.config.loadFromData:
            #     saver = tf.train.Saver()
            #     saver.restore(sess, self.config.tmp + '/TransRModel.vec')
            global_step = tf.Variable(0, name="global_step", trainable=False)

            if self.config.optimizer == 'gradient':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'rms':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            else:
                raise NotImplementedError("No support for %s optimizer" % self.config.optimizer)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            sess.run(tf.global_variables_initializer())

            for n_iter in range(self.config.epochs):
                acc_loss = 0
                batch = 0
                num_batch = len(self.data_handler.train_triples_ids) // self.config.batch_size
                start_time = timeit.default_timer()

                for i in range(num_batch):
                    ph, pr, pt, nh, nr, nt = list(next(gen_train))
                    feed_dict = {
                        self.pos_h: ph,
                        self.pos_t: pt,
                        self.pos_r: pr,
                        self.neg_h: nh,
                        self.neg_t: nt,
                        self.neg_r: nr
                    }

                    _, step, loss = sess.run(
                        [train_op, global_step, self.loss], feed_dict)

                    acc_loss += loss
                    batch += 1
                    print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time,
                                                                batch,
                                                                num_batch,
                                                                loss), end='\r')
                print('Epoch[%d] ---Train Loss: %.5f ---time: %.2f' % (
                    n_iter, acc_loss, timeit.default_timer() - start_time))

                # if n_iter % self.config.test_step == 0 or n_iter == 0 or n_iter == self.config.epochs - 1:
                #     evaluate.test(sess, n_iter)
                #     evaluate.print_test_summary(n_iter)

            if self.config.save_model:
                self.save_model(sess)

    def test(self):
        pass

    def embed(self, h, r, t):
        """function to get the embedding value"""
        pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                    h),
                             [-1, self.config.ent_hidden_size, 1])
        pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings,
                                                    r),
                             [-1, self.config.rel_hidden_size])
        pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                    t),
                             [-1, self.config.ent_hidden_size, 1])
        return pos_h_e, pos_r_e, pos_t_e

    def predict_embed(self, h, r, t, sess=None):
        """function to get the embedding value in numpy"""
        if not sess:
            raise NotImplementedError('No session found for predicting embedding!')
        pos_h_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                    h),
                             [-1, self.config.ent_hidden_size, 1])
        pos_r_e = tf.reshape(tf.nn.embedding_lookup(self.rel_embeddings,
                                                    r),
                             [-1, self.config.rel_hidden_size])
        pos_t_e = tf.reshape(tf.nn.embedding_lookup(self.ent_embeddings,
                                                    t),
                             [-1, self.config.ent_hidden_size, 1])
        h, r, t = sess.run([pos_h_e, pos_r_e, pos_t_e])
        return h, r, t

    def predict_embed_in_rel_space(self, h, r, t):
        """function to get the embedding value in numpy"""
        with tf.Session(config=self.config.gpu_config) as sess:
            # self.load_model(sess)
            try:
                ent_embeddings = tf.Variable(np.loadtxt("../intermediate/ent_embeddings.txt"),
                                             name="ent_embedding",
                                             dtype=np.float32)
                rel_embeddings = tf.Variable(np.loadtxt('../intermediate/rel_embeddings.txt'),
                                                  name="rel_embedding",
                                                  dtype=np.float32)
                rel_matrix = tf.Variable(np.loadtxt('../intermediate/rel_matrix.txt'),
                                                  name="rel_matrix",
                                                  dtype=np.float32)
            except FileNotFoundError:
                print("The model was not saved! Set save_model=True!")
            sess.run(tf.global_variables_initializer())
            pos_h_e = tf.reshape(tf.nn.embedding_lookup(ent_embeddings, h),
                                 [-1, self.config.ent_hidden_size, 1])
            pos_r_e = tf.reshape(tf.nn.embedding_lookup(rel_embeddings, r),
                                 [-1, self.config.rel_hidden_size])
            pos_t_e = tf.reshape(tf.nn.embedding_lookup(ent_embeddings, t),
                                 [-1, self.config.ent_hidden_size, 1])

            pos_matrix = tf.reshape(tf.nn.embedding_lookup(rel_matrix, r),
                                    [-1, self.config.rel_hidden_size, self.config.ent_hidden_size])

            pos_h_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_h_e),
                                                    [-1, self.config.rel_hidden_size]), 1)
            pos_t_e = tf.nn.l2_normalize(tf.reshape(tf.matmul(pos_matrix, pos_t_e),
                                                    [-1, self.config.rel_hidden_size]), 1)

            hr, rr, tr = sess.run([pos_h_e, pos_r_e, pos_t_e])
        return hr, rr, tr

    def display_entity_space(self, triples=None, sess=None):
        """function to display embedding"""
        viz = Visualization(triples=triples,
                            idx2entity=self.data_handler.idx2entity,
                            idx2relation=self.data_handler.idx2relation)

        viz.get_idx_n_emb(model=self, sess=sess)
        viz.reduce_dim()
        viz.draw_figure()

    def display_in_rel_space(self, fig_name=None):
        """function to display embedding"""

        h_name = []
        r_name = []
        t_name = []
        triples = self.data_handler.validation_triples_ids[:self.config.disp_triple_num]
        pos_h=[]
        pos_r=[]
        pos_t=[]
        for t in triples:
            h_name.append(self.data_handler.idx2entity[t.h])
            r_name.append(self.data_handler.idx2relation[t.r])
            t_name.append(self.data_handler.idx2entity[t.t])
            pos_h.append(t.h)
            pos_r.append(t.r)
            pos_t.append(t.t)

        h_emb, r_emb, t_emb = self.predict_embed_in_rel_space(pos_h,pos_r,pos_t)

        h_emb = np.array(h_emb)
        r_emb = np.array(r_emb)
        t_emb = np.array(t_emb)

        length = len(h_emb)
        x = np.concatenate((h_emb, r_emb, t_emb), axis=0)
        x_reduced = TSNE(n_components=2).fit_transform(x)

        h_embs = x_reduced[:length, :]
        r_embs = x_reduced[length:2 * length, :]
        t_embs = x_reduced[2 * length:3 * length, :]

        viz = Visualization()
        viz.draw_figure_v2(triples,
                           h_name,
                           r_name,
                           t_name,
                           h_embs,
                           r_embs,
                           t_embs,
                           fig_name)

    def save_model(self,sess):
        """function to save the model"""
        ee,er,rm = sess.run([self.ent_embeddings,self.rel_embeddings,self.rel_matrix])
        if not os.path.exists(self.config.tmp):
            os.mkdir('../intermediate')
        np.savetxt('../intermediate/ent_embeddings.txt', ee)
        np.savetxt('../intermediate/rel_embeddings.txt', er)
        np.savetxt('../intermediate/rel_matrix.txt', rm)

        # saver = tf.train.Saver()
        # saver.save(sess, '../intermediate/TransRModel.vec')

    def load_model(self, sess):
        """function to load the model"""
        saver = tf.train.Saver()
        saver.restore(sess, self.config.tmp + '/TransRModel.vec')

    def summary(self):
        """function to print the summary"""
        print("\n----------------SUMMARY----------------")
        # Acquire the max length and add four more spaces
        maxspace = len(max([k for k in self.config.__dict__.keys()])) + 15
        for key, val in self.config.__dict__.items():
            if 'gpu' in key:
                continue
            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            print(key, ":", val)
        print("-----------------------------------------")
    # TODO: Save summary
    # with open('../intermediate/TransRModel_summary.json', 'wb') as fp:
    # 	json.dump(self.config.__dict__, fp)


def main():
    parser = ArgumentParser(description='Knowledge Graph Embedding with TransR')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=200, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=100, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=5, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')

    args = parser.parse_args()

    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    data_handler = DataPrep(args.dataset)

    config = TransRConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          test_step=args.test_step,
                          test_num=args.test_num,
                          gpu_fraction=args.gpu_frac)

    model = TransR(config=config, data_handler=data_handler)
    model.summary()
    model.train()

    if model.config.disp_summary:
        model.summary()
    if model.config.disp_result:
        model.display_in_rel_space(fig_name='TransR_Result')


if __name__ == "__main__":
    main()
