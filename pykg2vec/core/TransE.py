#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
------------------Paper Title-----------------------------
Translating Embeddings for Modeling Multi-relational Data
------------------Paper Authors---------------------------
Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran
Universite de Technologie de Compiegne – CNRS
Heudiasyc UMR 7253
Compiegne, France
{bordesan, nusunier, agarciad}@utc.fr
Jason Weston, Oksana Yakhnenko
Google
111 8th avenue
New York, NY, USA
{jweston, oksana}@google.com
------------------Summary---------------------------------
TransE is an energy based model which represents the
relationships as translations in the embedding space. Which
means that if (h,l,t) holds then the embedding of the tail
't' should be close to the embedding of head entity 'h'
plus some vector that depends on the relationship 'l'.
Both entities and relations are vectors in the same space.
|        ......>.
|      .     .
|    .    .
|  .  .
|_________________
Portion of Code Based on https://github.com/thunlp/OpenKE/blob/master/models/TransE.py
 and https://github.com/wencolani/TransE.git
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../")
from core.KGMeta import KGMeta
from utils.visualization import Visualization
from utils.evaluation import Evaluation
from config.config import TransEConfig
from utils.dataprep import DataPrep
import pdb
from tensorflow.python import debug as tf_debug


# from pykg2vec.core.KGMeta import KGMeta
# from pykg2vec.utils.visualization import Visualization
# from pykg2vec.utils.evaluation import EvaluationTransE
# from pykg2vec.utils.evaluation import EvaluationTransE
# from pykg2vec.config.config import TransEConfig
# from pykg2vec.utils.dataprep import DataPrep

import pandas as pd
import tensorflow as tf
import timeit
from argparse import ArgumentParser
import os


class TransE(KGMeta):

    def __init__(self, config=None, data_handler=None):

        """ TransE Models
		Args:
		-----Inputs-------
		"""
        self.model_name = 'TransE'
        if not config:
            self.config = TransEConfig()
        else:
            self.config = config

        self.data_handler = data_handler

        with tf.name_scope("read_inputs"):
            self.pos_h = tf.placeholder(tf.int32, [None])
            self.pos_t = tf.placeholder(tf.int32, [None])
            self.pos_r = tf.placeholder(tf.int32, [None])
            self.neg_h = tf.placeholder(tf.int32, [None])
            self.neg_t = tf.placeholder(tf.int32, [None])
            self.neg_r = tf.placeholder(tf.int32, [None])
            self.test_h = tf.placeholder(tf.int32, [1])
            self.test_t = tf.placeholder(tf.int32, [1])
            self.test_r = tf.placeholder(tf.int32, [1])

        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                                  shape=[self.data_handler.tot_entity, self.config.hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.rel_embeddings = tf.get_variable(name="rel_embedding",
                                                  shape=[self.data_handler.tot_relation, self.config.hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        self.training_results = []

        with tf.name_scope('normalization'):
            self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
            self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, axis=1)
            
        with tf.name_scope('lookup_embeddings'):
            pos_h_e, pos_r_e, pos_t_e = self.embed(self.pos_h, self.pos_r, self.pos_t)
            neg_h_e, neg_r_e, neg_t_e = self.embed(self.neg_h, self.neg_r, self.neg_t)

        if self.config.L1_flag:
            score_pos = tf.reduce_sum(tf.abs(pos_h_e + pos_r_e - pos_t_e), axis=1, keepdims=True)
            score_neg = tf.reduce_sum(tf.abs(neg_h_e + neg_r_e - neg_t_e), axis=1, keepdims=True)
        else:
            score_pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e)**2, axis=1, keepdims=True)
            score_neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e)**2, axis=1, keepdims=True)

        self.loss = tf.reduce_sum(tf.maximum(score_pos + self.config.margin - score_neg, 0))

    def train(self):
        
        with tf.Session(config=self.config.gpu_config) as sess:
            evaluate = Evaluation(model=self, test_data='test')
            
            global_step = tf.Variable(0, name="global_step", trainable=False)

            if self.config.optimizer == 'gradient':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'rms':
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate)
            elif self.config.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            else:
                raise NotImplementedError("No support for %s optimizer" % self.config.optimizer)

            grads = optimizer.compute_gradients(self.loss)
            self.op_train = optimizer.apply_gradients(grads,  global_step=global_step)

            sess.run(tf.global_variables_initializer())

            gen_train = self.data_handler.batch_generator_train(batch_size=self.config.batch_size)

            if self.config.loadFromData:
                saver = tf.train.Saver()
                saver.restore(sess, self.config.tmp + '/TransEModel.vec')

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

                    _, step, loss = sess.run([self.op_train, global_step, self.loss], feed_dict)

                    acc_loss += loss
                    batch += 1
                    print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time,
                                                                batch,
                                                                num_batch,
                                                                loss), end='\r')

                print('iter[%d] ---Train Loss: %.5f ---time: %.2f' % (
                    n_iter, acc_loss, timeit.default_timer() - start_time))
                
                self.training_results.append([n_iter, acc_loss])
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

            evaluate.save_test_summary()
            self.save_training_result()

    def save_training_result(self):
        if not os.path.exists(self.config.result):
            os.mkdir(self.config.result)

        files = os.listdir(self.config.result)
        l = len([f for f in files if 'TransE' in f if 'Training' in f])
        df = pd.DataFrame(self.training_results, columns=['Epochs','Loss'])
        with open(self.config.result + '/' + self.model_name + '_Training_results_' + str(l) + '.csv', 'w') as fh:
            df.to_csv(fh)

    def test(self):
        head_vec, rel_vec, tail_vec = self.embed(self.test_h, self.test_r, self.test_t)

        

        _, self.head_rank = tf.nn.top_k(tf.reduce_sum(tf.abs(self.ent_embeddings
                                                             + rel_vec - tail_vec),
                                                      axis=1),
                                        k=self.data_handler.tot_entity)
        _, self.tail_rank = tf.nn.top_k(tf.reduce_sum(tf.abs(head_vec
                                                             + rel_vec - self.ent_embeddings),
                                                      axis=1),
                                        k=self.data_handler.tot_entity)
        norm_embedding_entity = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_embedding_relation = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        norm_head_vec = tf.nn.embedding_lookup(norm_embedding_entity, self.test_h)
        norm_rel_vec = tf.nn.embedding_lookup(norm_embedding_relation, self.test_r)
        norm_tail_vec = tf.nn.embedding_lookup(norm_embedding_entity, self.test_t)

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
        emb_h, emb_r, emb_t = self.embed(h, r, t)
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
    # with open('../intermediate/TransEModel_summary.json', 'wb') as fp:
    # 	json.dump(self.config.__dict__, fp)


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with TransE')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=100, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=10, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=5, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')
    parser.add_argument('-k', '--embed', default=50, type=int, help='Hidden embedding size')
    args = parser.parse_args()

    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    data_handler = DataPrep(args.dataset)
    args.test_num = min(len(data_handler.test_triples_ids), args.test_num)
    
    config = TransEConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          test_step=args.test_step,
                          test_num=args.test_num,
                          gpu_fraction=args.gpu_frac,
                          hidden_size=args.embed )

    model = TransE(config=config, data_handler=data_handler)
    model.summary()
    model.train()

if __name__ == "__main__":
    tf.app.run()

