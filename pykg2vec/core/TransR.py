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

sys.path.append("../")
from core.KGMeta import ModelMeta, TrainerMeta
import tensorflow as tf
from utils.trainer import Trainer
from utils.visualization import Visualization
from config.config import TransRConfig
from utils.dataprep import DataPrep

import tensorflow as tf
import timeit
from argparse import ArgumentParser
import os
import numpy as np
from sklearn.manifold import TSNE


class TransR(ModelMeta):
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
        
        self.def_inputs()
        self.def_parameters()
        self.def_loss()
        
    def def_inputs(self):
        with tf.name_scope("read_inputs"):
            self.pos_h = tf.placeholder(tf.int32, [self.config.batch_size])
            self.pos_t = tf.placeholder(tf.int32, [self.config.batch_size])
            self.pos_r = tf.placeholder(tf.int32, [self.config.batch_size])
            self.neg_h = tf.placeholder(tf.int32, [self.config.batch_size])
            self.neg_t = tf.placeholder(tf.int32, [self.config.batch_size])
            self.neg_r = tf.placeholder(tf.int32, [self.config.batch_size])
            self.test_h = tf.placeholder(tf.int32, [1])
            self.test_t = tf.placeholder(tf.int32, [1])
            self.test_r = tf.placeholder(tf.int32, [1])

    def def_parameters(self):
        with tf.name_scope("embedding"):
            self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                                  shape=[self.data_handler.tot_entity,
                                                         self.config.ent_hidden_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))
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

    def def_loss(self):
        with tf.name_scope('lookup_embeddings'):
            # h and t should be [batch_size, ent_hidden_size, 1]
            # r should be [batch_size, rel_hidden_size, 1]
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
            
            # matrix should be [batch_size, rel_hidden_size, ent_hidden_size]
            pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix,
                                                           self.pos_r),
                                    [-1, self.config.rel_hidden_size, self.config.ent_hidden_size])
            neg_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix,
                                                           self.neg_r),
                                    [-1, self.config.rel_hidden_size, self.config.ent_hidden_size])

            # [batch_size, rel_hidden_size, ent_hidden_size] * 
            # h and t should be [batch_size, ent_hidden_size, 1]
            transform_pos_h_e = self.transform(pos_matrix, pos_h_e)
            transform_pos_t_e = self.transform(pos_matrix, pos_t_e)
            transform_neg_h_e = self.transform(neg_matrix, neg_h_e)
            transform_neg_t_e = self.transform(neg_matrix, neg_t_e)

            # [batch_size, rel_hidden_size, 1]
            pos_h_e = tf.nn.l2_normalize(tf.reshape(transform_pos_h_e, [-1, self.config.rel_hidden_size]), 1)
            pos_r_e = tf.nn.l2_normalize(tf.reshape(pos_r_e, [-1, self.config.rel_hidden_size]), 1)
            pos_t_e = tf.nn.l2_normalize(tf.reshape(transform_pos_t_e, [-1, self.config.rel_hidden_size]), 1)
            
            neg_h_e = tf.nn.l2_normalize(tf.reshape(transform_neg_h_e, [-1, self.config.rel_hidden_size]), 1)
            neg_r_e = tf.nn.l2_normalize(tf.reshape(neg_r_e, [-1, self.config.rel_hidden_size]), 1)
            neg_t_e = tf.nn.l2_normalize(tf.reshape(transform_neg_t_e, [-1, self.config.rel_hidden_size]), 1)

        if self.config.L1_flag:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keepdims=True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keepdims=True)
            self.predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keepdims=True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keepdims=True)
            self.predict = pos

        self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.config.margin, 0))
    
    def transform(self, matrix, embeddings):
        return tf.matmul(matrix, embeddings)

    def test_step(self):
        # embedding triples h r t
        # [1, 64], [1, 32], [1, 64]
        # head_vec, rel_vec, tail_vec = self.embed(self.test_h, self.test_r, self.test_t)
        head_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_h)
        rel_vec = tf.nn.embedding_lookup(self.rel_embeddings, self.test_r)
        tail_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_t)

        # head_vec = tf.nn.l2_normalize(head_vec, axis=1)
        # rel_vec = tf.nn.l2_normalize(rel_vec, axis=1)
        # tail_vec = tf.nn.l2_normalize(tail_vec, axis=1)
        
        # [1, 64, 1], [1, 32, 1], [1, 64, 1]
        head_vec = tf.reshape(head_vec, [-1, self.config.ent_hidden_size, 1])
        rel_vec = tf.reshape(rel_vec, [-1, self.config.rel_hidden_size, 1])
        tail_vec = tf.reshape(tail_vec, [-1, self.config.ent_hidden_size, 1])

        # get the projection matrix for the given relations.
        # [1, 32, 64] 
        pos_matrix = tf.reshape(tf.nn.embedding_lookup(self.rel_matrix, self.test_r),
                                    [-1, self.config.rel_hidden_size, self.config.ent_hidden_size])
               

        # project the head and tail on the relation space.
        # [1, 32, 1] 
        head_vec = self.transform(pos_matrix, head_vec) 
        tail_vec = self.transform(pos_matrix, tail_vec) 
        
        head_vec = tf.nn.l2_normalize(tf.reshape(head_vec, [-1, self.config.rel_hidden_size]), 1)
        rel_vec = tf.nn.l2_normalize(tf.reshape(rel_vec, [-1, self.config.rel_hidden_size]), 1)
        tail_vec = tf.nn.l2_normalize(tf.reshape(tail_vec, [-1, self.config.rel_hidden_size]), 1)

        
        # [14951, 64] * [64, 32]
        project_ent_embedding = self.transform(self.ent_embeddings, tf.transpose(tf.squeeze(pos_matrix, [0])))

        project_ent_embedding = tf.nn.l2_normalize(project_ent_embedding, axis=1)
        
        _, self.head_rank = tf.nn.top_k(tf.reduce_sum(tf.abs(project_ent_embedding + rel_vec - tail_vec),
                                                      axis=1),
                                        k=self.data_handler.tot_entity)
        _, self.tail_rank = tf.nn.top_k(tf.reduce_sum(tf.abs(head_vec + rel_vec - project_ent_embedding),
                                                      axis=1),
                                        k=self.data_handler.tot_entity)

        # normalized version
        norm_embedding_entity = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
        norm_embedding_relation = tf.nn.l2_normalize(self.rel_embeddings, axis=1)

        norm_head_vec = tf.nn.embedding_lookup(norm_embedding_entity, self.test_h)
        norm_rel_vec = tf.nn.embedding_lookup(norm_embedding_relation, self.test_r)
        norm_tail_vec = tf.nn.embedding_lookup(norm_embedding_entity, self.test_t)

        norm_head_vec = tf.matmul(norm_head_vec, tf.transpose(tf.squeeze(pos_matrix, [0])))
        norm_tail_vec = tf.matmul(norm_tail_vec, tf.transpose(tf.squeeze(pos_matrix, [0])))

        # norm_project_ent_embedding = norm_embedding_entity - tf.reduce_sum(norm_embedding_entity * norm_pos_norm, 1, keepdims = True) * norm_pos_norm

        
        _, self.norm_head_rank = tf.nn.top_k(
            tf.reduce_sum(tf.abs(project_ent_embedding + norm_rel_vec - norm_tail_vec),
                          axis=1), k=self.data_handler.tot_entity)
        _, self.norm_tail_rank = tf.nn.top_k(
            tf.reduce_sum(tf.abs(norm_head_vec + norm_rel_vec - project_ent_embedding),
                          axis=1), k=self.data_handler.tot_entity)
        return self.head_rank, self.tail_rank, self.norm_head_rank, self.norm_tail_rank

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


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with TransR')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=200, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=100, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=5, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')
    parser.add_argument('-k', '--embed', default=50, type=int, help='Hidden embedding size')

    args = parser.parse_args()

    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    data_handler = DataPrep(args.dataset)
    args.test_num = min(len(data_handler.test_triples_ids), args.test_num)

    config = TransRConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          test_step=args.test_step,
                          test_num=args.test_num,
                          gpu_fraction=args.gpu_frac)

    model = TransR(config=config, data_handler=data_handler)
    
    trainer = Trainer(model=model)
    trainer.build_model()
    trainer.train_model()

if __name__ == "__main__":
    tf.app.run()
