#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
sys.path.append("../")

from core.KGMeta import ModelMeta, TrainerMeta
from utils.visualization import Visualization

class Rescal(ModelMeta):
    """
    ------------------Paper Title-----------------------------
    A Three-Way Model for Collective Learning on Multi-Relational Data
    ------------------Paper Authors---------------------------
    Maximilian Nickel, Volker Tresp, Hans-Peter Kriegel
    Ludwig-Maximilians-Universitat, Munich, Germany 
    Siemens AG, Corporate Technology, Munich, Germany
    {NICKEL@CIP.IFI.LMU.DE, VOLKER.TRESP@SIEMENS.COM, KRIEGEL@DBS.IFI.LMU.DE}
    ------------------Summary---------------------------------
    RESCAL is a tensor factorization approach to knowledge representation learning, 
    which is able to perform collective learning via the latent components of the factorization.
    
    Portion of Code Based on https://github.com/mnick/rescal.py/blob/master/rescal/rescal.py
     and https://github.com/thunlp/OpenKE/blob/master/models/RESCAL.py
    """
    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.model_name = 'Rescal'
       
    def def_inputs(self):
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

    def def_parameters(self):
        num_total_ent = self.data_handler.tot_entity
        num_total_rel = self.data_handler.tot_relation
        k = self.config.hidden_size

        with tf.name_scope("embedding"):
            # A: per each entity, store its embedding representation.
            self.ent_embeddings = tf.get_variable(name="ent_embedding",
                                                  shape=[num_total_ent, k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            # M: per each relation, store a matrix that models the interactions between entity embeddings.
            self.rel_matrices   = tf.get_variable(name="rel_matrices",
                                                  shape=[num_total_rel, k*k],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False))

            self.parameter_list = [self.ent_embeddings, self.rel_matrices]

    def cal_truth_val(self, h, r, t):
        # dim of h: [m, k, 1]
        #        r: [m, k, k]
        #        t: [m, k, 1]
        return tf.reduce_sum(h*tf.matmul(r,t), [1,2])

    def def_loss(self):
        k = self.config.hidden_size

        with tf.name_scope('normalization'):
            self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
            self.rel_matrices = tf.nn.l2_normalize(self.rel_matrices, axis=1)
        
        with tf.name_scope('lookup_embeddings'):
            pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
            pos_r_e = tf.nn.embedding_lookup(self.rel_matrices, self.pos_r)
            pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
            neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
            neg_r_e = tf.nn.embedding_lookup(self.rel_matrices, self.neg_r)
            neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)

        with tf.name_scope('reshaping'):
            pos_h_e = tf.reshape(pos_h_e, [-1, k, 1])
            pos_r_e = tf.reshape(pos_r_e, [-1, k, k])
            pos_t_e = tf.reshape(pos_t_e, [-1, k, 1])
            neg_h_e = tf.reshape(neg_h_e, [-1, k, 1])
            neg_r_e = tf.reshape(neg_r_e, [-1, k, k])
            neg_t_e = tf.reshape(neg_t_e, [-1, k, 1])
        
        pos_score = self.cal_truth_val(pos_h_e, pos_r_e, pos_t_e)
        neg_score = self.cal_truth_val(neg_h_e, neg_r_e, neg_t_e)

        self.loss = tf.reduce_sum(tf.maximum(neg_score + self.config.margin - pos_score, 0))

    def test_step(self):
        k = self.config.hidden_size
        num_entity = self.data_handler.tot_entity

        with tf.name_scope('lookup_embeddings'):
            h_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_h)
            r_vec = tf.nn.embedding_lookup(self.rel_matrices, self.test_r)
            t_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_t)
 
        with tf.name_scope('reshaping'):
            h_vec = tf.reshape(h_vec, [k, 1])
            r_vec = tf.reshape(r_vec, [k, k])
            t_vec = tf.reshape(t_vec, [k, 1])
       
        h_sim = tf.matmul(self.ent_embeddings, tf.matmul(r_vec, t_vec))
        t_sim = tf.transpose(tf.matmul(tf.matmul(tf.transpose(h_vec), r_vec), tf.transpose(self.ent_embeddings)))

        with tf.name_scope('normalization'):
            self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, axis=1)
            self.rel_matrices = tf.nn.l2_normalize(self.rel_matrices, axis=1)

        with tf.name_scope('lookup_embeddings'):
            norm_h_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_h)
            norm_r_vec = tf.nn.embedding_lookup(self.rel_matrices, self.test_r)
            norm_t_vec = tf.nn.embedding_lookup(self.ent_embeddings, self.test_t)

        with tf.name_scope('reshaping'):        
            norm_h_vec = tf.reshape(norm_h_vec, [k, 1])
            norm_r_vec = tf.reshape(norm_r_vec, [k, k])
            norm_t_vec = tf.reshape(norm_t_vec, [k, 1])

        norm_h_sim = tf.matmul(self.ent_embeddings, tf.matmul(norm_r_vec, norm_t_vec))
        norm_t_sim = tf.transpose(tf.matmul(tf.matmul(tf.transpose(norm_h_vec), norm_r_vec), tf.transpose(self.ent_embeddings)))

        _, self.head_rank      = tf.nn.top_k(tf.reduce_sum(tf.negative(h_sim), 1), k=num_entity)
        _, self.tail_rank      = tf.nn.top_k(tf.reduce_sum(tf.negative(t_sim), 1), k=num_entity)
        _, self.norm_head_rank = tf.nn.top_k(tf.reduce_sum(tf.negative(norm_h_sim), 1), k=num_entity)
        _, self.norm_tail_rank = tf.nn.top_k(tf.reduce_sum(tf.negative(norm_t_sim), 1), k=num_entity)

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

    def display(self, sess=None):
        """function to display embedding"""
        if self.config.plot_embedding:
            triples = self.data_handler.validation_triples_ids[:self.config.disp_triple_num]
            viz = Visualization(triples=triples,
                                idx2entity=self.data_handler.idx2entity,
                                idx2relation=self.data_handler.idx2relation)

            viz.get_idx_n_emb(model=self, sess=sess)
            viz.reduce_dim()
            viz.plot_embedding(resultpath=self.config.figures, algos=self.model_name)

        if self.config.plot_training_result:
            viz = Visualization()
            viz.plot_train_result(path=self.config.result,
                                  result=self.config.figures,
                                  algo=['TransE', 'TransR', 'TransH'],
                                  data=['Freebase15k'])

        if self.config.plot_testing_result:
            viz = Visualization()
            viz.plot_test_result(path=self.config.result,
                                 result=self.config.figures,
                                 algo=['TransE', 'TransR', 'TransH'],
                                 data=['Freebase15k'], paramlist=None, hits=self.config.hits)

def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with Rescal')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='../intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase15k', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=1, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=100, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=5, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.1, type=float, help='learning rate')
    parser.add_argument('-gp', '--gpu_frac', default=0.4, type=float, help='GPU fraction to use')
    parser.add_argument('-k', '--embed', default=50, type=int, help='Hidden embedding size')
    args = parser.parse_args()

    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    data_handler = DataPrep(args.dataset)
    args.test_num = min(len(data_handler.test_triples_ids), args.test_num)
    
    config = RescalConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          hidden_size=args.embed)

    config.test_step = args.test_step
    config.test_num  = args.test_num
    config.gpu_fraction = args.gpu_frac

    model = Rescal(config, data_handler)
    
    trainer = Trainer(model=model)
    trainer.build_model()
    trainer.train_model()

if __name__ == "__main__":
    tf.app.run()