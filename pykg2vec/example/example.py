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


import tensorflow as tf
from pykg2vec.config.config import TransEConfig
from pykg2vec.utils.dataprep import DataPrep
import timeit
from pykg2vec.utils.evaluation import EvaluationTransE
from argparse import ArgumentParser
from pykg2vec.core.TransE import TransE
import os


def main(_):
    parser = ArgumentParser(description='Knowledge Graph Embedding with TransE')
    parser.add_argument('-b', '--batch', default=128, type=int, help='batch size')
    parser.add_argument('-t', '--tmp', default='/intermediate', type=str, help='Temporary folder')
    parser.add_argument('-ds', '--dataset', default='Freebase', type=str, help='Dataset')
    parser.add_argument('-l', '--epochs', default=10, type=int, help='Number of Epochs')
    parser.add_argument('-tn', '--test_num', default=5, type=int, help='Number of test triples')
    parser.add_argument('-ts', '--test_step', default=5, type=int, help='Test every _ epochs')
    parser.add_argument('-lr', '--learn_rate', default=0.01, type=float, help='learning rate')

    args = parser.parse_args()
    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)
    data_handler = DataPrep(args.dataset)
    config = TransEConfig(learning_rate=args.learn_rate,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          test_step=args.test_step,
                          test_num=args.test_num)

    model = TransE(config=config, data_handler=data_handler)
    model.summary()

    evaluate = EvaluationTransE(model, 'test')
    loss, op_train, loss_every, norm_entity = model.train()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        norm_rel = sess.run(tf.nn.l2_normalize(model.rel_embeddings, axis=1))
        sess.run(tf.assign(model.rel_embeddings, norm_rel))

        norm_ent = sess.run(tf.nn.l2_normalize(model.ent_embeddings, axis=1))
        sess.run(tf.assign(model.ent_embeddings, norm_ent))

        gen_train = model.data_handler.batch_generator_train(batch=model.config.batch_size)

        if model.config.loadFromData:
            saver = tf.train.Saver()
            saver.restore(sess, args.tmp+'/TransEModel.vec')

        if not model.config.testFlag:

            for n_iter in range(model.config.epochs):
                acc_loss = 0
                batch = 0
                num_batch = len(model.data_handler.train_triples_ids) // model.config.batch_size
                start_time = timeit.default_timer()

                for i in range(num_batch):
                    ph, pt, pr, nh, nt, nr = list(next(gen_train))

                    feed_dict = {
                        model.pos_h: ph,
                        model.pos_t: pt,
                        model.pos_r: pr,
                        model.neg_h: nh,
                        model.neg_t: nt,
                        model.neg_r: nr
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

                if n_iter % model.config.test_step == 0 or n_iter == 0 or n_iter == model.config.epochs - 1:
                    evaluate.test(sess, n_iter)
                    evaluate.print_test_summary(n_iter)

        model.save_model(sess)
        model.summary()

        triples = model.data_handler.validation_triples_ids[:model.config.disp_triple_num]
        model.display(triples, sess)


if __name__ == "__main__":
    tf.app.run()

