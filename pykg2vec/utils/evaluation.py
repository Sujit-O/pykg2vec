#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for evaluating the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("D:\dev\pykg2vec\pykg2vec")
from core.KGMeta import Evaluation

# from pykg2vec.core.KGMeta import Evaluation
import numpy as np


class EvaluationTransE(Evaluation):

    def __init__(self, model=None, test_data=None):
        self.rank_head = []
        self.rank_tail = []
        self.filter_rank_head = []
        self.filter_rank_tail = []

        self.norm_rank_head = []
        self.norm_rank_tail = []
        self.norm_filter_rank_head = []
        self.norm_filter_rank_tail = []

        self.mean_rank_head = {}
        self.mean_rank_tail = {}
        self.filter_mean_rank_head = {}
        self.filter_mean_rank_tail = {}

        self.norm_mean_rank_head = {}
        self.norm_mean_rank_tail = {}
        self.norm_filter_mean_rank_head = {}
        self.norm_filter_mean_rank_tail = {}

        self.hit10_head = {}
        self.hit10_tail = {}
        self.filter_hit10_head = {}
        self.filter_hit10_tail = {}

        self.norm_hit10_head = {}
        self.norm_hit10_tail = {}
        self.norm_filter_hit10_head = {}
        self.norm_filter_hit10_tail = {}
        self.epoch = []

        if test_data == 'test':
            self.data = model.data_handler.test_triples_ids
        elif test_data == 'valid':
            self.data = model.data_handler.validation_triples_ids
        else:
            raise NotImplementedError('Invalid testing data: enter test or valid!')

        self.hr_t = model.data_handler.hr_t
        self.tr_h = model.data_handler.tr_t
        self.n_test = model.config.test_num
        self.model = model

    def test(self, sess=None, epoch=None):
        head_rank, tail_rank, norm_head_rank, norm_tail_rank = self.model.test()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')
        for i in range(self.model.config.test_num):
            t = self.data[i]
            feed_dict = {
                self.model.test_h: np.reshape(t.h, [1, ]),
                self.model.test_r: np.reshape(t.r, [1, ]),
                self.model.test_t: np.reshape(t.r, [1, ])
            }
            (id_replace_head,
             id_replace_tail,
             norm_id_replace_head,
             norm_id_replace_tail) = sess.run([head_rank,
                                               tail_rank,
                                               norm_head_rank,
                                               norm_tail_rank], feed_dict)
            hrank = 0
            fhrank = 0
            for i in range(len(id_replace_head)):
                val = id_replace_head[-i - 1]
                if val == t.h:
                    break
                else:
                    hrank += 1
                    fhrank += 1
                    if val in self.tr_h[(t.t, t.r)]:
                        fhrank -= 1

            norm_hrank = 0
            norm_fhrank = 0
            for i in range(len(norm_id_replace_head)):
                val = norm_id_replace_head[-i - 1]
                if val == t.h:
                    break
                else:
                    norm_hrank += 1
                    norm_fhrank += 1
                    if val in self.tr_h[(t.t, t.r)]:
                        norm_fhrank -= 1

            trank = 0
            ftrank = 0
            for i in range(len(id_replace_tail)):
                val = id_replace_tail[-i - 1]
                if val == t.t:
                    break
                else:
                    trank += 1
                    ftrank += 1
                    if val in self.hr_t[(t.h, t.t)]:
                        ftrank -= 1

            norm_trank = 0
            norm_ftrank = 0
            for i in range(len(norm_id_replace_tail)):
                val = norm_id_replace_tail[-i - 1]
                if val == t.t:
                    break
                else:
                    norm_trank += 1
                    norm_ftrank += 1
                    if val in self.hr_t[(t.h, t.r)]:
                        norm_ftrank -= 1

            self.rank_head.append(hrank)
            self.rank_tail.append(trank)
            self.filter_rank_head.append(fhrank)
            self.filter_rank_tail.append(ftrank)

            self.norm_rank_head.append(norm_hrank)
            self.norm_rank_tail.append(norm_trank)
            self.norm_filter_rank_head.append(norm_fhrank)
            self.norm_filter_rank_tail.append(norm_ftrank)

        self.mean_rank_head[epoch] = np.sum(self.rank_head, dtype=np.float32) / self.model.config.test_num
        self.mean_rank_tail[epoch] = np.sum(self.rank_tail, dtype=np.float32) / self.model.config.test_num

        self.filter_mean_rank_head[epoch] = np.sum(self.filter_rank_head,
                                                   dtype=np.float32) / self.model.config.test_num
        self.filter_mean_rank_tail[epoch] = np.sum(self.filter_rank_tail,
                                                   dtype=np.float32) / self.model.config.test_num

        self.norm_mean_rank_head[epoch] = np.sum(self.norm_rank_head,
                                                 dtype=np.float32) / self.model.config.test_num
        self.norm_mean_rank_tail[epoch] = np.sum(self.norm_rank_tail,
                                                 dtype=np.float32) / self.model.config.test_num

        self.norm_filter_mean_rank_head[epoch] = np.sum(self.norm_filter_rank_head,
                                                        dtype=np.float32) / self.model.config.test_num
        self.norm_filter_mean_rank_tail[epoch] = np.sum(self.norm_filter_rank_tail,
                                                        dtype=np.float32) / self.model.config.test_num

        self.hit10_head[epoch] = np.sum(np.asarray(np.asarray(self.rank_head) < 10,
                                                   dtype=np.float32)) / self.model.config.test_num
        self.hit10_tail[epoch] = np.sum(np.asarray(np.asarray(self.rank_tail) < 10,
                                                   dtype=np.float32)) / self.model.config.test_num

        self.filter_hit10_head[epoch] = np.sum(np.asarray(np.asarray(self.filter_rank_head) < 10,
                                                          dtype=np.float32)) / self.model.config.test_num
        self.filter_hit10_tail[epoch] = np.sum(np.asarray(np.asarray(self.filter_rank_tail) < 10,
                                                          dtype=np.float32)) / self.model.config.test_num

        self.norm_hit10_head[epoch] = np.sum(np.asarray(np.asarray(self.norm_rank_head) < 10,
                                                        dtype=np.float32)) / self.model.config.test_num
        self.norm_hit10_tail[epoch] = np.sum(np.asarray(np.asarray(self.norm_rank_tail) < 10,
                                                        dtype=np.float32)) / self.model.config.test_num

        self.norm_filter_hit10_head[epoch] = np.sum(np.asarray(np.asarray(self.norm_filter_rank_head) < 10,
                                                               dtype=np.float32)) / self.model.config.test_num
        self.norm_filter_hit10_tail[epoch] = np.sum(np.asarray(np.asarray(self.norm_filter_rank_tail) < 10,
                                                               dtype=np.float32)) / self.model.config.test_num

    def print_test_summary(self, epoch=None):
        if epoch:
            print('iter:%d --mean rank: %.2f --hit@10: %.2f' % (
                epoch, (self.mean_rank_head[epoch] + self.mean_rank_tail[epoch]) / 2,
                (self.hit10_tail[epoch] + self.hit10_head[epoch]) / 2))
            print('iter:%d --filter mean rank: %.2f --filter hit@10: %.2f' % (
                epoch, (self.filter_mean_rank_head[epoch] + self.filter_mean_rank_tail[epoch]) / 2,
                (self.filter_hit10_tail[epoch] + self.filter_hit10_head[epoch]) / 2))

            print('iter:%d --norm mean rank: %.2f --norm hit@10: %.2f' % (
                epoch, (self.norm_mean_rank_head[epoch] + self.norm_mean_rank_tail[epoch]) / 2,
                (self.norm_hit10_tail[epoch] + self.norm_hit10_head[epoch]) / 2))
            print('iter:%d --norm filter mean rank: %.2f --norm filter hit@10: %.2f' % (
                epoch, (self.norm_filter_mean_rank_head[epoch] + self.norm_filter_mean_rank_tail[epoch]) / 2,
                (self.norm_filter_hit10_tail[epoch] + self.norm_filter_hit10_head[epoch]) / 2))
        else:
            for epoch in self.epoch:
                print("---------------Test Results: iter%d------------------" % epoch)
                print('iter:%d --mean rank: %.2f --hit@10: %.2f' % (
                    epoch, (self.mean_rank_head[epoch] + self.mean_rank_tail[epoch]) / 2,
                    (self.hit10_tail[epoch] + self.hit10_head[epoch]) / 2))
                print('iter:%d --filter mean rank: %.2f --filter hit@10: %.2f' % (
                    epoch, (self.filter_mean_rank_head[epoch] + self.filter_mean_rank_tail[epoch]) / 2,
                    (self.filter_hit10_tail[epoch] + self.filter_hit10_head[epoch]) / 2))

                print('iter:%d --norm mean rank: %.2f --norm hit@10: %.2f' % (
                    epoch, (self.norm_mean_rank_head[epoch] + self.norm_mean_rank_tail[epoch]) / 2,
                    (self.norm_hit10_tail[epoch] + self.norm_hit10_head[epoch]) / 2))
                print('iter:%d --norm filter mean rank: %.2f --norm filter hit@10: %.2f' % (
                    epoch, (self.norm_filter_mean_rank_head[epoch] + self.norm_filter_mean_rank_tail[epoch]) / 2,
                    (self.norm_filter_hit10_tail[epoch] + self.norm_filter_hit10_head[epoch]) / 2))
                print("-----------------------------------------------------")

class EvaluationTransH(Evaluation):

    def __init__(self, model=None, test_data=None):
        self.rank_head = []
        self.rank_tail = []
        self.filter_rank_head = []
        self.filter_rank_tail = []

        self.norm_rank_head = []
        self.norm_rank_tail = []
        self.norm_filter_rank_head = []
        self.norm_filter_rank_tail = []

        self.mean_rank_head = {}
        self.mean_rank_tail = {}
        self.filter_mean_rank_head = {}
        self.filter_mean_rank_tail = {}

        self.norm_mean_rank_head = {}
        self.norm_mean_rank_tail = {}
        self.norm_filter_mean_rank_head = {}
        self.norm_filter_mean_rank_tail = {}

        self.hit10_head = {}
        self.hit10_tail = {}
        self.filter_hit10_head = {}
        self.filter_hit10_tail = {}

        self.norm_hit10_head = {}
        self.norm_hit10_tail = {}
        self.norm_filter_hit10_head = {}
        self.norm_filter_hit10_tail = {}
        self.epoch = []

        if test_data == 'test':
            self.data = model.data_handler.test_triples_ids
        elif test_data == 'valid':
            self.data = model.data_handler.validation_triples_ids
        else:
            raise NotImplementedError('Invalid testing data: enter test or valid!')

        self.hr_t = model.data_handler.hr_t
        self.tr_h = model.data_handler.tr_t
        self.n_test = model.config.test_num
        self.model = model

    def test(self, sess=None, epoch=None):
        head_rank, tail_rank, norm_head_rank, norm_tail_rank = self.model.test()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')
        for i in range(self.model.config.test_num):
            t = self.data[i]
            feed_dict = {
                self.model.test_h: np.reshape(t.h, [1, ]),
                self.model.test_r: np.reshape(t.r, [1, ]),
                self.model.test_t: np.reshape(t.r, [1, ])
            }
            (id_replace_head,
             id_replace_tail,
             norm_id_replace_head,
             norm_id_replace_tail) = sess.run([head_rank,
                                               tail_rank,
                                               norm_head_rank,
                                               norm_tail_rank], feed_dict)
            hrank = 0
            fhrank = 0
            for i in range(len(id_replace_head)):
                val = id_replace_head[-i - 1]
                if val == t.h:
                    break
                else:
                    hrank += 1
                    fhrank += 1
                    if val in self.tr_h[(t.t, t.r)]:
                        fhrank -= 1

            norm_hrank = 0
            norm_fhrank = 0
            for i in range(len(norm_id_replace_head)):
                val = norm_id_replace_head[-i - 1]
                if val == t.h:
                    break
                else:
                    norm_hrank += 1
                    norm_fhrank += 1
                    if val in self.tr_h[(t.t, t.r)]:
                        norm_fhrank -= 1

            trank = 0
            ftrank = 0
            for i in range(len(id_replace_tail)):
                val = id_replace_tail[-i - 1]
                if val == t.t:
                    break
                else:
                    trank += 1
                    ftrank += 1
                    if val in self.hr_t[(t.h, t.t)]:
                        ftrank -= 1

            norm_trank = 0
            norm_ftrank = 0
            for i in range(len(norm_id_replace_tail)):
                val = norm_id_replace_tail[-i - 1]
                if val == t.t:
                    break
                else:
                    norm_trank += 1
                    norm_ftrank += 1
                    if val in self.hr_t[(t.h, t.r)]:
                        norm_ftrank -= 1

            self.rank_head.append(hrank)
            self.rank_tail.append(trank)
            self.filter_rank_head.append(fhrank)
            self.filter_rank_tail.append(ftrank)

            self.norm_rank_head.append(norm_hrank)
            self.norm_rank_tail.append(norm_trank)
            self.norm_filter_rank_head.append(norm_fhrank)
            self.norm_filter_rank_tail.append(norm_ftrank)

        self.mean_rank_head[epoch] = np.sum(self.rank_head, dtype=np.float32) / self.model.config.test_num
        self.mean_rank_tail[epoch] = np.sum(self.rank_tail, dtype=np.float32) / self.model.config.test_num

        self.filter_mean_rank_head[epoch] = np.sum(self.filter_rank_head,
                                                   dtype=np.float32) / self.model.config.test_num
        self.filter_mean_rank_tail[epoch] = np.sum(self.filter_rank_tail,
                                                   dtype=np.float32) / self.model.config.test_num

        self.norm_mean_rank_head[epoch] = np.sum(self.norm_rank_head,
                                                 dtype=np.float32) / self.model.config.test_num
        self.norm_mean_rank_tail[epoch] = np.sum(self.norm_rank_tail,
                                                 dtype=np.float32) / self.model.config.test_num

        self.norm_filter_mean_rank_head[epoch] = np.sum(self.norm_filter_rank_head,
                                                        dtype=np.float32) / self.model.config.test_num
        self.norm_filter_mean_rank_tail[epoch] = np.sum(self.norm_filter_rank_tail,
                                                        dtype=np.float32) / self.model.config.test_num

        self.hit10_head[epoch] = np.sum(np.asarray(np.asarray(self.rank_head) < 10,
                                                   dtype=np.float32)) / self.model.config.test_num
        self.hit10_tail[epoch] = np.sum(np.asarray(np.asarray(self.rank_tail) < 10,
                                                   dtype=np.float32)) / self.model.config.test_num

        self.filter_hit10_head[epoch] = np.sum(np.asarray(np.asarray(self.filter_rank_head) < 10,
                                                          dtype=np.float32)) / self.model.config.test_num
        self.filter_hit10_tail[epoch] = np.sum(np.asarray(np.asarray(self.filter_rank_tail) < 10,
                                                          dtype=np.float32)) / self.model.config.test_num

        self.norm_hit10_head[epoch] = np.sum(np.asarray(np.asarray(self.norm_rank_head) < 10,
                                                        dtype=np.float32)) / self.model.config.test_num
        self.norm_hit10_tail[epoch] = np.sum(np.asarray(np.asarray(self.norm_rank_tail) < 10,
                                                        dtype=np.float32)) / self.model.config.test_num

        self.norm_filter_hit10_head[epoch] = np.sum(np.asarray(np.asarray(self.norm_filter_rank_head) < 10,
                                                               dtype=np.float32)) / self.model.config.test_num
        self.norm_filter_hit10_tail[epoch] = np.sum(np.asarray(np.asarray(self.norm_filter_rank_tail) < 10,
                                                               dtype=np.float32)) / self.model.config.test_num

    def print_test_summary(self, epoch=None):
        if epoch:
            print('iter:%d --mean rank: %.2f --hit@10: %.2f' % (
                epoch, (self.mean_rank_head[epoch] + self.mean_rank_tail[epoch]) / 2,
                (self.hit10_tail[epoch] + self.hit10_head[epoch]) / 2))
            print('iter:%d --filter mean rank: %.2f --filter hit@10: %.2f' % (
                epoch, (self.filter_mean_rank_head[epoch] + self.filter_mean_rank_tail[epoch]) / 2,
                (self.filter_hit10_tail[epoch] + self.filter_hit10_head[epoch]) / 2))

            print('iter:%d --norm mean rank: %.2f --norm hit@10: %.2f' % (
                epoch, (self.norm_mean_rank_head[epoch] + self.norm_mean_rank_tail[epoch]) / 2,
                (self.norm_hit10_tail[epoch] + self.norm_hit10_head[epoch]) / 2))
            print('iter:%d --norm filter mean rank: %.2f --norm filter hit@10: %.2f' % (
                epoch, (self.norm_filter_mean_rank_head[epoch] + self.norm_filter_mean_rank_tail[epoch]) / 2,
                (self.norm_filter_hit10_tail[epoch] + self.norm_filter_hit10_head[epoch]) / 2))
        else:
            for epoch in self.epoch:
                print("---------------Test Results: iter%d------------------" % epoch)
                print('iter:%d --mean rank: %.2f --hit@10: %.2f' % (
                    epoch, (self.mean_rank_head[epoch] + self.mean_rank_tail[epoch]) / 2,
                    (self.hit10_tail[epoch] + self.hit10_head[epoch]) / 2))
                print('iter:%d --filter mean rank: %.2f --filter hit@10: %.2f' % (
                    epoch, (self.filter_mean_rank_head[epoch] + self.filter_mean_rank_tail[epoch]) / 2,
                    (self.filter_hit10_tail[epoch] + self.filter_hit10_head[epoch]) / 2))

                print('iter:%d --norm mean rank: %.2f --norm hit@10: %.2f' % (
                    epoch, (self.norm_mean_rank_head[epoch] + self.norm_mean_rank_tail[epoch]) / 2,
                    (self.norm_hit10_tail[epoch] + self.norm_hit10_head[epoch]) / 2))
                print('iter:%d --norm filter mean rank: %.2f --norm filter hit@10: %.2f' % (
                    epoch, (self.norm_filter_mean_rank_head[epoch] + self.norm_filter_mean_rank_tail[epoch]) / 2,
                    (self.norm_filter_hit10_tail[epoch] + self.norm_filter_hit10_head[epoch]) / 2))
                print("-----------------------------------------------------")



if __name__ == '__main__':
    e = Evaluation()
