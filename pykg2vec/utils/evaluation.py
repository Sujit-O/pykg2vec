#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for evaluating the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd

sys.path.append("D:\dev\pykg2vec\pykg2vec")
from core.KGMeta import EvaluationMeta
import tensorflow as tf


# from pykg2vec.core.KGMeta import Evaluation


class Evaluation(EvaluationMeta):

    def __init__(self, hits=None, model=None, test_data=None):

        self.mean_rank_head = {}
        self.mean_rank_tail = {}
        self.filter_mean_rank_head = {}
        self.filter_mean_rank_tail = {}

        self.norm_mean_rank_head = {}
        self.norm_mean_rank_tail = {}
        self.norm_filter_mean_rank_head = {}
        self.norm_filter_mean_rank_tail = {}

        self.hit_head = {}
        self.hit_tail = {}
        self.filter_hit_head = {}
        self.filter_hit_tail = {}

        self.norm_hit_head = {}
        self.norm_hit_tail = {}
        self.norm_filter_hit_head = {}
        self.norm_filter_hit_tail = {}
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

        if hits is None:
            self.hits = model.config.hits
        else:
            self.hits = hits

    def test(self, sess=None, epoch=None):
        head_rank, tail_rank, norm_head_rank, norm_tail_rank = self.model.test()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        rank_head = []
        rank_tail = []
        filter_rank_head = []
        filter_rank_tail = []

        norm_rank_head = []
        norm_rank_tail = []
        norm_filter_rank_head = []
        norm_filter_rank_tail = []

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

            rank_head.append(hrank)
            rank_tail.append(trank)
            filter_rank_head.append(fhrank)
            filter_rank_tail.append(ftrank)

            norm_rank_head.append(norm_hrank)
            norm_rank_tail.append(norm_trank)
            norm_filter_rank_head.append(norm_fhrank)
            norm_filter_rank_tail.append(norm_ftrank)

        self.mean_rank_head[epoch] = np.sum(rank_head, dtype=np.float32) / self.model.config.test_num
        self.mean_rank_tail[epoch] = np.sum(rank_tail, dtype=np.float32) / self.model.config.test_num

        self.filter_mean_rank_head[epoch] = np.sum(filter_rank_head,
                                                   dtype=np.float32) / self.model.config.test_num
        self.filter_mean_rank_tail[epoch] = np.sum(filter_rank_tail,
                                                   dtype=np.float32) / self.model.config.test_num

        self.norm_mean_rank_head[epoch] = np.sum(norm_rank_head,
                                                 dtype=np.float32) / self.model.config.test_num
        self.norm_mean_rank_tail[epoch] = np.sum(norm_rank_tail,
                                                 dtype=np.float32) / self.model.config.test_num

        self.norm_filter_mean_rank_head[epoch] = np.sum(norm_filter_rank_head,
                                                        dtype=np.float32) / self.model.config.test_num
        self.norm_filter_mean_rank_tail[epoch] = np.sum(norm_filter_rank_tail,
                                                        dtype=np.float32) / self.model.config.test_num

        for hit in self.hits:
            self.hit_head[(epoch, hit)] = np.sum(np.asarray(rank_head) < hit,
                                                 dtype=np.float32) / self.model.config.test_num
            self.hit_tail[(epoch, hit)] = np.sum(np.asarray(rank_tail) < hit,
                                                 dtype=np.float32) / self.model.config.test_num
            self.filter_hit_head[(epoch, hit)] = np.sum(np.asarray(filter_rank_head) < hit,
                                                        dtype=np.float32) / self.model.config.test_num
            self.filter_hit_tail[(epoch, hit)] = np.sum(np.asarray(filter_rank_tail) < hit,
                                                        dtype=np.float32) / self.model.config.test_num

            self.norm_hit_head[(epoch, hit)] = np.sum(np.asarray(norm_rank_head) < hit,
                                                      dtype=np.float32) / self.model.config.test_num
            self.norm_hit_tail[(epoch, hit)] = np.sum(np.asarray(norm_rank_tail) < hit,
                                                      dtype=np.float32) / self.model.config.test_num
            self.norm_filter_hit_head[(epoch, hit)] = np.sum(np.asarray(norm_filter_rank_head) < hit,
                                                             dtype=np.float32) / self.model.config.test_num
            self.norm_filter_hit_tail[(epoch, hit)] = np.sum(np.asarray(norm_filter_rank_tail) < hit,
                                                             dtype=np.float32) / self.model.config.test_num

    def save_test_summary(self):
        if not os.path.exists(self.model.config.result):
            os.mkdir(self.model.config.result)

        files = os.listdir(self.model.config.result)
        l = len([f for f in files if 'TransE' in f])
        with open(self.model.config.result + '/' + self.model.model_name + '_summary_' + str(l) + '.txt', 'w') as fh:
            fh.write('----------------SUMMARY----------------\n')
            for key, val in self.model.config.__dict__.items():
                if 'gpu' in key:
                    continue
                if not isinstance(val, str):
                    if isinstance(val, list):
                        v_tmp = '['
                        for i, v in enumerate(val):
                            if i == 0:
                                v_tmp += str(v)
                            else:
                                v_tmp += ',' + str(v)
                        v_tmp += ']'
                        val = v_tmp
                    else:
                        val = str(val)
                fh.write(key + ':' + val + '\n')
            fh.write('-----------------------------------------\n')
        columns = ['Epoch', 'mean_rank', 'filter_mean_rank',
                   'norm_mean_rank', 'norm_filter_mean_rank']
        for hit in self.hits:
            columns += ['hits' + str(hit), 'filter_hits' + str(hit),
                        'norm_hit' + str(hit), 'norm_filter_hit' + str(hit)]

        results = []
        for epoch in self.epoch:
            res_tmp= [epoch, (self.mean_rank_head[epoch] + self.mean_rank_tail[epoch]) / 2,
                      (self.filter_mean_rank_head[epoch] + self.filter_mean_rank_tail[epoch]) / 2,
                      (self.norm_mean_rank_head[epoch] + self.norm_mean_rank_tail[epoch]) / 2,
                      (self.norm_filter_mean_rank_head[epoch] + self.norm_filter_mean_rank_tail[epoch]) / 2]

            for hit in self.hits:
                res_tmp.append((self.hit_head[(epoch, hit)] + self.hit_tail[(epoch, hit)]) / 2)
                res_tmp.append((self.filter_hit_head[(epoch, hit)] + self.filter_hit_tail[(epoch, hit)]) / 2)
                res_tmp.append((self.norm_hit_head[(epoch, hit)] + self.norm_hit_tail[(epoch, hit)]) / 2)
                res_tmp.append((self.norm_filter_hit_head[(epoch, hit)] + self.norm_filter_hit_tail[(epoch, hit)]) / 2)
            results.append(res_tmp)

        df = pd.DataFrame(results, columns=columns)
        with open(self.model.config.result + '/' + self.model.model_name + '_results_' + str(l) + '.csv', 'w') as fh:
            df.to_csv(fh)

    def display_summary(self, epoch):
        print("---------------Test Results: Epoch: %d----------------" % epoch)
        print('--mean rank          : %.4f' % ((self.mean_rank_head[epoch] +
                                                self.mean_rank_tail[epoch]) / 2))
        print('--filtered mean rank : %.4f' % ((self.filter_mean_rank_head[epoch] +
                                                self.filter_mean_rank_tail[epoch]) / 2))
        print('--norm mean rank     : %.4f' % ((self.norm_mean_rank_head[epoch] +
                                                self.norm_mean_rank_tail[epoch]) / 2))
        print('--norm fil mean rank : %.4f' % ((self.norm_filter_mean_rank_head[epoch] +
                                                self.norm_filter_mean_rank_tail[epoch]) / 2))
        for hit in self.hits:
            print('--hits%d             : %.4f ' % (hit, (self.hit_head[(epoch, hit)] +
                                                          self.hit_tail[(epoch, hit)]) / 2))
            print('--filter hits%d      : %.4f ' % (hit, (self.filter_hit_head[(epoch, hit)] +
                                                          self.filter_hit_tail[(epoch, hit)]) / 2))
            print('--norm hits%d        : %.4f ' % (hit, (self.norm_hit_head[(epoch, hit)] +
                                                          self.norm_hit_tail[(epoch, hit)]) / 2))
            print('--norm filter hits%d : %.4f ' % (hit, (self.norm_filter_hit_head[(epoch, hit)] +
                                                          self.norm_filter_hit_tail[(epoch, hit)]) / 2))
        print("-----------------------------------------------------")

    def print_test_summary(self, epoch=None):
        if epoch:
            self.display_summary(epoch)
        else:
            for epoch in self.epoch:
                self.display_summary(epoch)


if __name__ == '__main__':
    e = Evaluation()
