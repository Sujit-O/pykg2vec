#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for evaluating the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.append("../")
import os
import numpy as np
import pandas as pd
from core.KGMeta import EvaluationMeta


class Evaluation(EvaluationMeta):

    def __init__(self, model=None, test_data=None, algo=False, debug=False):
        self.model = model
        self.algo = algo
        self.debug = debug
        if not self.algo:
            self.batch = 1
        else:
            self.batch = self.model.config.batch_size
        if test_data == 'test':
            if not self.algo:
                self.data = model.data_handler.test_triples_ids
            else:
                self.data = model.data_handler.test_data
        elif test_data == 'valid':
            if not self.algo:
                self.data = model.data_handler.validation_triples_ids
            else:
                self.data = model.data_handler.valid_data
        else:
            raise NotImplementedError('Invalid testing data: enter test or valid!')

        self.hr_t = model.data_handler.hr_t
        self.tr_h = model.data_handler.tr_t
        self.n_test = model.config.test_num
        self.hits = model.config.hits

        self.mean_rank_head = {}
        self.mean_rank_tail = {}
        self.filter_mean_rank_head = {}
        self.filter_mean_rank_tail = {}

        self.hit_head = {}
        self.hit_tail = {}
        self.filter_hit_head = {}
        self.filter_hit_tail = {}

        self.epoch = []

    def test(self, sess=None, epoch=None):
        head_rank, tail_rank = self.model.test_step()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        rank_head = []
        rank_tail = []
        filter_rank_head = []
        filter_rank_tail = []
        if self.algo:
            gen = self.model.data_handler.batch_generator_test_hr_tr(batch_size=self.batch)
            loop_len = self.n_test // self.batch if not self.debug else 100
            total_test = loop_len * self.batch
        else:
            loop_len = self.model.config.test_num
            total_test = loop_len
        t = None
        e1 = None
        r = None
        e2 = None
        r_rev = None
        for i in range(loop_len):
            # print("batch_id:", i)

            if not self.algo:
                t = self.data[i]
                feed_dict = {
                    self.model.test_h: np.reshape(t.h, [1, ]),
                    self.model.test_r: np.reshape(t.r, [1, ]),
                    self.model.test_t: np.reshape(t.t, [1, ])
                }

            else:
                e1, r, e2_multi1, e2, r_rev, e2_multi2 = next(gen)

                feed_dict = {
                    self.model.test_e1: e1,
                    self.model.test_e2: e2,
                    self.model.test_r: r,
                    self.model.test_r_rev: r_rev,
                    self.model.test_e2_multi1: e2_multi1,
                    self.model.test_e2_multi2: e2_multi2
                }

            id_replace_head, id_replace_tail = sess.run([head_rank, tail_rank], feed_dict)

            if not self.algo:
                hrank = 0
                fhrank = 0
                for j in range(len(id_replace_head)):
                    val = id_replace_head[-j - 1]
                    if val == t.h:
                        break
                    else:
                        hrank += 1
                        fhrank += 1
                        if val in self.tr_h[(t.t, t.r)]:
                            fhrank -= 1
                            
                trank = 0
                ftrank = 0
                for j in range(len(id_replace_tail)):
                    val = id_replace_tail[-j - 1]
                    if val == t.t:
                        break
                    else:
                        trank += 1
                        ftrank += 1
                        if val in self.hr_t[(t.h, t.r)]:
                            ftrank -= 1
                
                rank_head.append(hrank)
                rank_tail.append(trank)
                filter_rank_head.append(fhrank)
                filter_rank_tail.append(ftrank)
            else:
                for b in range(self.batch):
                    hrank = 0
                    fhrank = 0
                    
                    for j in range(len(id_replace_head[b])):
                        val = id_replace_head[b, -j - 1]
                        if val == e1[b]:
                            break
                        else:
                            hrank += 1
                            fhrank += 1
                            if val in self.tr_h[(e2[b], r_rev[b])]:
                                fhrank -= 1

                    trank = 0
                    ftrank = 0
                    
                    for j in range(len(id_replace_tail[b])):
                        val = id_replace_tail[b, -j - 1]
                        if val == e2[b]:
                            break
                        else:
                            trank += 1
                            ftrank += 1
                            if val in self.hr_t[(e1[b], r[b])]:
                                ftrank -= 1

                    rank_head.append(hrank)
                    rank_tail.append(trank)
                    filter_rank_head.append(fhrank)
                    filter_rank_tail.append(ftrank)

        self.mean_rank_head[epoch] = np.sum(rank_head, dtype=np.float32) / total_test
        self.mean_rank_tail[epoch] = np.sum(rank_tail, dtype=np.float32) / total_test

        self.filter_mean_rank_head[epoch] = np.sum(filter_rank_head,
                                                   dtype=np.float32) / total_test
        self.filter_mean_rank_tail[epoch] = np.sum(filter_rank_tail,
                                                   dtype=np.float32) / total_test

        for hit in self.hits:
            self.hit_head[(epoch, hit)] = np.sum(np.asarray(rank_head) < hit,

                                                 dtype=np.float32) / total_test
            self.hit_tail[(epoch, hit)] = np.sum(np.asarray(rank_tail) < hit,
                                                 dtype=np.float32) / total_test
            self.filter_hit_head[(epoch, hit)] = np.sum(np.asarray(filter_rank_head) < hit,
                                                        dtype=np.float32) / total_test
            self.filter_hit_tail[(epoch, hit)] = np.sum(np.asarray(filter_rank_tail) < hit,
                                                        dtype=np.float32) / total_test

    def test_proje(self, sess=None, epoch=None):
        head_rank, tail_rank = self.model.test_step()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        rank_head = []
        rank_tail = []
        filter_rank_head = []
        filter_rank_tail = []
       
        gen = self.model.data_handler.batch_generator_train_proje(src_triples=self.model.data_handler.test_triples_ids, batch_size=self.batch)
        loop_len = self.n_test // self.batch if not self.debug else 100
        total_test = loop_len * self.batch

        for i in range(loop_len):
            # print("batch_id:", i)
            datas = next(gen)
            # import pdb
            # pdb.set_trace()
            feed_dict = {
                self.model.test_h: datas[:,0],
                self.model.test_r: datas[:,1],
                self.model.test_t: datas[:,2]
            }

            id_replace_head, id_replace_tail = sess.run([head_rank, tail_rank], feed_dict)

            for b in range(self.batch):
                hrank = 0
                fhrank = 0
                
                for j in range(len(id_replace_head[b])):
                    val = id_replace_head[b, -j - 1]
                    if val == datas[:,0][b]:
                        break
                    else:
                        hrank += 1
                        fhrank += 1
                        if val in self.tr_h[(datas[:,2][b], datas[:,1][b])]:
                            fhrank -= 1

                trank = 0
                ftrank = 0
                
                for j in range(len(id_replace_tail[b])):
                    val = id_replace_tail[b, -j - 1]
                    if val == datas[:,2][b]:
                        break
                    else:
                        trank += 1
                        ftrank += 1
                        if val in self.hr_t[(datas[:,0][b], datas[:,1][b])]:
                            ftrank -= 1

                rank_head.append(hrank)
                rank_tail.append(trank)
                filter_rank_head.append(fhrank)
                filter_rank_tail.append(ftrank)

        self.mean_rank_head[epoch] = np.sum(rank_head, dtype=np.float32) / total_test
        self.mean_rank_tail[epoch] = np.sum(rank_tail, dtype=np.float32) / total_test

        self.filter_mean_rank_head[epoch] = np.sum(filter_rank_head,
                                                   dtype=np.float32) / total_test
        self.filter_mean_rank_tail[epoch] = np.sum(filter_rank_tail,
                                                   dtype=np.float32) / total_test

        for hit in self.hits:
            self.hit_head[(epoch, hit)] = np.sum(np.asarray(rank_head) < hit,

                                                 dtype=np.float32) / total_test
            self.hit_tail[(epoch, hit)] = np.sum(np.asarray(rank_tail) < hit,
                                                 dtype=np.float32) / total_test
            self.filter_hit_head[(epoch, hit)] = np.sum(np.asarray(filter_rank_head) < hit,
                                                        dtype=np.float32) / total_test
            self.filter_hit_tail[(epoch, hit)] = np.sum(np.asarray(filter_rank_tail) < hit,
                                                        dtype=np.float32) / total_test

    def save_training_result(self, losses):
        if not os.path.exists(self.model.config.result):
            os.mkdir(self.model.config.result)

        files = os.listdir(self.model.config.result)
        l = len([f for f in files if self.model.model_name in f if 'Training' in f])
        df = pd.DataFrame(losses, columns=['Epochs', 'Loss'])
        with open(self.model.config.result + '/' + self.model.model_name + '_Training_results_' + str(l) + '.csv',
                  'w') as fh:
            df.to_csv(fh)

    def save_test_summary(self):
        if not os.path.exists(self.model.config.result):
            os.mkdir(self.model.config.result)

        files = os.listdir(self.model.config.result)
        l = len([f for f in files if self.model.model_name in f if 'Testing' in f])
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
        columns = ['Epoch', 'mean_rank', 'filter_mean_rank']
        for hit in self.hits:
            columns += ['hits' + str(hit), 'filter_hits' + str(hit)]

        results = []
        for epoch in self.epoch:
            res_tmp = [epoch, (self.mean_rank_head[epoch] + self.mean_rank_tail[epoch]) / 2,
                       (self.filter_mean_rank_head[epoch] + self.filter_mean_rank_tail[epoch]) / 2]

            for hit in self.hits:
                res_tmp.append((self.hit_head[(epoch, hit)] + self.hit_tail[(epoch, hit)]) / 2)
                res_tmp.append((self.filter_hit_head[(epoch, hit)] + self.filter_hit_tail[(epoch, hit)]) / 2)
            results.append(res_tmp)

        df = pd.DataFrame(results, columns=columns)
        with open(self.model.config.result + '/' + self.model.model_name + '_Testing_results_' + str(l) + '.csv',
                  'w') as fh:
            df.to_csv(fh)

    def display_summary(self, epoch):
        print("---------------Test Results: Epoch: %d----------------" % epoch)
        print('--mean rank          : %.4f' % ((self.mean_rank_head[epoch] +
                                                self.mean_rank_tail[epoch]) / 2))
        print('--filtered mean rank : %.4f' % ((self.filter_mean_rank_head[epoch] +
                                                self.filter_mean_rank_tail[epoch]) / 2))
        for hit in self.hits:
            print('--hits%d             : %.4f ' % (hit, (self.hit_head[(epoch, hit)] +
                                                          self.hit_tail[(epoch, hit)]) / 2))
            print('--filter hits%d      : %.4f ' % (hit, (self.filter_hit_head[(epoch, hit)] +
                                                          self.filter_hit_tail[(epoch, hit)]) / 2))
        print("-----------------------------------------------------")

    def print_test_summary(self, epoch=None):
        if epoch:
            self.display_summary(epoch)
        else:
            for epoch in self.epoch:
                self.display_summary(epoch)


if __name__ == '__main__':
    e = Evaluation()
