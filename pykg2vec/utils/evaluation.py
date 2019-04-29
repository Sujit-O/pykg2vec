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
from utils.generator import Generator
from utils.dataprep import DataStats
from config.global_config import GeneratorConfig
import pickle
from multiprocessing.pool import ThreadPool
from pathlib import Path


class Evaluation(EvaluationMeta):

    def __init__(self, model=None, debug=False):
        self.model = model
        self.debug = debug
        self.batch = self.model.config.batch_size

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

        with open(self.model.config.tmp_data / 'hr_t.pkl', 'rb') as f:
            self.hr_t = pickle.load(f)
        with open(self.model.config.tmp_data / 'tr_h.pkl', 'rb') as f:
            self.tr_h = pickle.load(f)
        with open(self.model.config.tmp_data / 'data_stats.pkl', 'rb') as f:
            self.data_stats = pickle.load(f)

    def test_batch(self, sess=None, epoch=None, test_data='test'):

        head_rank, tail_rank = self.model.test_batch()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        rank_head = []
        rank_tail = []
        filter_rank_head = []
        filter_rank_tail = []

        gen_test = Generator(config=GeneratorConfig(data='test', algo=self.model.model_name,
                                                         batch_size=self.model.config.batch_size))
        self.n_test = min(self.n_test, self.data_stats.tot_test_triples)
        loop_len = self.n_test // self.batch if not self.debug else 2
        print("Testing [%d/%d] Triples" % (self.n_test, self.data_stats.tot_test_triples))
        total_test = loop_len * self.batch
        if self.n_test < self.batch:
            loop_len = 1
            total_test = self.n_test

        for i in range(loop_len):
            data = list(next(gen_test))
            ph = data[0]
            pr = data[1]
            pt = data[2]
            feed_dict = {
                self.model.test_h_batch: ph,
                self.model.test_r_batch: pr,
                self.model.test_t_batch: pt}

            id_replace_head, id_replace_tail = sess.run([head_rank, tail_rank], feed_dict)
            do = ThreadPool(20)

            hdata = do.map(self.zip_eval_batch_head, zip(id_replace_head, ph, pt, pr))
            tdata = do.map(self.zip_eval_batch_tail, zip(id_replace_tail, ph, pr, pt))

            rank_head += [i for i, _ in hdata]
            rank_tail += [i for i, _ in tdata]
            filter_rank_head += [i for _, i in hdata]
            filter_rank_tail += [i for _, i in tdata]

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
        gen_test.stop()

    def test(self, sess=None, epoch=None, test_data='test'):
        if test_data == 'test':
            with open(self.model.config.tmp_data / 'test_triples_ids.pkl', 'rb') as f:
                data = pickle.load(f)
        elif test_data == 'valid':
            with open(self.model.config.tmp_data / 'validation_triples_ids.pkl', 'rb') as f:
                data = pickle.load(f)
        else:
            raise NotImplementedError('Invalid testing data: enter test or valid!')

        head_rank, tail_rank = self.model.test_step()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        rank_head = []
        rank_tail = []
        filter_rank_head = []
        filter_rank_tail = []
        loop_len = self.model.config.test_num
        total_test = loop_len

        for i in range(loop_len):
            # print("batch_id:", i)
            t = data[i]
            feed_dict = {
                self.model.test_h: np.reshape(t.h, [1, ]),
                self.model.test_r: np.reshape(t.r, [1, ]),
                self.model.test_t: np.reshape(t.t, [1, ])
            }

            id_replace_head, id_replace_tail = sess.run([head_rank, tail_rank], feed_dict)

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

    def eval_batch_head(self, id_replace_head, e1, e2, r_rev):
        hrank = 0
        fhrank = 0

        for j in range(len(id_replace_head)):
            val = id_replace_head[-j - 1]
            if val == e1:
                break
            else:
                hrank += 1
                fhrank += 1
                if val in self.tr_h[(e2, r_rev)]:
                    fhrank -= 1

        return hrank, fhrank

    def zip_eval_batch_head(self, data):
        return self.eval_batch_head(*data)

    def eval_batch_tail(self, id_replace_tail, e1, r, e2):
        trank = 0
        ftrank = 0

        for j in range(len(id_replace_tail)):
            val = id_replace_tail[-j - 1]
            if val == e2:
                break
            else:
                trank += 1
                ftrank += 1
                if val in self.hr_t[(e1, r)]:
                    ftrank -= 1
        return trank, ftrank

    def zip_eval_batch_tail(self, data):
        return self.eval_batch_tail(*data)

    def test_conve(self, sess=None, epoch=None):
        head_rank, tail_rank = self.model.test_step()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        rank_head = []
        rank_tail = []
        filter_rank_head = []
        filter_rank_tail = []

        gen_test = Generator(config=GeneratorConfig(data='test', algo=self.model.model_name,
                                                         batch_size=self.model.config.batch_size))
        self.n_test = min(self.n_test, self.data_stats.tot_test_triples)
        loop_len = self.n_test // self.batch if not self.debug else 2
        print("Testing [%d/%d] Triples" % (self.n_test, self.data_stats.tot_test_triples))
        total_test = loop_len * self.batch
        if self.n_test < self.batch:
            loop_len = 1
            total_test = self.n_test

        for i in range(loop_len):
            data = list(next(gen_test))

            e1 = data[0]
            r = data[1]
            e2_multi1 = data[2]
            e2 = data[3]
            r_rev = data[4]
            e2_multi2 = data[5]

            feed_dict = {
                self.model.test_e1: e1,
                self.model.test_e2: e2,
                self.model.test_r: r,
                self.model.test_r_rev: r_rev,
                self.model.test_e2_multi1: e2_multi1,
                self.model.test_e2_multi2: e2_multi2
            }

            id_replace_head, id_replace_tail = sess.run([head_rank, tail_rank], feed_dict)

            do = ThreadPool(50)
            hdata = do.map(self.zip_eval_batch_head, zip(id_replace_head, e1, e2, r_rev))
            tdata = do.map(self.zip_eval_batch_tail, zip(id_replace_tail, e1, r, e2))
            rank_head += [i for i, _ in hdata]
            rank_tail += [i for i, _ in tdata]
            filter_rank_head += [i for _, i in hdata]
            filter_rank_tail += [i for _, i in tdata]

        gen_test.stop()
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
        gen_test = Generator(config=GeneratorConfig(data='test', algo=self.model.model_name))
        loop_len = self.n_test // self.batch if not self.debug else 100
        total_test = loop_len * self.batch

        for i in range(loop_len):
            # print("batch_id:", i)
            data = list(next(gen_test))
            # import pdb
            # pdb.set_trace()
            feed_dict = {
                self.model.test_h: data[:, 0],
                self.model.test_r: data[:, 1],
                self.model.test_t: data[:, 2]
            }

            id_replace_head, id_replace_tail = sess.run([head_rank, tail_rank], feed_dict)

            for b in range(self.batch):
                hrank = 0
                fhrank = 0

                for j in range(len(id_replace_head[b])):
                    val = id_replace_head[b, -j - 1]
                    if val == data[:, 0][b]:
                        break
                    else:
                        hrank += 1
                        fhrank += 1
                        if val in self.tr_h[(data[:, 2][b], data[:, 1][b])]:
                            fhrank -= 1

                trank = 0
                ftrank = 0

                for j in range(len(id_replace_tail[b])):
                    val = id_replace_tail[b, -j - 1]
                    if val == data[:, 2][b]:
                        break
                    else:
                        trank += 1
                        ftrank += 1
                        if val in self.hr_t[(data[:, 0][b], data[:, 1][b])]:
                            ftrank -= 1

                rank_head.append(hrank)
                rank_tail.append(trank)
                filter_rank_head.append(fhrank)
                filter_rank_tail.append(ftrank)
        gen_test.stop()
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
