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
from config.global_config import GeneratorConfig
import pickle
from multiprocessing.pool import ThreadPool
import timeit
from multiprocessing import Process, Manager, current_process
import progressbar


def eval_batch_head(id_replace_head, h, r, t, tr_h):
    hrank = 0
    fhrank = 0

    for j in range(len(id_replace_head)):
        val = id_replace_head[-j - 1]
        if val == h:
            break
        else:
            hrank += 1
            fhrank += 1
            if val in tr_h[(t, r)]:
                fhrank -= 1

    return hrank, fhrank


def eval_batch_tail(id_replace_tail, h, r, t, hr_t):
    trank = 0
    ftrank = 0

    for j in range(len(id_replace_tail)):
        val = id_replace_tail[-j - 1]
        if val == t:
            break
        else:
            trank += 1
            ftrank += 1
            if val in hr_t[(h, r)]:
                ftrank -= 1
    return trank, ftrank


def display_summary(epoch, hits, mean_rank_head, mean_rank_tail,
                    filter_mean_rank_head, filter_mean_rank_tail,
                    hit_head, hit_tail, filter_hit_head, filter_hit_tail, start_time):
    print("------Test Results: Epoch: %d --- Time Taken: %.2f--------" % (epoch, timeit.default_timer() - start_time))
    print('--mean rank          : %.4f' % ((mean_rank_head[epoch] +
                                            mean_rank_tail[epoch]) / 2))
    print('--filtered mean rank : %.4f' % ((filter_mean_rank_head[epoch] +
                                            filter_mean_rank_tail[epoch]) / 2))
    for hit in hits:
        print('--hits%d             : %.4f ' % (hit, (hit_head[(epoch, hit)] +
                                                      hit_tail[(epoch, hit)]) / 2))
        print('--filter hits%d      : %.4f ' % (hit, (filter_hit_head[(epoch, hit)] +
                                                      filter_hit_tail[(epoch, hit)]) / 2))
    print("---------------------------------------------------------")


def evaluation_process(id_replace_tail, id_replace_head, h_list,
                       r_list, t_list, tr_h, hr_t,
                       mean_rank_head, mean_rank_tail, filter_mean_rank_head,
                       filter_mean_rank_tail, hit_head, hit_tail,
                       filter_hit_head, filter_hit_tail, epoch, hits):
    rank_head = []
    rank_tail = []
    filter_rank_head = []
    filter_rank_tail = []
    total_test = len(h_list)
    start_time = timeit.default_timer()
    for triple_id in range(total_test):
        t_rank, fil_t_rank = eval_batch_tail(id_replace_tail[triple_id], h_list[triple_id],
                                             r_list[triple_id], t_list[triple_id], hr_t)
        h_rank, fil_h_rank = eval_batch_head(id_replace_head[triple_id], h_list[triple_id],
                                             r_list[triple_id], t_list[triple_id], tr_h)
        rank_head.append(h_rank)
        filter_rank_head.append(fil_h_rank)
        rank_tail.append(t_rank)
        filter_rank_tail.append(fil_t_rank)

    mean_rank_head[epoch] = np.sum(rank_head, dtype=np.float32) / total_test
    mean_rank_tail[epoch] = np.sum(rank_tail, dtype=np.float32) / total_test

    filter_mean_rank_head[epoch] = np.sum(filter_rank_head,
                                          dtype=np.float32) / total_test
    filter_mean_rank_tail[epoch] = np.sum(filter_rank_tail,
                                          dtype=np.float32) / total_test

    for hit in hits:
        hit_head[(epoch, hit)] = np.sum(np.asarray(rank_head) < hit,

                                        dtype=np.float32) / total_test
        hit_tail[(epoch, hit)] = np.sum(np.asarray(rank_tail) < hit,
                                        dtype=np.float32) / total_test
        filter_hit_head[(epoch, hit)] = np.sum(np.asarray(filter_rank_head) < hit,
                                               dtype=np.float32) / total_test
        filter_hit_tail[(epoch, hit)] = np.sum(np.asarray(filter_rank_tail) < hit,
                                               dtype=np.float32) / total_test
    del rank_head, rank_tail, filter_rank_head, filter_rank_tail

    display_summary(epoch, hits, mean_rank_head, mean_rank_tail,
                    filter_mean_rank_head, filter_mean_rank_tail,
                    hit_head, hit_tail, filter_hit_head, filter_hit_tail, start_time)

    return 0


class Evaluation(EvaluationMeta):

    def __init__(self, model=None, debug=False):
        self.model = model
        self.debug = debug
        self.size_per_batch = self.model.config.batch_size_testing

        self.n_test = model.config.test_num
        self.hits = model.config.hits

        manager = Manager()

        self.mean_rank_head = manager.dict()
        self.mean_rank_tail = manager.dict()
        self.filter_mean_rank_head = manager.dict()
        self.filter_mean_rank_tail = manager.dict()

        self.hit_head = manager.dict()
        self.hit_tail = manager.dict()
        self.filter_hit_head = manager.dict()
        self.filter_hit_tail = manager.dict()

        self.epoch = []
        self.hr_t = manager.dict()
        self.tr_h = manager.dict()

        self.hr_t = self.model.config.knowledge_graph.hr_t
        self.tr_h = self.model.config.knowledge_graph.tr_h

        self.data_stats = self.model.config.kg_meta

    def test_batch(self, sess=None, epoch=None, test_data='test'):

        head_rank, tail_rank = self.model.test_batch()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        knowledge_graph = self.model.config.knowledge_graph

        if test_data == 'test':
            eval_data = knowledge_graph.triplets['test']
        elif test_data == 'valid':
            eval_data = knowledge_graph.triplets['valid']
        else:
            raise NotImplementedError("%s datatype is not available!" % test_data)

        del knowledge_graph

        tot_data = len(eval_data)

        if self.n_test == 0:
            self.n_test = tot_data
        else:
            self.n_test = min(self.n_test, tot_data)

        loop_len = self.n_test // self.size_per_batch if not self.debug else 1

        if self.n_test < self.size_per_batch:
            loop_len = 1
        self.n_test = self.size_per_batch * loop_len

        print("Testing [%d/%d] Triples" % (self.n_test, tot_data))

        h_list = []
        r_list = []
        t_list = []

        id_replace_head = []
        id_replace_tail = []
        with progressbar.ProgressBar(max_value=loop_len) as bar:
            for i in range(loop_len):
                data = np.asarray([[eval_data[x].h, eval_data[x].r, eval_data[x].t]
                                   for x in range(self.size_per_batch * i, self.size_per_batch * (i + 1))])
                h = data[:, 0]
                r = data[:, 1]
                t = data[:, 2]

                feed_dict = {
                    self.model.test_h_batch: h,
                    self.model.test_r_batch: r,
                    self.model.test_t_batch: t}

                head_tmp, tail_tmp = np.squeeze(sess.run([head_rank, tail_rank], feed_dict))

                h_list.extend(h)
                r_list.extend(r)
                t_list.extend(t)

                if i == 0:
                    id_replace_head = head_tmp
                    id_replace_tail = tail_tmp
                else:
                    id_replace_head = np.concatenate((id_replace_head, head_tmp), axis=0)
                    id_replace_tail = np.concatenate((id_replace_tail, tail_tmp), axis=0)

                bar.update(i)

        p = Process(target=evaluation_process,
                    args=(id_replace_tail, id_replace_head, h_list, r_list, t_list, self.tr_h, self.hr_t,
                          self.mean_rank_head, self.mean_rank_tail, self.filter_mean_rank_head,
                          self.filter_mean_rank_tail, self.hit_head, self.hit_tail,
                          self.filter_hit_head, self.filter_hit_tail, epoch, self.hits))
        p.start()
        del id_replace_tail, id_replace_head, h_list, r_list, t_list

    def test_tucker_v2(self, sess=None, epoch=None):
        head_rank, tail_rank = self.model.test_batch()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        rank_head = []
        rank_tail = []
        filter_rank_head = []
        filter_rank_tail = []

        gen_test = Generator(config=GeneratorConfig(data='test', algo=self.model.model_name,
                                                    batch_size=self.size_per_batch), model_config=self.model.config)

        self.n_test = min(self.n_test, gen_test.tot_test_data)
        loop_len = self.n_test // self.size_per_batch if not self.debug else 2

        if self.n_test < self.size_per_batch:
            loop_len = 1

        total_test = loop_len * self.size_per_batch
        print("Testing [%d/%d] Triples" % (loop_len * self.size_per_batch, gen_test.tot_test_data))

        for i in range(loop_len):
            data = list(next(gen_test))
            h = data[0]
            rel = data[1]
            t = data[2]

            feed_dict = {
                self.model.test_h: h,
                self.model.test_r: rel,
                self.model.test_t: t
            }

            id_replace_head, id_replace_tail = np.squeeze(sess.run([head_rank], feed_dict))

            for i in range(self.model.config.batch_size):
                tranks, f_trank = self.eval_batch_tail(id_replace_tail[i], h[i], rel[i], t[i])
                hranks, f_hrank = self.eval_batch_head(id_replace_head[i], h[i], rel[i], t[i])
                rank_head.append(hranks)
                filter_rank_head.append(f_hrank)
                rank_tail.append(tranks)
                filter_rank_tail.append(f_trank)

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

    def test_simple(self, sess=None, epoch=None):
        head_rank = self.model.test_batch()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        rank_head = []
        # rank_tail = []
        filter_rank_head = []
        # filter_rank_tail = []

        # gen_test = Generator(config=GeneratorConfig(data='test', algo=self.model.model_name,
        #                                             batch_size=self.size_per_batch))

        with open(str(self.model.config.tmp_data / 'test_data.pkl'), 'rb') as f:
            test_data = pickle.load(f)
            rand_ids_test = np.random.permutation(len(test_data))

        loop_len = self.n_test // self.size_per_batch if not self.debug else 2

        if self.n_test < self.size_per_batch:
            loop_len = 1

        total_test = loop_len * self.size_per_batch
        print("Testing [%d/%d] Triples" % (loop_len * self.size_per_batch, len(test_data)))

        for i in range(loop_len):
            # data = list(next(gen_test))
            data = np.asarray([[test_data[x].h,
                                test_data[x].r,
                                test_data[x].t
                                ] for x in
                               rand_ids_test[
                               self.model.config.batch_size * i: self.model.config.batch_size * (i + 1)]])
            h = data[:, 0]
            rel = data[:, 1]
            t = data[:, 2]

            feed_dict = {
                self.model.test_h: h,
                self.model.test_r: rel,
                self.model.test_t: t
            }

            id_replace_head = np.squeeze(sess.run([head_rank], feed_dict))

            for i in range(self.model.config.batch_size):
                ranks, f_rank = self.eval_batch_tail(id_replace_head[i], h[i], rel[i], t[i])
                rank_head.append(ranks)
                filter_rank_head.append(f_rank)

            # import pdb
            # pdb.set_trace()
            # do = ThreadPool(20)
            # hdata = do.map(self.zip_eval_batch_tail, zip(id_replace_head, h,rel,t))

            # rank_head += [i for i, _ in hdata]
            # filter_rank_head += [i for _, i in hdata]

        # gen_test.stop()
        self.mean_rank_head[epoch] = np.sum(rank_head, dtype=np.float32) / total_test
        self.mean_rank_tail[epoch] = np.sum(rank_head, dtype=np.float32) / total_test

        self.filter_mean_rank_head[epoch] = np.sum(filter_rank_head,
                                                   dtype=np.float32) / total_test
        self.filter_mean_rank_tail[epoch] = np.sum(filter_rank_head,
                                                   dtype=np.float32) / total_test

        for hit in self.hits:
            self.hit_head[(epoch, hit)] = np.sum(np.asarray(rank_head) < hit,
                                                 dtype=np.float32) / total_test
            self.hit_tail[(epoch, hit)] = np.sum(np.asarray(rank_head) < hit,
                                                 dtype=np.float32) / total_test
            self.filter_hit_head[(epoch, hit)] = np.sum(np.asarray(filter_rank_head) < hit,
                                                        dtype=np.float32) / total_test
            self.filter_hit_tail[(epoch, hit)] = np.sum(np.asarray(filter_rank_head) < hit,
                                                        dtype=np.float32) / total_test

    def test_conve(self, sess=None, epoch=None):
        head_rank, tail_rank = self.model.test_batch()
        self.epoch.append(epoch)
        if not sess:
            raise NotImplementedError('No session found for evaluation!')

        rank_head = []
        rank_tail = []
        filter_rank_head = []
        filter_rank_tail = []

        gen_test = Generator(config=GeneratorConfig(data='test', algo=self.model.model_name,
                                                    batch_size=self.size_per_batch), model_config=self.model.config)
        self.n_test = min(self.n_test, gen_test.tot_test_data)
        loop_len = self.n_test // self.size_per_batch if not self.debug else 2

        if self.n_test < self.size_per_batch:
            loop_len = 1

        total_test = loop_len * self.size_per_batch
        print("Testing [%d/%d] Triples" % (loop_len * self.size_per_batch, gen_test.tot_test_data))

        for i in range(loop_len):
            data = list(next(gen_test))

            e1 = data[0]
            r = data[1]
            # e2_multi1 = data[2]
            e2 = data[2]
            r_rev = data[3]
            # e2_multi2 = data[5]

            feed_dict = {
                self.model.test_e1: e1,
                self.model.test_e2: e2,
                self.model.test_r: r,
                self.model.test_r_rev: r_rev
                # self.model.test_e2_multi1: e2_multi1,
                # self.model.test_e2_multi2: e2_multi2
            }

            id_replace_head, id_replace_tail = sess.run([head_rank, tail_rank], feed_dict)

            do = ThreadPool(20)
            hdata = do.map(self.zip_eval_batch_head, zip(id_replace_head, e1, e2, r_rev))
            tdata = do.map(self.zip_eval_batch_tail, zip(id_replace_tail, e1, r, e2))
            # print(hdata)
            # import pdb
            # pdb.set_trace()
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
        gen_test = Generator(config=GeneratorConfig(data='test', algo=self.model.model_name, \
                                                    batch_size=self.model.config.batch_size),
                             model_config=self.model.config)
        loop_len = self.n_test // self.size_per_batch if not self.debug else 100
        total_test = loop_len * self.size_per_batch

        for i in range(loop_len):
            # print("batch_id:", i)
            data = list(next(gen_test))
            # import pdb
            # pdb.set_trace()
            feed_dict = {
                self.model.test_h: data[0],
                self.model.test_r: data[1],
                self.model.test_t: data[2]
            }

            id_replace_head, id_replace_tail = sess.run([head_rank, tail_rank], feed_dict)

            for b in range(self.size_per_batch):
                hrank = 0
                fhrank = 0

                for j in range(len(id_replace_head[b])):
                    val = id_replace_head[b, -j - 1]
                    if val == data[0][b]:
                        break
                    else:
                        hrank += 1
                        fhrank += 1
                        if val in self.tr_h[(data[2][b], data[1][b])]:
                            fhrank -= 1

                trank = 0
                ftrank = 0

                for j in range(len(id_replace_tail[b])):
                    val = id_replace_tail[b, -j - 1]
                    if val == data[2][b]:
                        break
                    else:
                        trank += 1
                        ftrank += 1
                        if val in self.hr_t[(data[0][b], data[1][b])]:
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
        files = os.listdir(str(self.model.config.result))
        l = len([f for f in files if self.model.model_name in f if 'Training' in f])
        df = pd.DataFrame(losses, columns=['Epochs', 'Loss'])
        with open(str(self.model.config.result / (self.model.model_name + '_Training_results_' + str(l) + '.csv')),
                  'w') as fh:
            df.to_csv(fh)

    def save_test_summary(self):

        files = os.listdir(str(self.model.config.result))
        l = len([f for f in files if self.model.model_name in f if 'Testing' in f])
        with open(str(self.model.config.result / (self.model.model_name + '_summary_' + str(l) + '.txt')), 'w') as fh:
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

        with open(str(self.model.config.result / (self.model.model_name + '_Testing_results_' + str(l) + '.csv')),
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
