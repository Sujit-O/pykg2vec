#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for evaluating the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import sys
sys.path.append("../")
from core.KGMeta import EvaluationMeta
# from pykg2vec.core.KGMeta import EvaluationMeta
import timeit
from multiprocessing import Process, Queue
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
    print("------Test Results: Epoch: %d --- time: %.2f------------" % (epoch, timeit.default_timer() - start_time))
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


def save_test_summary(result_path, model_name, hits,
                      mean_rank_head, mean_rank_tail,
                      filter_mean_rank_head,
                      filter_mean_rank_tail, hit_head,
                      hit_tail, filter_hit_head, filter_hit_tail, config):
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    files = os.listdir(str(result_path))
    l = len([f for f in files if model_name in f if 'Testing' in f])
    with open(str(result_path / (model_name + '_summary_' + str(l) + '.txt')), 'w') as fh:
        fh.write('----------------SUMMARY----------------\n')
        for key, val in config.__dict__.items():
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
    for hit in hits:
        columns += ['hits' + str(hit), 'filter_hits' + str(hit)]

    results = []
    for epoch in mean_rank_head.keys():
        res_tmp = [epoch, (mean_rank_head[epoch] + mean_rank_tail[epoch]) / 2,
                   (filter_mean_rank_head[epoch] + filter_mean_rank_tail[epoch]) / 2]

        for hit in hits:
            res_tmp.append((hit_head[(epoch, hit)] + hit_tail[(epoch, hit)]) / 2)
            res_tmp.append((filter_hit_head[(epoch, hit)] + filter_hit_tail[(epoch, hit)]) / 2)
        results.append(res_tmp)

    df = pd.DataFrame(results, columns=columns)

    with open(str(result_path / (model_name + '_Testing_results_' + str(l) + '.csv')),
              'a') as fh:
        df.to_csv(fh)


def evaluation_process(result_queue, tr_h, hr_t, hits,
                       total_epoch, result_path, model_name,
                       config):
    mean_rank_head = {}
    mean_rank_tail = {}
    filter_mean_rank_head = {}
    filter_mean_rank_tail = {}

    hit_head = {}
    hit_tail = {}
    filter_hit_head = {}
    filter_hit_tail = {}

    while True:
        result = result_queue.get()
        id_replace_tail = result[0]
        id_replace_head = result[1]
        h_list = result[2]
        r_list = result[3]
        t_list = result[4]
        epoch = result[5]

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

        if epoch >= total_epoch - 1:
            save_test_summary(result_path, model_name, hits,
                              mean_rank_head, mean_rank_tail,
                              filter_mean_rank_head,
                              filter_mean_rank_tail, hit_head,
                              hit_tail, filter_hit_head, filter_hit_tail, config)
            break


class Evaluation(EvaluationMeta):

    def __init__(self, model=None, debug=False, data_type='test'):
        
        self.model = model
        self.debug = debug

        hr_t = self.model.config.knowledge_graph.read_cache_data('hr_t')
        tr_h = self.model.config.knowledge_graph.read_cache_data('tr_h')

        if data_type == 'test':
            self.eval_data = self.model.config.knowledge_graph.read_cache_data('triplets_test')
        elif data_type == 'valid':
            self.eval_data = self.model.config.knowledge_graph.read_cache_data('triplets_valid')
        else:
            raise NotImplementedError("%s datatype is not available!" % data_type)

        '''
            n_test: number of triplets to be tested
            1) if n_test == 0, test all the triplets. 
            2) if n_test >= # of testable triplets, then set n_test to # of testable triplets
        '''
        self.n_test = model.config.test_num
        if self.n_test == 0:
            self.n_test = len(self.eval_data)
        else:
            self.n_test = min(self.n_test, len(self.eval_data))

        ''' 
            loop_len: the # of loops to perform batch evaluation. 
            if debug mode is turned on, then set to only 2. 
        ''' 
        if self.n_test < self.model.config.batch_size_testing:
            self.loop_len = 1
        else:
            self.loop_len = (self.n_test // self.model.config.batch_size_testing) if not self.debug else 2
        
        self.n_test = self.model.config.batch_size_testing * self.loop_len

        self.result_queue = Queue()
        self.rank_calculator = Process(target=evaluation_process,
                                       args=(self.result_queue, tr_h, hr_t,
                                             self.model.config.hits, self.model.config.epochs,
                                             self.model.config.result,
                                             self.model.model_name,
                                             self.model.config))
        self.rank_calculator.start()

    def stop(self):
        self.rank_calculator.join()
        self.rank_calculator.terminate()

    def test_batch(self, sess=None, epoch=None):
        
        if sess == None:
            raise NotImplementedError('No session found for evaluation!')
        print("Testing [%d/%d] Triples" % (self.n_test, len(self.eval_data)))

        size_per_batch = self.model.config.batch_size_testing
        head_rank, tail_rank = self.model.test_batch()
              
        h_list = np.zeros(shape=(self.n_test,), dtype=np.int32)
        r_list = np.zeros(shape=(self.n_test,), dtype=np.int32)
        t_list = np.zeros(shape=(self.n_test,), dtype=np.int32)
        id_replace_head = np.zeros(shape=(self.n_test, self.model.config.kg_meta.tot_entity), dtype=np.int32)
        id_replace_tail = np.zeros(shape=(self.n_test, self.model.config.kg_meta.tot_entity), dtype=np.int32)

        widgets = ['Inferring for Evaluation: ', progressbar.AnimatedMarker(), " Done:",
                   progressbar.Percentage(), " ", progressbar.AdaptiveETA()]
        with progressbar.ProgressBar(max_value=self.loop_len, widgets=widgets) as bar:
            for i in range(self.loop_len):
                data = np.asarray([[self.eval_data[x].h, self.eval_data[x].r, self.eval_data[x].t]
                                   for x in range(size_per_batch * i, size_per_batch * (i + 1))])
                h = data[:, 0]
                r = data[:, 1]
                t = data[:, 2]

                feed_dict = {
                    self.model.test_h_batch: h,
                    self.model.test_r_batch: r,
                    self.model.test_t_batch: t}

                head_tmp, tail_tmp = np.squeeze(sess.run([head_rank, tail_rank], feed_dict))

                h_list[size_per_batch * i: size_per_batch * (i + 1)] = h
                r_list[size_per_batch * i: size_per_batch * (i + 1)] = r
                t_list[size_per_batch * i: size_per_batch * (i + 1)] = t

                id_replace_head[size_per_batch * i: size_per_batch * (i + 1), :] = head_tmp
                id_replace_tail[size_per_batch * i: size_per_batch * (i + 1), :] = tail_tmp
                bar.update(i)

        result_data = [id_replace_tail, id_replace_head, h_list, r_list, t_list, epoch]
        self.result_queue.put(result_data)

        del id_replace_tail, id_replace_head, h_list, r_list, t_list

    def save_training_result(self, losses):
        if not os.path.exists(self.model.config.result):
            os.mkdir(self.model.config.result)
        files = os.listdir(str(self.model.config.result))
        l = len([f for f in files if self.model.model_name in f if 'Training' in f])
        df = pd.DataFrame(losses, columns=['Epochs', 'Loss'])

        with open(str(self.model.config.result / (self.model.model_name + '_Training_results_' + str(l) + '.csv')),
                  'w') as fh:
            df.to_csv(fh)


if __name__ == '__main__':
    e = Evaluation()
