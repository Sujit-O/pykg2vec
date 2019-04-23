#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for preparing the data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("../")

from config.global_config import GeneratorConfig
import numpy as np
from scipy import sparse as sps
from multiprocessing import Process, Queue, Lock
from ctypes import cdll
import ctypes
import file_handler as fh


class Generator(object):
    """Generator class for the embedding algorithms
        Args:
          config: generator configuration
        Returns:
          batch for training algorithms
    """
    def __init__(self, config=None):
        self.config = config
        if not config:
            raise NotImplementedError('No configuration found for Generator!')
        self.Queue = Queue(self.config.queue_size)
        self.mem_lock = Lock()
        self.current_batch_idx = 0
        self.queue_lock = Lock()
        self.total_batch = self.config.total_Data // self.config.batch_size
        # c extension to handle data
        self.read_data = fh.read_data
        # if sys.platform.startswith('win'):
        #     file =  Path(self.config.data_path+'.dll')
        # elif sys.platform.startswith('linux'):
        #     file =  Path(self.config.data_path+'.so')
        # elif sys.platform.startswith('darwin'):
        #     file =  Path(self.config.data_path+'.dylib')
        # else:
        #     raise NotImplementedError("The file handler has not been compiled %s", sys.platform)


    def __iter__(self):
        return self

    def read_one_batch(self):
        # read the batch
        self.mem_lock.acquire()
        batch_data = self.read_data(self.current_batch_idx,
                                  self.config.batch_size,
                                  self.config.total_entity,
                                  self.config.data_path)

        #update the current_batch size for the next process

        self.current_batch_idx+=1
        if self.current_batch_idx>=self.total_batch:
            self.current_batch_idx=0
        self.mem_lock.release()

        self.queue_lock.acquire()
        #TODO: Store the batch in the queue
        self.queue_lock.release()


    def next(self):





    def batch_generator_train_hr_tr(self, batch_size=128):

        batch_size = batch_size
        array_rand_ids = np.random.permutation(len(self.train_data))
        number_of_batches = len(self.train_data) // batch_size

        # print("Number of batches:", number_of_batches)

        batch_idx = 0
        while True:
            train_data = np.asarray([[self.train_data[x].e1,
                                      self.train_data[x].r,
                                      self.train_data[x].e2_multi1] for x in
                                     array_rand_ids[batch_size * batch_idx:batch_size * (batch_idx + 1)]])

            e1 = train_data[:, 0]
            r = train_data[:, 1]
            col = []
            for k in train_data[:, 2]:
                col.append(k)

            row = []
            for k in range(batch_size):
                row.append([k] * len(col[k]))
            col_n = []
            row_n = []
            for i in range(batch_size):
                for j in range(len(col[i])):
                    col_n.append(col[i][j])
                    row_n.append(row[i][j])

            e2_multi1 = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(batch_size, self.tot_entity))

            batch_idx += 1

            yield e1, r, np.array(e2_multi1.todense())

            if batch_idx == number_of_batches:
                batch_idx = 0

    def batch_generator_test_hr_tr(self, batch_size=128):

        batch_size = batch_size
        array_rand_ids = np.random.permutation(len(self.test_data))
        number_of_batches = len(self.test_data) // batch_size

        batch_idx = 0
        while True:
            test_data = np.asarray([[self.test_data[x].e1,
                                     self.test_data[x].r,
                                     self.test_data[x].e2_multi1,
                                     self.test_data[x].e2,
                                     self.test_data[x].r_rev,
                                     self.test_data[x].e2_multi2] for x in
                                    array_rand_ids[batch_size * batch_idx:batch_size * (batch_idx + 1)]])

            e1 = test_data[:, 0]
            try:
                r = np.asarray(test_data[:, 1], dtype=int)
            except:
                import pdb
                pdb.set_trace()
            col = []
            for k in test_data[:, 2]:
                col.append(k)
            row = []
            for k in range(batch_size):
                if col[k]:
                    row.append([k] * len(col[k]))
            col_n = []
            row_n = []
            for i in range(batch_size):
                for j in range(len(col[i])):
                    col_n.append(col[i][j])
                    row_n.append(row[i][j])

            e2_multi1 = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(batch_size, self.tot_entity))

            e2 = test_data[:, 3]
            r_rev = test_data[:, 4]
            col = []
            for k in test_data[:, 5]:
                col.append(k)

            row = []
            for k in range(batch_size):
                if col[k]:
                    row.append([k] * len(col[k]))
            col_n = []
            row_n = []
            for i in range(batch_size):
                for j in range(len(col[i])):
                    col_n.append(col[i][j])
                    row_n.append(row[i][j])

            e2_multi2 = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(batch_size, self.tot_entity))

            batch_idx += 1

            yield e1, r, np.array(e2_multi1.todense()), e2, r_rev, np.array(e2_multi2.todense())

            if batch_idx == number_of_batches:
                batch_idx = 0

    def batch_generator_train(self, src_triples=None, batch_size=128):

        batch_size = batch_size // 2

        if src_triples is None:
            # TODO: add parameter for specifying the source of triple
            src_triples = self.train_triples_ids

        observed_triples = {(t.h, t.r, t.t): 1 for t in src_triples}
        # 1 as positive, 0 as negative

        array_rand_ids = np.random.permutation(len(src_triples))
        number_of_batches = len(src_triples) // batch_size

        # print("Number of batches:", number_of_batches)

        batch_idx = 0
        last_h = 0
        last_r = 0
        last_t = 0

        while True:

            pos_triples = np.asarray([[src_triples[x].h,
                                       src_triples[x].r,
                                       src_triples[x].t] for x in
                                      array_rand_ids[batch_size * batch_idx:batch_size * (batch_idx + 1)]])

            ph = pos_triples[:, 0]
            pr = pos_triples[:, 1]
            pt = pos_triples[:, 2]
            nh = []
            nr = []
            nt = []

            for t in pos_triples:
                if self.config.negative_sample == 'uniform':
                    prob = 0.5
                elif self.config.negative_sample == 'bern':
                    prob = self.relation_property[t[1]]
                else:
                    raise NotImplementedError("%s sampling not supported!" % self.config.negative_sample)

                if np.random.random() > prob:
                    idx_replace_tail = np.random.randint(self.tot_entity)

                    break_cnt = 0
                    while ((t[0], t[1], idx_replace_tail) in observed_triples
                           or (t[0], t[1], idx_replace_tail) in observed_triples):
                        idx_replace_tail = np.random.randint(self.tot_entity)
                        break_cnt += 1
                        if break_cnt >= 100:
                            break

                    if break_cnt >= 100:  # can not find new negative triple.
                        nh.append(last_h)
                        nr.append(last_r)
                        nt.append(last_t)
                    else:
                        nh.append(t[0])
                        nr.append(t[1])
                        nt.append(idx_replace_tail)
                        last_h = t[0]
                        last_r = t[1]
                        last_t = idx_replace_tail

                        observed_triples[(t[0], t[1], idx_replace_tail)] = 0

                else:
                    idx_replace_head = np.random.randint(self.tot_entity)
                    break_cnt = 0
                    while ((idx_replace_head, t[1], t[2]) in observed_triples
                           or (idx_replace_head, t[1], t[2]) in observed_triples):
                        idx_replace_head = np.random.randint(self.tot_entity)
                        break_cnt += 1
                        if break_cnt >= 100:
                            break

                    if break_cnt >= 100:  # can not find new negative triple.
                        nh.append(last_h)
                        nr.append(last_r)
                        nt.append(last_t)
                    else:
                        nh.append(idx_replace_head)
                        nr.append(t[1])
                        nt.append(t[2])
                        last_h = idx_replace_head
                        last_r = t[1]
                        last_t = t[2]

                        observed_triples[(idx_replace_head, t[1], t[2])] = 0

            batch_idx += 1

            yield ph, pr, pt, nh, nr, nt

            if batch_idx == number_of_batches:
                batch_idx = 0

    def batch_generator_bern(self, src_triples=None, batch_size=128):

        batch_size = batch_size // 2

        if src_triples is None:
            # TODO: add parameter for specifying the source of triple
            src_triples = self.train_triples_ids

        observed_triples = {(t.h, t.r, t.t): 1 for t in src_triples}
        # 1 as positive, 0 as negative

        array_rand_ids = np.random.permutation(len(src_triples))
        number_of_batches = len(src_triples) // batch_size

        # print("Number of batches:", number_of_batches)

        batch_idx = 0
        last_h = 0
        last_r = 0
        last_t = 0

        while True:

            pos_triples = np.asarray([[src_triples[x].h,
                                       src_triples[x].r,
                                       src_triples[x].t] for x in
                                      array_rand_ids[batch_size * batch_idx:batch_size * (batch_idx + 1)]])

            ph = pos_triples[:, 0]
            pr = pos_triples[:, 1]
            pt = pos_triples[:, 2]
            nh = []
            nr = []
            nt = []

            for t in pos_triples:
                prob = self.relation_property[t[1]]

                if np.random.random() > prob:
                    idx_replace_tail = np.random.randint(self.tot_entity)

                    break_cnt = 0
                    while ((t[0], t[1], idx_replace_tail) in observed_triples
                           or (t[0], t[1], idx_replace_tail) in observed_triples):
                        idx_replace_tail = np.random.randint(self.tot_entity)
                        break_cnt += 1
                        if break_cnt >= 100:
                            break

                    if break_cnt >= 100:  # can not find new negative triple.
                        nh.append(last_h)
                        nr.append(last_r)
                        nt.append(last_t)
                    else:
                        nh.append(t[0])
                        nr.append(t[1])
                        nt.append(idx_replace_tail)
                        last_h = t[0]
                        last_r = t[1]
                        last_t = idx_replace_tail

                        observed_triples[(t[0], t[1], idx_replace_tail)] = 0

                else:
                    idx_replace_head = np.random.randint(self.tot_entity)
                    break_cnt = 0
                    while ((idx_replace_head, t[1], t[2]) in observed_triples
                           or (idx_replace_head, t[1], t[2]) in observed_triples):
                        idx_replace_head = np.random.randint(self.tot_entity)
                        break_cnt += 1
                        if break_cnt >= 100:
                            break

                    if break_cnt >= 100:  # can not find new negative triple.
                        nh.append(last_h)
                        nr.append(last_r)
                        nt.append(last_t)
                    else:
                        nh.append(idx_replace_head)
                        nr.append(t[1])
                        nt.append(t[2])
                        last_h = idx_replace_head
                        last_r = t[1]
                        last_t = t[2]

                        observed_triples[(idx_replace_head, t[1], t[2])] = 0

            batch_idx += 1

            yield ph, pr, pt, nh, nr, nt

            if batch_idx == number_of_batches:
                batch_idx = 0


def test_data_prep():
    data_handler = DataPrep('Freebase15k')
    data_handler.dump()


def test_data_prep_generator():
    data_handler = DataPrep('Freebase15k')
    data_handler.dump()
    gen = data_handler.batch_generator_train(batch_size=8)
    for i in range(10):
        ph, pr, pt, nh, nr, nt = list(next(gen))
        print("")
        print("ph:", ph)
        print("pr:", pr)
        print("pt:", pt)
        print("nh:", nh)
        print("nr:", nr)
        print("nt:", nt)


def test_data_prep_generator_hr_t():
    data_handler = DataPrep('Freebase15k', algo=True)
    data_handler.dump()
    gen = data_handler.batch_generator_test_hr_tr(batch_size=128)
    # import tensorflow as tf
    for i in range(1000):
        e1, r, e2_multi1, e2, r_rev, e2_multi2 = list(next(gen))
        # if np.shape(r)[-1]>1:
        #     print(e1,r,e2)
        # e2_multi1 = np.asarray((e2_multi1 * 0.2) + 1.0 / data_handler.tot_entity)
        # e2_multi2 = np.asarray((e2_multi2 * 0.2) + 1.0 / data_handler.tot_entity)
        # # print("")
        # print("e1:", e1)
        # print("r:", r)
        # print("e2_multi1:", e2_multi1)
        # print("e1:", e2)
        # print("r:", r_rev)
        # print("e2_multi1:", e2_multi2)


if __name__ == '__main__':
    # test_data_prep()
    test_data_prep_generator_hr_t()
