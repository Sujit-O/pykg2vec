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
from utils.dataprep import DataPrep, DataInput, DataStats, DataInputSimple
import numpy as np
from scipy import sparse as sps
from threading import Thread, currentThread
import pickle
from multiprocessing import Process, Queue, JoinableQueue
from numba import jit

def gen_id(ids):
    i = 0
    while True:
        yield ids[i]
        i += 1
        if i >= len(ids):
            np.random.shuffle(ids)
            i = 0


def get_sparse_mat(data, bs, te):
    mat = np.zeros(shape=(bs, te), dtype=np.int16)
    for i in range(bs):
        for j in range(len(data[i])):
            mat[i][data[i][j]] = 1
    return mat

class Generator:
    """Generator class for the embedding algorithms
        Args:
          config: generator configuration
        Returns:
          batch for training algorithms
    """

    def __init__(self, config=None):

        if not config:
            self.config = GeneratorConfig()
        else:
            self.config = config

        self.queue = Queue(self.config.queue_size)
        self.raw_queue = Queue(self.config.raw_queue_size)
        self.processed_queue = Queue(self.config.processed_queue_size)
        self.process_list = []
        self.tot_train_data = None
        self.tot_test_data = None
        self.tot_valid_data = None

        with open(str(self.config.data_path / 'data_stats.pkl'), 'rb') as f:
            self.data_stats = pickle.load(f)

        if self.config.algo.lower() in ["tucker"]:
            with open(str(self.config.data_path / 'train_data.pkl'), 'rb') as f:
                self.train_data = pickle.load(f)
                self.tot_train_data = len(self.train_data)
            with open(str(self.config.data_path / 'test_data.pkl'), 'rb') as f:
                self.test_data = pickle.load(f)
                self.tot_test_data = len(self.test_data)
            with open(str(self.config.data_path / 'valid_data.pkl'), 'rb') as f:
                self.valid_data = pickle.load(f)
                self.tot_valid_data = len(self.valid_data)
            self.rand_ids_train = np.random.permutation(len(self.train_data))
            self.rand_ids_test = np.random.permutation(len(self.test_data))
            self.rand_ids_valid = np.random.permutation(len(self.valid_data))

            self.gen_batch_simple()

        elif self.config.algo.lower() in ["conve", "complex", "distmult"]:
            with open(str(self.config.data_path / 'train_data.pkl'), 'rb') as f:
                self.train_data = pickle.load(f)
                self.tot_train_data = len(self.train_data)
            with open(str(self.config.data_path / 'test_data.pkl'), 'rb') as f:
                self.test_data = pickle.load(f)
                self.tot_test_data = len(self.test_data)
            with open(str(self.config.data_path / 'valid_data.pkl'), 'rb') as f:
                self.valid_data = pickle.load(f)
                self.tot_valid_data = len(self.valid_data)
            self.rand_ids_train = np.random.permutation(len(self.train_data))
            self.rand_ids_test = np.random.permutation(len(self.test_data))
            self.rand_ids_valid = np.random.permutation(len(self.valid_data))
            self.gen_batch_conve()

        elif self.config.algo.lower().startswith('proje'):
            with open(str(self.config.data_path / 'train_triples_ids.pkl'), 'rb') as f:
                self.train_triples_ids = pickle.load(f)
                self.tot_train_data = len(self.train_triples_ids)
            with open(str(self.config.data_path / 'test_triples_ids.pkl'), 'rb') as f:
                self.test_triples_ids = pickle.load(f)
                self.tot_test_data = len(self.test_triples_ids)
            with open(str(self.config.data_path / 'validation_triples_ids.pkl'), 'rb') as f:
                self.valid_triples_ids = pickle.load(f)
                self.tot_valid_data = len(self.valid_triples_ids)
            with open(str(self.config.data_path / 'hr_t_ids_train.pkl'), 'rb') as f:
                self.hr_t_ids_train = pickle.load(f)
            with open(str(self.config.data_path / 'tr_h_ids_train.pkl'), 'rb') as f:
                self.tr_h_ids_train = pickle.load(f)
            self.rand_ids_train = np.random.permutation(len(self.train_triples_ids))
            self.rand_ids_test = np.random.permutation(len(self.test_triples_ids))
            self.rand_ids_valid = np.random.permutation(len(self.valid_triples_ids))
            self.gen_batch_proje()

        else:
            with open(str(self.config.data_path / 'train_triples_ids.pkl'), 'rb') as f:
                self.train_triples_ids = pickle.load(f)
                self.tot_train_data = len(self.train_triples_ids)
            with open(str(self.config.data_path / 'test_triples_ids.pkl'), 'rb') as f:
                self.test_triples_ids = pickle.load(f)
                self.tot_test_data = len(self.test_triples_ids)
            with open(str(self.config.data_path / 'validation_triples_ids.pkl'), 'rb') as f:
                self.valid_triples_ids = pickle.load(f)
                self.tot_valid_data = len(self.valid_triples_ids)
            self.rand_ids_train = np.random.permutation(len(self.train_triples_ids))
            self.rand_ids_test = np.random.permutation(len(self.test_triples_ids))
            self.rand_ids_valid = np.random.permutation(len(self.valid_triples_ids))
            self.lh = None
            self.lr = None
            self.lt = None
            self.observed_triples = None

            self.gen_batch_trans()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            return self.processed_queue.get()

    def stop(self):
        for p in self.process_list:
            p.terminate()

    def gen_batch_simple(self):
        bs = self.config.batch_size
        te = self.data_stats.tot_entity
        if self.config.data.startswith('train'):
            number_of_batches = len(self.train_data) // bs
        elif self.config.data.startswith('test'):
            number_of_batches = len(self.test_data) // bs
        elif self.config.data.startswith('valid'):
            number_of_batches = len(self.valid_data) // bs
        else:
            raise NotImplementedError("The data type passed is wrong!")
        print("Number_of_batches:", number_of_batches)

        ids = np.random.permutation(number_of_batches)

        worker = Process(target=self.raw_data_generator_simple, args=(ids,))
        worker.daemon = True
        self.process_list.append(worker)
        worker.start()
        self.pool_process_simple()

    def gen_batch_proje(self, n_entity=None, neg_weight=0.5):
        bs = self.config.batch_size
        if not n_entity:
            n_entity = self.data_stats.tot_entity
        if self.config.data.startswith('train'):
            number_of_batches = len(self.train_triples_ids) // bs
        elif self.config.data.startswith('test'):
            number_of_batches = len(self.test_triples_ids) // bs
        elif self.config.data.startswith('valid'):
            number_of_batches = len(self.valid_triples_ids) // bs
        else:
            raise NotImplementedError("The data type passed is wrong!")
        print("Number_of_batches:", number_of_batches)

        ids = np.random.permutation(number_of_batches)

        worker = Process(target=self.raw_data_generator_proje, args=(ids,))
        worker.daemon = True
        self.process_list.append(worker)
        worker.start()
        self.pool_process_proje(bs=self.config.batch_size, n_entity=n_entity, neg_weight=neg_weight)

    def gen_batch_conve(self):
        bs = self.config.batch_size
        te = self.data_stats.tot_entity
        if self.config.data.startswith('train'):
            number_of_batches = len(self.train_data) // bs
        elif self.config.data.startswith('test'):
            number_of_batches = len(self.test_data) // bs
        elif self.config.data.startswith('valid'):
            number_of_batches = len(self.valid_data) // bs
        else:
            raise NotImplementedError("The data type passed is wrong!")
        print("Number_of_batches:", number_of_batches)

        ids = np.random.permutation(number_of_batches)

        worker = Process(target=self.raw_data_generator_conve, args=(ids,))
        worker.daemon = True
        self.process_list.append(worker)
        worker.start()
        self.pool_process_conve()

    def gen_batch_trans(self):
        bs = self.config.batch_size
        te = self.data_stats.tot_entity
        if self.config.data.startswith('train'):
            number_of_batches = len(self.train_triples_ids) // bs
            self.observed_triples = {(t.h, t.r, t.t): 1 for t in self.train_triples_ids}
        elif self.config.data.startswith('test'):
            number_of_batches = len(self.test_triples_ids) // bs
            self.observed_triples = {(t.h, t.r, t.t): 1 for t in self.test_triples_ids}
        elif self.config.data.startswith('valid'):
            number_of_batches = len(self.valid_triples_ids) // bs
            self.observed_triples = {(t.h, t.r, t.t): 1 for t in self.valid_triples_ids}
        else:
            raise NotImplementedError("The data type passed is wrong!")
        print("Number_of_batches:", number_of_batches)

        ids = np.random.permutation(number_of_batches)

        worker = Process(target=self.raw_data_generator_trans, args=(ids,))
        worker.daemon = True
        self.process_list.append(worker)
        worker.start()
        self.pool_process_trans()

    def raw_data_generator_simple(self, ids):
        gen = iter(gen_id(ids))
        bs = self.config.batch_size
        while True:
            batch_idx = next(gen)
            raw_data = np.asarray([[self.train_data[x].h,
                                    self.train_data[x].r,
                                    self.train_data[x].t,
                                    self.train_data[x].hr_t
                                    ] for x in
                                   self.rand_ids_train[bs * batch_idx: bs * (batch_idx + 1)]])

            self.raw_queue.put(raw_data)

    def raw_data_generator_proje(self, ids):
        gen = iter(gen_id(ids))
        bs = self.config.batch_size
        while True:
            batch_idx = next(gen)
            if self.config.data.startswith('train'):
                raw_data = np.asarray([[self.train_triples_ids[x].h,
                                        self.train_triples_ids[x].r,
                                        self.train_triples_ids[x].t] for x in
                                       self.rand_ids_train[bs * batch_idx:bs * (batch_idx + 1)]])
            elif self.config.data.startswith('test'):
                raw_data = np.asarray([[self.test_triples_ids[x].h,
                                        self.test_triples_ids[x].r,
                                        self.test_triples_ids[x].t] for x in
                                       self.rand_ids_test[bs * batch_idx:bs * (batch_idx + 1)]])
            elif self.config.data.startswith('valid'):
                raw_data = np.asarray([[self.valid_triples_ids[x].h,
                                        self.valid_triples_ids[x].r,
                                        self.valid_triples_ids[x].t] for x in
                                       self.rand_ids_valid[bs * batch_idx:bs * (batch_idx + 1)]])
            else:
                raise NotImplementedError("The data type passed is wrong!")

            self.raw_queue.put(raw_data)
            # print("raw_producer", thread.name, self.raw_queue.qsize())

    def raw_data_generator_conve(self, ids):
        gen = iter(gen_id(ids))
        bs = self.config.batch_size
        while True:
            batch_idx = next(gen)
            if self.config.data.startswith('train'):
                raw_data = np.asarray([[self.train_data[x].e1,
                                        self.train_data[x].r,
                                        self.train_data[x].e2_multi1] for x in
                                       self.rand_ids_train[bs * batch_idx: bs * (batch_idx + 1)]])
            elif self.config.data.startswith('test'):
                raw_data = np.asarray([[self.test_data[x].e1,
                                        self.test_data[x].r,
                                        # self.test_data[x].e2_multi1,
                                        self.test_data[x].e2,
                                        self.test_data[x].r_rev
                                        # self.test_data[x].e2_multi2
                                        ] for x in
                                       self.rand_ids_test[bs * batch_idx:bs * (batch_idx + 1)]])
            elif self.config.data.startswith('valid'):
                raw_data = np.asarray([[self.valid_data[x].e1,
                                        self.valid_data[x].r,
                                        # self.valid_data[x].e2_multi1,
                                        self.valid_data[x].e2,
                                        self.valid_data[x].r_rev
                                        # self.valid_data[x].e2_multi2
                                        ] for x in
                                       self.rand_ids_valid[bs * batch_idx:bs * (batch_idx + 1)]])
            else:
                raise NotImplementedError("The data type passed is wrong!")

            self.raw_queue.put(raw_data)

    def raw_data_generator_trans(self, ids):
        gen = iter(gen_id(ids))
        bs = self.config.batch_size
        while True:
            batch_idx = next(gen)
            if self.config.data.startswith('train'):
                raw_data = np.asarray([[self.train_triples_ids[x].h,
                                        self.train_triples_ids[x].r,
                                        self.train_triples_ids[x].t] for x in
                                       self.rand_ids_train[bs * batch_idx:bs * (batch_idx + 1)]])
            elif self.config.data.startswith('test'):
                raw_data = np.asarray([[self.test_triples_ids[x].h,
                                        self.test_triples_ids[x].r,
                                        self.test_triples_ids[x].t] for x in
                                       self.rand_ids_test[bs * batch_idx:bs * (batch_idx + 1)]])
            elif self.config.data.startswith('valid'):
                raw_data = np.asarray([[self.valid_triples_ids[x].h,
                                        self.valid_triples_ids[x].r,
                                        self.valid_triples_ids[x].t] for x in
                                       self.rand_ids_valid[bs * batch_idx:bs * (batch_idx + 1)]])
            else:
                raise NotImplementedError("The data type passed is wrong!")

            self.raw_queue.put(raw_data)

    def pool_process_proje(self, bs=None, n_entity=None, neg_weight=None):
        for i in range(self.config.process_num):
            if self.config.data.startswith('train'):
                p = Process(target=self.process_function_train_proje, args=(bs, n_entity, neg_weight,))
            else:
                p = Process(target=self.process_function_test_proje, args=(bs, n_entity, neg_weight,))
            self.process_list.append(p)
            p.daemon = True
            p.start()

    def pool_process_simple(self):
        for i in range(self.config.process_num):
            if self.config.data.startswith('train'):
                p = Process(target=self.process_function_train_simple, args=())
            else:
                p = Process(target=self.process_function_test_simple, args=())
            self.process_list.append(p)
            p.daemon = True
            p.start()

    def pool_process_conve(self):
        for i in range(self.config.process_num):
            if self.config.data.startswith('train'):
                p = Process(target=self.process_function_train_conve, args=())
            else:
                p = Process(target=self.process_function_test_conve, args=())
            self.process_list.append(p)
            p.daemon = True
            p.start()

    def pool_process_trans(self):
        for i in range(self.config.process_num):
            if self.config.data.startswith('train'):
                p = Process(target=self.process_function_train_trans, args=())
            else:
                p = Process(target=self.process_function_test_trans, args=())
            self.process_list.append(p)
            p.daemon = True
            p.start()

    # @jit(nopython=True, parallel=True)

    def process_function_train_simple(self):
        bs = self.config.batch_size
        te = self.data_stats.tot_entity
        while True:
            raw_data = self.raw_queue.get()
            h = raw_data[:, 0]
            r = raw_data[:, 1]
            t = raw_data[:, 2]
            hr_t = get_sparse_mat(raw_data[:,3], bs, te)
            # col = []
            # for k in raw_data[:, 3]:
            #     col.append(k)
            # row = []
            # for k in range(bs):
            #     row.append([k] * len(col[k]))
            # col_n = []
            # row_n = []
            # # TODO: Vectorize the loops
            # for i in range(bs):
            #     for j in range(len(col[i])):
            #         col_n.append(col[i][j])
            #         row_n.append(row[i][j])
            #
            # hr_t = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(bs, te))

            # col = []
            # for k in raw_data[:, 4]:
            #     col.append(k)
            # row = []
            # for k in range(bs):
            #     row.append([k] * len(col[k]))
            # col_n = []
            # row_n = []
            # # TODO: Vectorize the loops
            # for i in range(bs):
            #     for j in range(len(col[i])):
            #         col_n.append(col[i][j])
            #         row_n.append(row[i][j])
            #
            # rt_h = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(bs, te))
            # rt_h = self.get_sparse_mat(raw_data[:, 4], bs, te)
            self.processed_queue.put([h, r, t, hr_t])

    def process_function_test_simple(self):
        while True:
            raw_data = self.raw_queue.get()
            h = raw_data[:, 0]
            r = raw_data[:, 1]
            t = raw_data[:, 2]
            self.processed_queue.put([h, r, t])

    def process_function_train_trans(self):
        te = self.data_stats.tot_entity
        bs = self.config.batch_size
        while True:
            pos_triples = self.raw_queue.get()
            ph = pos_triples[:, 0]
            pr = pos_triples[:, 1]
            pt = pos_triples[:, 2]
            nh = []
            nr = []
            nt = []

            for t in pos_triples:
                if self.config.sampling == 'uniform':
                    prob = 0.5
                elif self.config.sampling == 'bern':
                    prob = self.relation_property[t[1]]
                else:
                    raise NotImplementedError("%s sampling not supported!" % self.config.negative_sample)

                if np.random.random() > prob:
                    idx_replace_tail = np.random.randint(te)

                    break_cnt = 0
                    while (t[0], t[1], idx_replace_tail) in self.observed_triples:
                        idx_replace_tail = np.random.randint(te)
                        break_cnt += 1
                        if break_cnt >= 100:
                            break

                    if break_cnt >= 100:  # can not find new negative triple.
                        nh.append(self.lh)
                        nr.append(self.lr)
                        nt.append(self.lt)
                    else:
                        nh.append(t[0])
                        nr.append(t[1])
                        nt.append(idx_replace_tail)
                        self.lh = t[0]
                        self.lr = t[1]
                        self.lt = idx_replace_tail

                        self.observed_triples[(t[0], t[1], idx_replace_tail)] = 0

                else:
                    idx_replace_head = np.random.randint(te)
                    break_cnt = 0
                    while ((idx_replace_head, t[1], t[2]) in self.observed_triples
                           or (idx_replace_head, t[1], t[2]) in self.observed_triples):
                        idx_replace_head = np.random.randint(te)
                        break_cnt += 1
                        if break_cnt >= 100:
                            break

                    if break_cnt >= 100:  # can not find new negative triple.
                        nh.append(self.lh)
                        nr.append(self.lr)
                        nt.append(self.lt)
                    else:
                        nh.append(idx_replace_head)
                        nr.append(t[1])
                        nt.append(t[2])
                        self.lh = idx_replace_head
                        self.lr = t[1]
                        self.lt = t[2]

                        self.observed_triples[(idx_replace_head, t[1], t[2])] = 0

            self.processed_queue.put([ph, pr, pt, nh, nr, nt])

    def process_function_test_trans(self):
        while True:
            pos_triples = self.raw_queue.get()
            ph = pos_triples[:, 0]
            pr = pos_triples[:, 1]
            pt = pos_triples[:, 2]

            self.processed_queue.put([ph, pr, pt])

    def process_function_train_conve(self):
        bs = self.config.batch_size
        te = self.data_stats.tot_entity
        while True:
            raw_data = self.raw_queue.get()
            e1 = raw_data[:, 0]
            r = raw_data[:, 1]
            col = []
            for k in raw_data[:, 2]:
                col.append(k)
            row = []
            for k in range(bs):
                row.append([k] * len(col[k]))
            col_n = []
            row_n = []
            # TODO: Vectorize the loops
            for i in range(bs):
                for j in range(len(col[i])):
                    col_n.append(col[i][j])
                    row_n.append(row[i][j])

            e2_multi1 = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(bs, te))
            self.processed_queue.put([e1, r, np.array(e2_multi1.todense())])

    def process_function_test_conve(self):
        # read the batch
        bs = self.config.batch_size
        te = self.data_stats.tot_entity
        while True:
            raw_data = self.raw_queue.get()
            e1 = raw_data[:, 0]
            r = raw_data[:, 1]
            # col = []
            # for k in raw_data[:, 2]:
            #     col.append(k)
            # row = []
            # for k in range(bs):
            #     if col[k]:
            #         row.append([k] * len(col[k]))
            # col_n = []
            # row_n = []
            # # TODO: Vectorize the loops
            # for i in range(bs):
            #     if col[i]:
            #         for j in range(len(col[i])):
            #             col_n.append(col[i][j])
            #             row_n.append(row[i][j])
            #
            # e2_multi1 = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(bs, te))

            e2 = raw_data[:, 2]
            r_rev = raw_data[:, 3]
            # col = []
            # for k in raw_data[:, 5]:
            #     col.append(k)
            #
            # row = []
            # for k in range(bs):
            #     if col[k]:
            #         row.append([k] * len(col[k]))
            # col_n = []
            # row_n = []
            # # TODO: Vectorize the loops
            # for i in range(bs):
            #     if col[i]:
            #         for j in range(len(col[i])):
            #             col_n.append(col[i][j])
            #             row_n.append(row[i][j])
            #
            # e2_multi2 = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(bs, te))

            # self.processed_queue.put([e1, r, np.array(e2_multi1.todense()), e2, r_rev, np.array(e2_multi2.todense())])
            self.processed_queue.put([e1, r, e2, r_rev])

    def process_function_train_proje(self, bs, n_entity, neg_weight):
        while True:
            raw_data = self.raw_queue.get()
            if raw_data is None:
                break
            h = raw_data[:, 0]
            r = raw_data[:, 1]
            t = raw_data[:, 2]

            hr_hr_batch = list()
            hr_tweight = list()
            tr_tr_batch = list()
            tr_hweight = list()

            for idx in range(bs):
                if np.random.uniform(-1, 1) > 0:  # t r predict h
                    temp = np.zeros(n_entity)
                    for idx2 in np.random.permutation(n_entity)[0:n_entity // 2]:
                        temp[idx2] = -1.0
                    for head in self.tr_h_ids_train[(t[idx], r[idx])]:
                        temp[head] = 1.0
                    tr_hweight.append(temp)
                    # tr_hweight.append(
                    #     [1. if x in self.tr_h_ids_train[(r[idx],t[idx])] else y for
                    #      x, y in enumerate(np.random.choice([0., -1.], size=n_entity, p=[1 - neg_weight, neg_weight]))])
                    tr_tr_batch.append((t[idx], r[idx]))
                else:  # h r predict t
                    temp = np.zeros(n_entity)
                    for idx2 in np.random.permutation(n_entity)[0:n_entity // 2]:
                        temp[idx2] = -1.0
                    for tail in self.hr_t_ids_train[(h[idx], r[idx])]:
                        temp[tail] = 1.0
                    hr_tweight.append(temp)
                    # hr_tweight.append(
                    #     [1. if x in self.hr_t_ids_train[(h[idx], t[idx])] else y for
                    #      x, y in enumerate(np.random.choice([0., -1.], size=n_entity, p=[1 - neg_weight, neg_weight]))])
                    hr_hr_batch.append((h[idx], r[idx]))

            self.processed_queue.put([np.asarray(hr_hr_batch, dtype=np.int32), np.asarray(hr_tweight, dtype=np.float32),
                                      np.asarray(tr_tr_batch, dtype=np.int32),
                                      np.asarray(tr_hweight, dtype=np.float32)])

    def process_function_test_proje(self):
        while True:
            raw_data = self.raw_queue.get()
            if raw_data is None:
                break
            h = raw_data[:, 0]
            r = raw_data[:, 1]
            t = raw_data[:, 2]

            self.processed_queue.put([h, r, t])


def test_generator_conve():
    gen = iter(Generator(config=GeneratorConfig(data='test', algo='ConvE')))
    for i in range(5000):
        data = list(next(gen))
        e1 = data[0]
        r = data[1]
        e2_multi1 = data[2]
        e2 = data[3]
        r_rev = data[4]
        e2_multi2 = data[5]
        print("----batch:", i)
        # print("e1:", e1)
        # print("r:", r)
        # print("e2_multi1:", e2_multi1)
        # print("e2:", e2)
        # print("r_rev:", r_rev)
        # print("e2_multi2:", e2_multi2)


def test_generator_proje():
    import time
    gen = iter(Generator(config=GeneratorConfig(data='train', algo='ProjE')))
    for i in range(1000):
        data = list(next(gen))
        print("----batch:", i)
        hr_hr = data[0]
        hr_t = data[1]
        tr_tr = data[2]
        tr_h = data[3]
        # time.sleep(0.05)
        # print("hr_hr:", hr_hr)
        # print("hr_t:", hr_t)
        # print("tr_tr:", tr_tr)
        # print("tr_h:", tr_h)
    # gen.stop()


def test_generator_trans():
    import time
    gen = Generator(config=GeneratorConfig(data='test', algo='TransE'))
    for i in range(1000):
        data = list(next(gen))
        print("----batch:", i)
        ph = data[0]
        pr = data[1]
        pt = data[2]
        # nh = data[3]
        # nr = data[4]
        # nt = data[5]
        # print("ph:", ph)
        # print("pr:", pr)
        # print("pt:", pt)
        # print("nh:", nh)
        # print("nr:", nr)
        # print("nt:", nt)
    gen.stop()


if __name__ == '__main__':
    # test_generator_proje()
    # test_generator_conve()
    test_generator_trans()
