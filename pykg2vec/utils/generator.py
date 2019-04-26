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
from utils.dataprep import DataPrep, DataInput, DataStats
import numpy as np
from scipy import sparse as sps
from threading import Thread, currentThread
import pickle
from queue import Queue


class Generator(object):
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

        with open(self.config.data_path / 'data_stats.pkl', 'rb') as f:
            self.data_stats = pickle.load(f)

        # c extension to handle data
        if self.config.algo.lower().startswith('conve'):
            with open(self.config.data_path / 'train_data.pkl', 'rb') as f:
                self.train_data = pickle.load(f)
            with open(self.config.data_path / 'test_data.pkl', 'rb') as f:
                self.test_data = pickle.load(f)
            with open(self.config.data_path / 'valid_data.pkl', 'rb') as f:
                self.valid_data = pickle.load(f)
            self.rand_ids_train = np.random.permutation(len(self.train_data))
            self.rand_ids_test = np.random.permutation(len(self.test_data))
            self.rand_ids_valid = np.random.permutation(len(self.valid_data))

        elif self.config.algo.lower().startswith('proje'):
            with open(self.config.data_path / 'train_triples_ids.pkl', 'rb') as f:
                self.train_triples_ids = pickle.load(f)
            with open(self.config.data_path / 'test_triples_ids.pkl', 'rb') as f:
                self.test_triples_ids = pickle.load(f)
            with open(self.config.data_path / 'validation_triples_ids.pkl', 'rb') as f:
                self.valid_triples_ids = pickle.load(f)
            with open(self.config.data_path / 'hr_t_ids_train.pkl', 'rb') as f:
                self.hr_t_ids_train = pickle.load(f)
            with open(self.config.data_path / 'tr_h_ids_train.pkl', 'rb') as f:
                self.tr_h_ids_train = pickle.load(f)
            self.rand_ids_train = np.random.permutation(len(self.train_triples_ids))
            self.rand_ids_test = np.random.permutation(len(self.test_triples_ids))
            self.rand_ids_valid = np.random.permutation(len(self.valid_triples_ids))

        else:
            with open(self.config.data_path / 'train_triples_ids.pkl', 'rb') as f:
                self.train_triples_ids = pickle.load(f)
            with open(self.config.data_path / 'test_triples_ids.pkl', 'rb') as f:
                self.test_triples_ids = pickle.load(f)
            with open(self.config.data_path / 'validation_triples_ids.pkl', 'rb') as f:
                self.valid_triples_ids = pickle.load(f)
            self.rand_ids_train = np.random.permutation(len(self.train_triples_ids))
            self.rand_ids_test = np.random.permutation(len(self.test_triples_ids))
            self.rand_ids_valid = np.random.permutation(len(self.valid_triples_ids))
            self.lh = None
            self.lr = None
            self.lt = None
            self.observed_triples = None

        if self.config.sampling == "bern":
            with open(self.config.tmp_data / 'relation_property.pkl', 'rb') as f:
                self.relation_property = pickle.load(f)

        self.thread_list = []
        self.thread_cnt = 0

    def __iter__(self):
        if self.config.algo.lower().startswith('conve'):
            return self.gen_batch_conve()
        elif self.config.algo.lower().startswith('proje'):
            return self.gen_batch_proje()
        else:
            return self.gen_batch_trans()

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

        batch_idx = 0

        while True:
            for i in range(self.config.thread_num):
                if self.thread_cnt < self.config.thread_num:
                    if self.config.data.startswith('train'):
                        raw_data = np.asarray([[self.train_data[x].e1,
                                                self.train_data[x].r,
                                                self.train_data[x].e2_multi1] for x in
                                               self.rand_ids_train[bs * batch_idx: bs * (batch_idx + 1)]])
                    elif self.config.data.startswith('test'):
                        raw_data = np.asarray([[self.test_data[x].e1,
                                                self.test_data[x].r,
                                                self.test_data[x].e2_multi1,
                                                self.test_data[x].e2,
                                                self.test_data[x].r_rev,
                                                self.test_data[x].e2_multi2] for x in
                                               self.rand_ids_test[bs * batch_idx:bs * (batch_idx + 1)]])
                    elif self.config.data.startswith('valid'):
                        raw_data = np.asarray([[self.valid_data[x].e1,
                                                self.valid_data[x].r,
                                                self.valid_data[x].e2_multi1,
                                                self.valid_data[x].e2,
                                                self.valid_data[x].r_rev,
                                                self.valid_data[x].e2_multi2] for x in
                                               self.rand_ids_valid[bs * batch_idx:bs * (batch_idx + 1)]])
                    else:
                        raise NotImplementedError("The data type passed is wrong!")
                    if self.config.data.startswith('train'):
                        worker = Thread(target=self.process_one_train_batch_conve,
                                        args=(raw_data, bs, te,))
                    else:
                        worker = Thread(target=self.process_one_test_batch_conve,
                                        args=(raw_data, bs, te,))
                    # print("batch id:", batch_idx, " qL:", self.queue.qsize(),
                    # " theadID: ", worker.name)
                    self.thread_list.append(worker)
                    worker.setDaemon(True)
                    worker.start()
                    self.thread_cnt += 1
                    batch_idx += 1

                    if batch_idx >= number_of_batches:
                        # print("\n \t \t----Finished  Epoch-----------")
                        batch_idx = 0

            # print("2: getting one batch!", self.queue.qsize())
            data = self.queue.get()
            # print("2: removed one batch:", self.queue.qsize())
            # print("Total Threads:", self.thread_cnt)
            # print("Queue Size:",self.queue.qsize())
            yield data

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

        batch_idx = 0

        while True:
            for i in range(self.config.thread_num):
                if self.thread_cnt < self.config.thread_num:
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
                    # self.process_one_train_batch_proje(raw_data, n_entity, neg_weight)
                    if self.config.data.startswith('train'):
                        worker = Thread(target=self.process_one_train_batch_proje,
                                        args=(raw_data, n_entity, neg_weight,))
                    else:
                        worker = Thread(target=self.process_one_test_batch_proje,
                                        args=(raw_data,))
                    # print("batch id:", batch_idx, " qL:", self.queue.qsize(),
                    # " theadID: ", worker.name)
                    self.thread_list.append(worker)
                    worker.setDaemon(True)
                    worker.start()
                    self.thread_cnt += 1
                    batch_idx += 1

                    if batch_idx >= number_of_batches:
                        print("\n \t \t----Finished  Epoch-----------")
                        batch_idx = 0

            # print("2: getting one batch!", self.queue.qsize())
            # while self.queue.empty():
            #     pass
            data = self.queue.get()
            # print("2: removed one batch:", self.queue.qsize())
            # print("Total Threads:", self.thread_cnt)
            # print("Queue Size:", self.queue.qsize())
            yield data

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

        batch_idx = 0

        while True:
            for i in range(self.config.thread_num):
                if self.thread_cnt < self.config.thread_num:
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
                    if self.config.data.startswith('train'):
                        worker = Thread(target=self.process_one_train_batch_trans,
                                        args=(raw_data,))
                    else:
                        worker = Thread(target=self.process_one_test_batch_trans,
                                        args=(raw_data,))
                    # print("batch id:", batch_idx, " qL:", self.queue.qsize(),
                    # " theadID: ", worker.name)
                    self.thread_list.append(worker)
                    worker.setDaemon(True)
                    worker.start()
                    self.thread_cnt += 1
                    batch_idx += 1

                    if batch_idx >= number_of_batches:
                        # print("\n \t \t----Finished  Epoch-----------")
                        batch_idx = 0

            # print("2: getting one batch!", self.queue.qsize())
            data = self.queue.get()
            # print("2: removed one batch:", self.queue.qsize())
            # print("Total Threads:", self.thread_cnt)
            # print("Queue Size:",self.queue.qsize())
            yield data

    def process_one_train_batch_trans(self, pos_triples):
        te = self.data_stats.tot_entity
        bs = self.config.batch_size
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

        self.queue.put([ph, pr, pt, nh, nr, nt])
        self.thread_cnt -= 1

    def process_one_test_batch_trans(self, pos_triples):

        ph = pos_triples[:, 0]
        pr = pos_triples[:, 1]
        pt = pos_triples[:, 2]

        self.queue.put([ph, pr, pt])
        self.thread_cnt -= 1

    def process_one_train_batch_proje(self, raw_data, n_entity, neg_weight):
        bs = self.config.batch_size

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

        self.queue.put([np.asarray(hr_hr_batch, dtype=np.int32), np.asarray(hr_tweight, dtype=np.float32),
                        np.asarray(tr_tr_batch, dtype=np.int32), np.asarray(tr_hweight, dtype=np.float32)])
        # print(self.thread_cnt, self.queue.qsize())
        self.thread_cnt -= 1

    def process_one_test_batch_proje(self, raw_data):
        h = raw_data[:, 0]
        r = raw_data[:, 1]
        t = raw_data[:, 2]

        self.queue.put([h,r,t])
        self.thread_cnt -= 1

    def process_one_train_batch_conve(self, raw_data, bs, te):
        # read the batch
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
        self.queue.put([e1, r, np.array(e2_multi1.todense())])
        self.thread_cnt -= 1

    def process_one_test_batch_conve(self, raw_data, bs, te):
        # read the batch
        e1 = raw_data[:, 0]
        r = raw_data[:, 1]
        col = []
        for k in raw_data[:, 2]:
            col.append(k)
        row = []
        for k in range(bs):
            if col[k]:
                row.append([k] * len(col[k]))
        col_n = []
        row_n = []
        # TODO: Vectorize the loops
        for i in range(bs):
            if col[i]:
                for j in range(len(col[i])):
                    col_n.append(col[i][j])
                    row_n.append(row[i][j])

        e2_multi1 = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(bs, te))

        e2 = raw_data[:, 3]
        r_rev = raw_data[:, 4]
        col = []
        for k in raw_data[:, 5]:
            col.append(k)

        row = []
        for k in range(bs):
            if col[k]:
                row.append([k] * len(col[k]))
        col_n = []
        row_n = []
        # TODO: Vectorize the loops
        for i in range(bs):
            if col[i]:
                for j in range(len(col[i])):
                    col_n.append(col[i][j])
                    row_n.append(row[i][j])

        e2_multi2 = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(bs, te))

        self.queue.put([e1, r, np.array(e2_multi1.todense()), e2, r_rev, np.array(e2_multi2.todense())])
        self.thread_cnt -= 1


def test_generator_conve():
    gen = iter(Generator(config=GeneratorConfig(data='train', algo='ConvE')))
    for i in range(5000):
        data = list(next(gen))
        e1 = data[0]
        r = data[1]
        e2_multi1 = data[2]
        # e2 = data[3]
        # r_rev = data[4]
        # e2_multi2 = data[5]
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
    for i in range(100):
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


def test_generator_trans():
    import time
    gen = iter(Generator(config=GeneratorConfig(data='train', algo='TransE')))
    for i in range(5):
        data = list(next(gen))
        print("----batch:", i)
        ph = data[0]
        pr = data[1]
        pt = data[2]
        nh = data[3]
        nr = data[4]
        nt = data[5]
        print("ph:", ph)
        print("pr:", pr)
        print("pt:", pt)
        print("nh:", nh)
        print("nr:", nr)
        print("nt:", nt)


if __name__ == '__main__':
    # test_generator_proje()
    # test_generator_conve()
    test_generator_trans()
