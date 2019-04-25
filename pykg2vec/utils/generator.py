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
from utils.dataprep import DataPrep, DataInput
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

    def __init__(self, config=None, data_handler=None):
        if not config:
            self.config = GeneratorConfig()
        else:
            self.config = config
        if not data_handler:
            self.data_handler = DataPrep('Freebase15k',
                                         algo=True if self.config.loss_type.startswith("entropy") else False)
        else:
            self.data_handler = data_handler

        self.queue = Queue(self.config.queue_size)
        # c extension to handle data
        if self.config.loss_type == 'entropy':
            with open(self.config.data_path / 'train_data.pkl', 'rb') as f:
                self.train_data = pickle.load(f)
            with open(self.config.data_path / 'test_data.pkl', 'rb') as f:
                self.test_data = pickle.load(f)
            with open(self.config.data_path / 'valid_data.pkl', 'rb') as f:
                self.valid_data = pickle.load(f)
            self.rand_ids_train = np.random.permutation(len(self.train_data))
            self.rand_ids_test = np.random.permutation(len(self.train_data))
            self.rand_ids_valid = np.random.permutation(len(self.train_data))
        self.thread_list = []
        self.thread_cnt = 0

    def __iter__(self):
        return self.gen_train_batch_entropy()

    def gen_train_batch_entropy(self):
        bs = self.config.batch_size
        te = self.data_handler.tot_entity
        number_of_batches = len(self.train_data) // bs
        print("Number_of_batches:", number_of_batches)

        batch_idx = 0

        while True:
            if not self.queue.full():
                threads = []
                for i in range(self.config.thread_num):
                    if self.thread_cnt < self.config.thread_num:
                        raw_data = np.asarray([[self.train_data[x].e1,
                                                self.train_data[x].r,
                                                self.train_data[x].e2_multi1] for x in
                                               self.rand_ids_train[bs * batch_idx: bs * (batch_idx + 1)]])
                        worker = Thread(target=self.process_one_train_batch_entropy, args=(raw_data, bs, te,))
                        # print("batch id:", batch_idx, " qL:", self.queue.qsize(), " theadID: ", worker.name)
                        self.thread_list.append(worker)
                        worker.setDaemon(True)
                        worker.start()
                        threads.append(worker)
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
            yield data[0], data[1], data[2]

    def process_one_train_batch_entropy(self, raw_data, bs, te):
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
        for i in range(bs):
            for j in range(len(col[i])):
                col_n.append(col[i][j])
                row_n.append(row[i][j])

        e2_multi1 = sps.csr_matrix(([1] * len(row_n), (row_n, col_n)), shape=(bs, te))
        self.queue.put([e1, r, np.array(e2_multi1.todense())])
        self.thread_cnt -= 1


def test_generator():
    gen = iter(Generator())
    for i in range(5000):
        e1, r, e2_multi1 = list(next(gen))
        print("\n----batch:", i)
        print("e1:", e1)
        print("r:", r)
        print("e2_multi1:", e2_multi1)


if __name__ == '__main__':
    test_generator()
