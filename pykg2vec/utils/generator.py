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
import pickle


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
        if not config:
            raise NotImplementedError('No configuration found for Generator!')
        self.Queue = Queue(self.config.queue_size)
        # c extension to handle data
        self.read_data = fh.read_data
        if self.config.loss_type == 'entropy':
            with open(self.config.tmp_data / 'train_data.pkl', 'rb') as f:
                self.train_data = pickle.load(f)
            with open(self.config.tmp_data / 'test_data.pkl', 'rb') as f:
                self.test_data = pickle.load(f)
            with open(self.config.tmp_data / 'valid_data.pkl', 'rb') as f:
                self.valid_data = pickle.load(f)

    def __iter__(self):
        return self

    def process_data(self):
        pass

    def process_one_train_batch(self):
        # read the batch
        self.mem_lock.acquire()
        batch_data = self.read_data(self.current_batch_idx,
                                    self.config.batch_size,
                                    self.config.total_entity,
                                    self.config.data_path)

        # update the current_batch size for the next process

        self.current_batch_idx += 1
        if self.current_batch_idx >= self.total_batch:
            self.current_batch_idx = 0
        self.mem_lock.release()

        self.queue_lock.acquire()
        # TODO: Store the batch in the queue
        self.queue_lock.release()

    def next(self):
        pass


def test_generator():
    pass


if __name__ == '__main__':
    test_generator()
