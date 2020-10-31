#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for generating the batch data for training and testing.
"""
import torch
import gc
import numpy as np
from multiprocessing import Process, Queue
from pykg2vec.common import TrainingStrategy

# def raw_data_generator(command_queue, raw_queue, config):
#     """Function to feed  triples to raw queue for multiprocessing.
#
#         Args:
#             raw_queue (Queue) : Multiprocessing Queue to put the raw data to be processed.
#             data (list) : List of integer ids denoting positive triples.
#             batch_size (int) : Size of each batch.
#             number_of_batch (int) : Total number of batch.
#
#     """
#     data = config.knowledge_graph.read_cache_data('triplets_train')
#
#     number_of_batch = len(data) // config.batch_size
#
#     random_ids = np.random.permutation(len(data))
#
#     while True:
#
#         command = command_queue.get()
#
#         if command != "quit":
#             number_of_batch = command
#             for batch_idx in range(number_of_batch):
#                 pos_start = config.batch_size * batch_idx
#                 pos_end = config.batch_size * (batch_idx + 1)
#                 raw_data = np.asarray([[data[x].h, data[x].r, data[x].t] for x in random_ids[pos_start:pos_end]])
#                 raw_queue.put((batch_idx, raw_data))
#         else:
#             raw_queue.put(None)
#             raw_queue.put(None)
#             return

#
# def process_function_pairwise(raw_queue, processed_queue, config):
#     """Function that puts the processed data in the queue.
#
#         Args:
#             raw_queue (Queue) : Multiprocessing Queue to put the raw data to be processed.
#             processed_queue (Queue) : Multiprocessing Queue to put the processed data.
#             te (int): Total number of entities
#             bs (int): Total size of each batch.
#             positive_triplets (list) : List of positive triples.
#             lh (int): Id of the last processed head.
#             lr (int): Id of the last processed relation.
#             lt (int): Id of the last processed tail.
#     """
#     data = config.knowledge_graph.read_cache_data('triplets_train')
#     relation_property = config.knowledge_graph.read_cache_data('relationproperty')
#     positive_triplets = {(t.h, t.r, t.t): 1 for t in data}
#     neg_rate = config.neg_rate
#
#     del data # save memory space
#
#     while True:
#         item = raw_queue.get()
#         if item is None:
#             return
#         _, pos_triples = item
#
#         ph = pos_triples[:, 0]
#         pr = pos_triples[:, 1]
#         pt = pos_triples[:, 2]
#
#         nh = []
#         nr = []
#         nt = []
#
#         for t in pos_triples:
#
#             prob = relation_property[t[1]] if config.sampling == "bern" else 0.5
#
#             for _ in range(neg_rate):
#
#                 if np.random.random() > prob:
#                     idx_replace_tail = np.random.randint(config.tot_entity)
#
#                     while (t[0], t[1], idx_replace_tail) in positive_triplets:
#                         idx_replace_tail = np.random.randint(config.tot_entity)
#
#                     nh.append(t[0])
#                     nr.append(t[1])
#                     nt.append(idx_replace_tail)
#
#                 else:
#                     idx_replace_head = np.random.randint(config.tot_entity)
#
#                     while (idx_replace_head, t[1], t[2]) in positive_triplets:
#                         idx_replace_head = np.random.randint(config.tot_entity)
#
#                     nh.append(idx_replace_head)
#                     nr.append(t[1])
#                     nt.append(t[2])
#
#         processed_queue.put([ph, pr, pt, nh, nr, nt])
#
# def process_function_pointwise(raw_queue, processed_queue, config):
#     """Function that puts the processed data in the queue.
#
#         Args:
#             raw_queue (Queue) : Multiprocessing Queue to put the raw data to be processed.
#             processed_queue (Queue) : Multiprocessing Queue to put the processed data.
#             te (int): Total number of entities
#             bs (int): Total size of each batch.
#             positive_triplets (list) : List of positive triples.
#             lh (int): Id of the last processed head.
#             lr (int): Id of the last processed relation.
#             lt (int): Id of the last processed tail.
#     """
#     data = config.knowledge_graph.read_cache_data('triplets_train')
#     relation_property = config.knowledge_graph.read_cache_data('relationproperty')
#     positive_triplets = {(t.h, t.r, t.t): 1 for t in data}
#     neg_rate = config.neg_rate
#
#     del data # save memory space
#
#     while True:
#         item = raw_queue.get()
#         if item is None:
#             return
#         _, pos_triples = item
#
#         point_h = []
#         point_r = []
#         point_t = []
#         point_y = []
#
#         for t in pos_triples:
#             # postive sample
#             point_h.append(t[0])
#             point_r.append(t[1])
#             point_t.append(t[2])
#             point_y.append(1)
#
#             prob = relation_property[t[1]] if config.sampling == "bern" else 0.5
#
#             for _ in range(neg_rate):
#
#                 if np.random.random() > prob:
#                     idx_replace_tail = np.random.randint(config.tot_entity)
#
#                     while (t[0], t[1], idx_replace_tail) in positive_triplets:
#                         idx_replace_tail = np.random.randint(config.tot_entity)
#
#                     point_h.append(t[0])
#                     point_r.append(t[1])
#                     point_t.append(idx_replace_tail)
#                     point_y.append(-1)
#
#                 else:
#                     idx_replace_head = np.random.randint(config.tot_entity)
#
#                     while (idx_replace_head, t[1], t[2]) in positive_triplets:
#                         idx_replace_head = np.random.randint(config.tot_entity)
#
#                     point_h.append(idx_replace_head)
#                     point_r.append(t[1])
#                     point_t.append(t[2])
#                     point_y.append(-1)
#
#         processed_queue.put([point_h, point_r, point_t, point_y])
#
#
# def process_function_multiclass(raw_queue, processed_queue, config):
#     """Function that puts the processed data in the queue.
#
#         Args:
#             raw_queue (Queue) : Multiprocessing Queue to put the raw data to be processed.
#             processed_queue (Queue) : Multiprocessing Queue to put the processed data.
#             te (int): Total number of entities
#             bs (int): Total size of each batch.
#             neg_rate (int): Ratio of negative to positive samples.
#     """
#
#     def _to_sparse_i(indices):
#         x = []
#         y = []
#         for index in indices:
#             x.append(index[0])
#             y.append(index[1])
#         return [x, y]
#
#     hr_t_train = config.knowledge_graph.read_cache_data('hr_t_train')
#     tr_h_train = config.knowledge_graph.read_cache_data('tr_h_train')
#
#     neg_rate = config.neg_rate
#
#     shape = [config.batch_size, config.tot_entity]
#
#     while True:
#         item = raw_queue.get()
#         if item is None:
#             return
#         idx, raw_data = item
#
#         h = raw_data[:, 0]
#         r = raw_data[:, 1]
#         t = raw_data[:, 2]
#
#         indices_hr_t = []
#         indices_tr_h = []
#         neg_indices_hr_t = []
#         neg_indices_tr_h = []
#
#         random_ids = np.random.permutation(config.tot_entity)
#
#         for i in range(config.batch_size):
#             hr_t = hr_t_train[(h[i], r[i])]
#             tr_h = tr_h_train[(t[i], r[i])]
#
#             for idx in hr_t:
#                 indices_hr_t.append([i, idx])
#             for idx in tr_h:
#                 indices_tr_h.append([i, idx])
#
#             if neg_rate > 0:
#                 for idx in random_ids[0:100]:
#                     if idx not in hr_t:
#                         neg_indices_hr_t.append([i, idx])
#                 for idx in random_ids[0:100]:
#                     if idx not in tr_h:
#                         neg_indices_tr_h.append([i, idx])
#
#
#         values_hr_t = torch.FloatTensor([1]).repeat([len(indices_hr_t)])
#         values_tr_h = torch.FloatTensor([1]).repeat([len(indices_tr_h)])
#
#         if neg_rate > 0:
#             neg_values_hr_t = torch.FloatTensor([-1]).repeat([len(neg_indices_hr_t)])
#             neg_values_tr_h = torch.FloatTensor([-1]).repeat([len(neg_indices_tr_h)])
#
#         # It looks Torch sparse tensor does not work in multi processing
#         # so they need to be converted to dense, which is not memory efficient
#         # https://github.com/pytorch/pytorch/pull/27062
#         # https://github.com/pytorch/pytorch/issues/20248
#         hr_t = torch.sparse.LongTensor(torch.LongTensor(_to_sparse_i(indices_hr_t)), values_hr_t, torch.Size(shape)).to_dense()
#         tr_h = torch.sparse.LongTensor(torch.LongTensor(_to_sparse_i(indices_tr_h)), values_tr_h, torch.Size(shape)).to_dense()
#
#         if neg_rate > 0:
#             neg_hr_t = torch.sparse.LongTensor(torch.LongTensor(_to_sparse_i(neg_indices_hr_t)), neg_values_hr_t, torch.Size(shape)).to_dense()
#             neg_tr_h = torch.sparse.LongTensor(torch.LongTensor(_to_sparse_i(neg_indices_tr_h)), neg_values_tr_h, torch.Size(shape)).to_dense()
#
#             hr_t = hr_t.add(neg_hr_t)
#             tr_h = tr_h.add(neg_tr_h)
#
#         processed_queue.put([h, r, t, hr_t, tr_h])


class Generator:
    """Generator class for the embedding algorithms

        Args:
          config (object): generator configuration object.
          model_config (object): Model configuration object.

        Yields:
            matrix : Batch size of processed triples

        Examples:
            >>> from pykg2vec.utils.generator import Generator
            >>> from pykg2vec.models.TransE impor TransE
            >>> model = TransE()
            >>> gen_train = Generator(model.config, training_strategy=TrainingStrategy.PAIRWISE_BASED)
    """

    @staticmethod
    def load_raw_data(config):
        data = config.knowledge_graph.read_cache_data('triplets_train')
        number_of_batch = len(data) // config.batch_size # TODO: we lost the last not intact batch in here
        random_ids = np.random.permutation(len(data))
        raw_data = []

        for batch_idx in range(number_of_batch):
            pos_start = config.batch_size * batch_idx
            pos_end = config.batch_size * (batch_idx + 1)
            raw_data.append(
                np.asarray([[data[x].h, data[x].r, data[x].t] for x in random_ids[pos_start:pos_end]]))

        del data  # save memory space
        gc.collect()

        return raw_data

    def stop(self):
        """Function to stop the generator."""

    def start_one_epoch(self, num_batch):
        """Function to start one epoch training."""


class PairwiseDataset(torch.utils.data.Dataset, Generator):
    """Dataset for holding triples for pairwise training"""

    def __init__(self, config):
        self.relation_property = config.knowledge_graph.read_cache_data('relationproperty')
        self.neg_rate = config.neg_rate
        self.sampling = config.sampling
        self.tot_entity = config.tot_entity
        self.raw_data = self.load_raw_data(config)
        self.positive_triplets = {(t[0], t[1], t[2]): 1 for t in self.raw_data}

    def __len__(self):
        """Total number of samples"""
        return len(self.raw_data)

    def __getitem__(self, index):
        """Generate one data sample"""
        item = self.raw_data[index]
        if item is None:
            return
        _, pos_triples = item

        ph = pos_triples[:, 0]
        pr = pos_triples[:, 1]
        pt = pos_triples[:, 2]

        nh = []
        nr = []
        nt = []

        for t in pos_triples:

            prob = self.relation_property[t[1]] if self.sampling == "bern" else 0.5

            for _ in range(self.neg_rate):

                if np.random.random() > prob:
                    idx_replace_tail = np.random.randint(self.tot_entity)

                    while (t[0], t[1], idx_replace_tail) in self.positive_triplets:
                        idx_replace_tail = np.random.randint(self.tot_entity)

                    nh.append(t[0])
                    nr.append(t[1])
                    nt.append(idx_replace_tail)

                else:
                    idx_replace_head = np.random.randint(self.tot_entity)

                    while (idx_replace_head, t[1], t[2]) in self.positive_triplets:
                        idx_replace_head = np.random.randint(self.tot_entity)

                    nh.append(idx_replace_head)
                    nr.append(t[1])
                    nt.append(t[2])

        return [ph, pr, pt, nh, nr, nt]

    def stop(self):
        """Function to stop all the worker process."""
        self.command_queue.put("quit")
        for worker_process in self.process_list:
            while True:
                worker_process.join(1)
                if not worker_process.is_alive():
                    break

    def start_one_epoch(self, num_batch):
        self.command_queue.put(num_batch)


class PointwiseDataset(torch.utils.data.Dataset, Generator):
    """Dataset for holding triples for pointwise training"""

    def __init__(self, config):
        self.relation_property = config.knowledge_graph.read_cache_data('relationproperty')
        self.neg_rate = config.neg_rate
        self.sampling = config.sampling
        self.tot_entity = config.tot_entity
        self.batch_size = config.batch_size
        self.raw_data = self.load_raw_data(config)
        self.positive_triplets = {(t[0], t[1], t[2]): 1 for t in self.raw_data}

    def __len__(self):
        """Total number of samples"""
        return len(self.raw_data)

    def __getitem__(self, index):
        """Generate one data sample"""
        item = self.raw_data[index]
        if item is None:
            return
        _, pos_triples = item

        point_h = []
        point_r = []
        point_t = []
        point_y = []

        for t in pos_triples:
            # postive sample
            point_h.append(t[0])
            point_r.append(t[1])
            point_t.append(t[2])
            point_y.append(1)

            prob = self.relation_property[t[1]] if self.sampling == "bern" else 0.5

            for _ in range(self.neg_rate):

                if np.random.random() > prob:
                    idx_replace_tail = np.random.randint(self.tot_entity)

                    while (t[0], t[1], idx_replace_tail) in self.positive_triplets:
                        idx_replace_tail = np.random.randint(self.tot_entity)

                    point_h.append(t[0])
                    point_r.append(t[1])
                    point_t.append(idx_replace_tail)
                    point_y.append(-1)

                else:
                    idx_replace_head = np.random.randint(self.tot_entity)

                    while (idx_replace_head, t[1], t[2]) in self.positive_triplets:
                        idx_replace_head = np.random.randint(self.tot_entity)

                    point_h.append(idx_replace_head)
                    point_r.append(t[1])
                    point_t.append(t[2])
                    point_y.append(-1)

        return [point_h, point_r, point_t, point_y]

    def stop(self):
        """Function to stop all the worker process."""
        self.command_queue.put("quit")
        for worker_process in self.process_list:
            while True:
                worker_process.join(1)
                if not worker_process.is_alive():
                    break

    def start_one_epoch(self, num_batch):
        self.command_queue.put(num_batch)

class MulticlassDataset(torch.utils.data.Dataset, Generator):
    """Dataset for holding triples for multiclass training"""

    def __init__(self, config):
        self.data = [[data.h, data.r, data.t] for data in config.knowledge_graph.read_cache_data('triplets_train')]
        self.tot_entity = config.tot_entity
        self.batch_size = config.batch_size
        self.neg_rate = config.neg_rate
        self.hr_t_train = config.knowledge_graph.read_cache_data('hr_t_train')
        self.tr_h_train = config.knowledge_graph.read_cache_data('tr_h_train')
        self.raw_data = self.load_raw_data(config)


    def __len__(self):
        """Total number of samples"""
        return len(self.raw_data)

    def __getitem__(self, index):
        """Generate one data sample"""

        def _to_sparse_i(indices):
            x = []
            y = []
            for index in indices:
                x.append(index[0])
                y.append(index[1])
            return [x, y]


        neg_rate = self.neg_rate

        shape = [self.batch_size, self.tot_entity]

        item = self.raw_data[index]
        if item is None:
            return
        _, pos_triples = item

        if item is None:
            return
        idx, raw_data = item

        h = raw_data[:, 0]
        r = raw_data[:, 1]
        t = raw_data[:, 2]

        indices_hr_t = []
        indices_tr_h = []
        neg_indices_hr_t = []
        neg_indices_tr_h = []

        random_ids = np.random.permutation(self.tot_entity)

        for i in range(self.batch_size):
            hr_t = self.hr_t_train[(h[i], r[i])]
            tr_h = self.tr_h_train[(t[i], r[i])]

            for idx in hr_t:
                indices_hr_t.append([i, idx])
            for idx in tr_h:
                indices_tr_h.append([i, idx])

            if neg_rate > 0:
                for idx in random_ids[0:100]:
                    if idx not in hr_t:
                        neg_indices_hr_t.append([i, idx])
                for idx in random_ids[0:100]:
                    if idx not in tr_h:
                        neg_indices_tr_h.append([i, idx])

        values_hr_t = torch.FloatTensor([1]).repeat([len(indices_hr_t)])
        values_tr_h = torch.FloatTensor([1]).repeat([len(indices_tr_h)])

        if neg_rate > 0:
            neg_values_hr_t = torch.FloatTensor([-1]).repeat([len(neg_indices_hr_t)])
            neg_values_tr_h = torch.FloatTensor([-1]).repeat([len(neg_indices_tr_h)])

        # It looks Torch sparse tensor does not work in multi processing
        # so they need to be converted to dense, which is not memory efficient
        # https://github.com/pytorch/pytorch/pull/27062
        # https://github.com/pytorch/pytorch/issues/20248
        hr_t = torch.sparse.LongTensor(torch.LongTensor(_to_sparse_i(indices_hr_t)), values_hr_t,
                                       torch.Size(shape)).to_dense()
        tr_h = torch.sparse.LongTensor(torch.LongTensor(_to_sparse_i(indices_tr_h)), values_tr_h,
                                       torch.Size(shape)).to_dense()

        if neg_rate > 0:
            neg_hr_t = torch.sparse.LongTensor(torch.LongTensor(_to_sparse_i(neg_indices_hr_t)), neg_values_hr_t,
                                               torch.Size(shape)).to_dense()
            neg_tr_h = torch.sparse.LongTensor(torch.LongTensor(_to_sparse_i(neg_indices_tr_h)), neg_values_tr_h,
                                               torch.Size(shape)).to_dense()

            hr_t = hr_t.add(neg_hr_t)
            tr_h = tr_h.add(neg_tr_h)

        return [h, r, t, hr_t, tr_h]

    def stop(self):
        """Function to stop all the worker process."""
        self.command_queue.put("quit")
        for worker_process in self.process_list:
            while True:
                worker_process.join(1)
                if not worker_process.is_alive():
                    break

    def start_one_epoch(self, num_batch):
        self.command_queue.put(num_batch)

