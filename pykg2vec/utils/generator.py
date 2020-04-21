#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for generating the batch data for training and testing.
"""
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
from multiprocessing import Process, Queue, Event
from enum import Enum

import os

def raw_data_generator(command_queue, raw_queue, config):
    """Function to feed  triples to raw queue for multiprocessing.
           
        Args:
            raw_queue (Queue) : Multiprocessing Queue to put the raw data to be processed.
            data (list) : List of integer ids denoting positive triples.
            batch_size (int) : Size of each batch.
            number_of_batch (int) : Total number of batch.

    """
    data = config.knowledge_graph.read_cache_data('triplets_train')

    number_of_batch = len(data) // config.batch_size

    random_ids = np.random.permutation(len(data))
    
    while True:

        command = command_queue.get()

        if command == "quit":
            raw_queue.put(None)
            raw_queue.put(None)
            return 
        else:
            number_of_batch = command 
            for batch_idx in range(number_of_batch):                
                pos_start = config.batch_size * batch_idx
                pos_end   = config.batch_size * (batch_idx+1)
                
                raw_data = np.asarray([[data[x].h, data[x].r, data[x].t] for x in random_ids[pos_start:pos_end]])
                
                raw_queue.put((batch_idx, raw_data))


def process_function_pairwise(raw_queue, processed_queue, config):
    """Function that puts the processed data in the queue.
           
        Args:
            raw_queue (Queue) : Multiprocessing Queue to put the raw data to be processed.
            processed_queue (Queue) : Multiprocessing Queue to put the processed data.
            te (int): Total number of entities
            bs (int): Total size of each batch.
            positive_triplets (list) : List of positive triples.
            lh (int): Id of the last processed head.
            lr (int): Id of the last processed relation.
            lt (int): Id of the last processed tail.
    """ 
    data = config.knowledge_graph.read_cache_data('triplets_train')
    relation_property = config.knowledge_graph.read_cache_data('relationproperty')
    positive_triplets = {(t.h, t.r, t.t): 1 for t in data}
    neg_rate = config.neg_rate
    
    del data # save memory space

    while True:
        item = raw_queue.get()
        if item is None:
            return
        idx, pos_triples = item

        ph = pos_triples[:, 0]
        pr = pos_triples[:, 1]
        pt = pos_triples[:, 2]
        
        nh = []
        nr = []
        nt = []

        for t in pos_triples:
            
            prob = relation_property[t[1]] if config.sampling == "bern" else 0.5
            
            for i in range(neg_rate):
                
                if np.random.random() > prob:
                    idx_replace_tail = np.random.randint(config.kg_meta.tot_entity)

                    while (t[0], t[1], idx_replace_tail) in positive_triplets:
                        idx_replace_tail = np.random.randint(config.kg_meta.tot_entity)

                    nh.append(t[0])
                    nr.append(t[1])
                    nt.append(idx_replace_tail)

                else:
                    idx_replace_head = np.random.randint(config.kg_meta.tot_entity)
                    
                    while ((idx_replace_head, t[1], t[2]) in positive_triplets):
                        idx_replace_head = np.random.randint(config.kg_meta.tot_entity)
                    
                    nh.append(idx_replace_head)
                    nr.append(t[1])
                    nt.append(t[2])

        processed_queue.put([ph, pr, pt, nh, nr, nt])

def process_function_pointwise(raw_queue, processed_queue, config):
    """Function that puts the processed data in the queue.
           
        Args:
            raw_queue (Queue) : Multiprocessing Queue to put the raw data to be processed.
            processed_queue (Queue) : Multiprocessing Queue to put the processed data.
            te (int): Total number of entities
            bs (int): Total size of each batch.
            positive_triplets (list) : List of positive triples.
            lh (int): Id of the last processed head.
            lr (int): Id of the last processed relation.
            lt (int): Id of the last processed tail.
    """ 
    data = config.knowledge_graph.read_cache_data('triplets_train')
    relation_property = config.knowledge_graph.read_cache_data('relationproperty')
    positive_triplets = {(t.h, t.r, t.t): 1 for t in data}
    neg_rate = config.neg_rate
    
    del data # save memory space

    while True:
        item = raw_queue.get()
        if item is None:
            return
        idx, pos_triples = item

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

            prob = relation_property[t[1]] if config.sampling == "bern" else 0.5
            
            for i in range(neg_rate):
                
                if np.random.random() > prob:
                    idx_replace_tail = np.random.randint(config.kg_meta.tot_entity)

                    while (t[0], t[1], idx_replace_tail) in positive_triplets:
                        idx_replace_tail = np.random.randint(config.kg_meta.tot_entity)

                    point_h.append(t[0])
                    point_r.append(t[1])
                    point_t.append(idx_replace_tail)
                    point_y.append(-1)

                else:
                    idx_replace_head = np.random.randint(config.kg_meta.tot_entity)
                    
                    while ((idx_replace_head, t[1], t[2]) in positive_triplets):
                        idx_replace_head = np.random.randint(config.kg_meta.tot_entity)
                    
                    point_h.append(idx_replace_head)
                    point_r.append(t[1])
                    point_t.append(t[2])
                    point_y.append(-1)

        processed_queue.put([point_h, point_r, point_t, point_y])


def process_function_multiclass(raw_queue, processed_queue, config):
    """Function that puts the processed data in the queue.
           
        Args:
            raw_queue (Queue) : Multiprocessing Queue to put the raw data to be processed.
            processed_queue (Queue) : Multiprocessing Queue to put the processed data.
            te (int): Total number of entities
            bs (int): Total size of each batch.
            neg_rate (int): Ratio of negative to positive samples.
    """
    hr_t_train = config.knowledge_graph.read_cache_data('hr_t_train')
    tr_h_train = config.knowledge_graph.read_cache_data('tr_h_train')
    
    neg_rate = config.neg_rate
    
    shape = [config.batch_size, config.kg_meta.tot_entity] 
    shape = tf.convert_to_tensor(shape, dtype=tf.int64)
    
    while True:
        item = raw_queue.get()
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

        random_ids = np.random.permutation(config.kg_meta.tot_entity)

        for i in range(config.batch_size):
            hr_t = hr_t_train[(h[i], r[i])]
            tr_h = tr_h_train[(t[i], r[i])]

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


        values_hr_t = tf.tile([1], [len(indices_hr_t)])
        values_tr_h = tf.tile([1], [len(indices_tr_h)])
        
        if neg_rate > 0:
            neg_values_hr_t = tf.tile([-1], [len(neg_indices_hr_t)])
            neg_values_tr_h = tf.tile([-1], [len(neg_indices_tr_h)])

        hr_t = tf.SparseTensor(indices=indices_hr_t, values=values_hr_t, dense_shape=shape)
        tr_h = tf.SparseTensor(indices=indices_tr_h, values=values_tr_h, dense_shape=shape)

        if neg_rate > 0:
            neg_hr_t = tf.SparseTensor(indices=neg_indices_hr_t, values=neg_values_hr_t, dense_shape=shape)
            neg_tr_h = tf.SparseTensor(indices=neg_indices_tr_h, values=neg_values_tr_h, dense_shape=shape)
        
            hr_t = tf.sparse.add(hr_t, neg_hr_t)
            tr_h = tf.sparse.add(tr_h, neg_tr_h)

        processed_queue.put([h, r, t, hr_t, tr_h])


class Generator:
    """Generator class for the embedding algorithms
        
        Args:
          config (object): generator configuration object.
          model_config (object): Model configuration object.

        Yields:
            matrix : Batch size of processed triples

        Examples:
            >>> from pykg2vec.utils.generator import Generator
            >>> from pykg2vec.core.TransE impor TransE
            >>> model = TransE()
            >>> gen_train = Generator(model.config, training_strategy=TrainingStrategy.PAIRWISE_BASED)
    """

    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.training_strategy = model.training_strategy

        self.process_list = []
        
        self.raw_queue_size = 10
        self.processed_queue_size = 10
        self.command_queue = Queue(self.raw_queue_size)
        self.raw_queue = Queue(self.raw_queue_size)
        self.processed_queue = Queue(self.processed_queue_size)
        
        self.create_feeder_process()
        self.create_train_processor_process()

    def __iter__(self):
        return self

    def __next__(self):
        return self.processed_queue.get()
        
    def stop(self):
        """Function to stop all the worker process."""
        self.command_queue.put("quit")        
        for worker_process in self.process_list:
            while True:
                worker_process.join(1)
                if not worker_process.is_alive():
                    break

    def create_feeder_process(self):
        """Function create the feeder process."""
        feeder_worker = Process(target=raw_data_generator, args=(self.command_queue, self.raw_queue, self.config))
        self.process_list.append(feeder_worker)
        feeder_worker.daemon = True
        feeder_worker.start()

    def create_train_processor_process(self):
        """Function ro create the process for generating training samples."""
        for i in range(self.config.num_process_gen):
            if self.training_strategy == TrainingStrategy.PROJECTION_BASED:
                process_worker = Process(target=process_function_multiclass, args=(self.raw_queue, self.processed_queue, self.config))
            elif self.training_strategy == TrainingStrategy.PAIRWISE_BASED:
                process_worker = Process(target=process_function_pairwise, args=(self.raw_queue, self.processed_queue, self.config))
            elif self.training_strategy == TrainingStrategy.POINTWISE_BASED:
                process_worker = Process(target=process_function_pointwise, args=(self.raw_queue, self.processed_queue, self.config))
            else:
                raise NotImplementedError("This strategy is not supported.")
            self.process_list.append(process_worker)
            process_worker.daemon = True
            process_worker.start()

    def start_one_epoch(self, num_batch):
        self.command_queue.put(num_batch)


class TrainingStrategy(Enum):
    PROJECTION_BASED = "projection_based"   # matching models with neural network
    PAIRWISE_BASED = "pairwise_based"       # translational distance models
    POINTWISE_BASED = "pointwise_based"     # semantic matching models
