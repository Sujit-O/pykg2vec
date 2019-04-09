#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for preparing the data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append("D:\louis\Dropbox\louis_research\pyKG2Vec\pykg2vec")
from config.global_config import GlobalConfig

import numpy as np
# from pykg2vec.config.config import GlobalConfig
import pickle
import os
from collections import defaultdict
import pprint


class Triple(object):
    def __init__(self, head=None, relation=None, tail=None):
        self.h = head
        self.r = relation
        self.t = tail


def parse_line(line):
    h, r, t = line.split('\t')
    h = h.split(' ')[0]
    r = r.split(' ')[0]
    t = t.split(' ')[0]
    return Triple(h, r, t)


class DataPrep(object):

    def __init__(self, name_dataset='Freebase15k'):

        '''store the information of database'''
        self.config = GlobalConfig(dataset=name_dataset)

        self.train_triples = []
        self.test_triples = []
        self.validation_triples = []
        self.train_triples_ids = []
        self.test_triples_ids = []
        self.validation_triples_ids = []

        self.entity2idx = {}
        self.idx2entity = {}

        self.relation2idx = {}
        self.idx2relation = {}

        self.hr_t = defaultdict(set)
        self.tr_t = defaultdict(set)

        self.tot_relation = 0
        self.tot_triple = 0
        self.tot_entity = 0

        self.relation_property_head = None
        self.relation_property_tail = None
        self.relation_property = None

        # read the train, test and valid triples
        print("\tReading Triples")
        self.read_triple(['train', 'test', 'valid'])
        # TODO: save the triples to prevent parsing everytime
        print("\tConverting triple to idx")
        self.get_idx()
        self.convert2idx()

        if self.config.negative_sample == 'bern':
            self.negative_sampling()

    def dumpdata(self):
        pprint.pprint(self.relation2idx)
        pprint.pprint(self.idx2relation)
        # for key,val in self.__dict__.items():
        #     pprint.pprint(key, val)

    def read_triple(self, datatype=None):
        if datatype is None:
            datatype = ['train']
        for data in datatype:
            with open(str(self.config.dataset.downloaded_path) + data + '.txt', 'r') as f:
                lines = f.readlines()
                for l in lines:
                    triple = parse_line(l)
                    if data == 'train':
                        self.train_triples.append(triple)
                    elif data == 'test':
                        self.test_triples.append(triple)
                    elif data == 'valid':
                        self.validation_triples.append(triple)
                    else:
                        continue

    def get_idx(self):
        if os.path.isfile(self.config.dataset.entity2idx_path):
            with open(self.config.dataset.entity2idx_path, 'rb') as f:
                self.entity2idx = pickle.load(f)

            with open(self.config.dataset.idx2entity_path, 'rb') as f:
                self.idx2entity = pickle.load(f)

            with open(self.config.dataset.relation2idx_path, 'rb') as f:
                self.relation2idx = pickle.load(f)

            with open(self.config.dataset.idx2relation_path, 'rb') as f:
                self.idx2relation = pickle.load(f)

            self.tot_entity = len(self.entity2idx)
            self.tot_relation = len(self.relation2idx)
            self.tot_triple = len(self.train_triples) + \
                              len(self.test_triples) + \
                              len(self.validation_triples)
            return

        heads = []
        tails = []
        relations = []

        for triple in self.train_triples:
            heads += [triple.h]
            tails += [triple.t]
            relations += [triple.r]
            self.tot_triple += 1

        for triple in self.test_triples:
            heads += [triple.h]
            tails += [triple.t]
            relations += [triple.r]
            self.tot_triple += 1

        for triple in self.validation_triples:
            heads += [triple.h]
            tails += [triple.t]
            relations += [triple.r]
            self.tot_triple += 1

        entities = np.sort(list(set(heads) | (set(tails))))
        relations = np.sort(list(set(relations)))

        self.tot_entity = len(entities)
        self.tot_relation = len(relations)

        self.entity2idx = {v: k for k, v in enumerate(entities)}
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}

        self.relation2idx = {v: k for k, v in enumerate(relations)}
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}

        # pdb.set_trace()
        # save entity2idx
        if not os.path.isfile(self.config.dataset.entity2idx_path):
            with open(self.config.dataset.entity2idx_path, 'wb') as f:
                pickle.dump(self.entity2idx, f)
        # save idx2entity
        if not os.path.isfile(self.config.dataset.idx2entity_path):
            with open(self.config.dataset.idx2entity_path, 'wb') as f:
                pickle.dump(self.idx2entity, f)
        # save relation2idx
        if not os.path.isfile(self.config.dataset.relation2idx_path):
            with open(self.config.dataset.relation2idx_path, 'wb') as f:
                pickle.dump(self.relation2idx, f)
        # save idx2relation
        if not os.path.isfile(self.config.dataset.idx2relation_path):
            with open(self.config.dataset.idx2relation_path, 'wb') as f:
                pickle.dump(self.idx2relation, f)

        print("\n----------Train Triple Stats---------------")
        print("Total Training Triples   :", len(self.train_triples))
        print("Total Testing Triples    :", len(self.test_triples))
        print("Total validation Triples :", len(self.validation_triples))
        print("Total Entities           :", self.tot_entity)
        print("Total Relations          :", self.tot_relation)
        print("---------------------------------------------")

    def convert2idx(self):
        for t in self.test_triples:
            self.test_triples_ids.append(Triple(self.entity2idx[t.h],
                                                self.relation2idx[t.r],
                                                self.entity2idx[t.t]))
            self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
            self.tr_t[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])
        for t in self.train_triples:
            self.train_triples_ids.append(Triple(self.entity2idx[t.h],
                                                 self.relation2idx[t.r],
                                                 self.entity2idx[t.t]))
            self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
            self.tr_t[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])

        for t in self.validation_triples:
            self.validation_triples_ids.append(Triple(self.entity2idx[t.h],
                                                      self.relation2idx[t.r],
                                                      self.entity2idx[t.t]))
            self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
            self.tr_t[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])

    def print_triple(self):
        for triple in self.train_triples:
            print(triple.h, triple.r, triple.t)
        for triple in self.test_triples:
            print(triple.h, triple.r, triple.t)
        for triple in self.validation_triples:
            print(triple.h, triple.r, triple.t)

    def negative_sampling(self):
        self.relation_property_head = {x: [] for x in
                                       range(self.tot_relation)}
        self.relation_property_tail = {x: [] for x in
                                       range(self.tot_relation)}
        for t in self.train_triples:
            self.relation_property_head[t[1]].append(t[0])
            self.relation_property_tail[t[1]].append(t[2])
        self.relation_property = {x: (len(set(self.relation_property_tail[x]))) / (
                len(set(self.relation_property_head[x])) + len(set(self.relation_property_tail[x]))) \
                                  for x in
                                  self.relation_property_head.keys()}

    def batch_generator(self, batch=128, data='test'):

        if data == 'test':
            triples = self.test_triples_ids
        elif data == 'valid':
            triples = self.validation_triples_ids
        else:
            raise NotImplementedError("%s data not present" % data)
        num_triples = len(triples)

        rand_ids = np.random.permutation(num_triples)
        number_of_batches = num_triples // batch
        # print("Number of batches:", number_of_batches)

        counter = 0
        while True:
            pos_triples = np.asarray(
                [[triples[x].h, triples[x].r, triples[x].t] for x in rand_ids[batch * counter:batch * (counter + 1)]])
            ph = pos_triples[:, 0]
            pr = pos_triples[:, 1]
            pt = pos_triples[:, 2]

            counter += 1
            yield ph, pr, pt
            if counter == number_of_batches:
                counter = 0

    def batch_generator_train(self, batch=128):
        num_triples = len(self.train_triples_ids)
        pos_triples_hm = {}
        neg_triples_hm = {}

        for t in self.train_triples_ids:
            pos_triples_hm[(t.h, t.r, t.t)] = 1

        rand_ids = np.random.permutation(num_triples)
        number_of_batches = num_triples // batch
        print("Number of batches:", number_of_batches)

        counter = 0
        last_h = 0
        last_r = 0
        last_t = 0
        while True:
            pos_triples = np.asarray([[self.train_triples_ids[x].h,
                                       self.train_triples_ids[x].r,
                                       self.train_triples_ids[x].t] for x in
                                      rand_ids[batch * counter:batch * (counter + 1)]])
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
                    idx = np.random.randint(self.tot_entity)
                    break_cnt = 0
                    flag = False
                    while ((t[0], t[1], idx) in pos_triples_hm
                           or (t[0], t[1], idx) in neg_triples_hm):
                        idx = np.random.randint(self.tot_entity)
                        break_cnt += 1
                        if break_cnt >= 100:
                            flag = True
                            break
                    if flag:
                        nh.append(last_h)
                        nr.append(last_r)
                        nt.append(last_t)
                    else:
                        nh.append(t[0])
                        nr.append(t[1])
                        nt.append(idx)
                        last_h = t[0]
                        last_r = t[1]
                        last_t = idx
                        neg_triples_hm[(t[0], t[1], idx)] = 1
                else:
                    idx = np.random.randint(self.tot_entity)
                    break_cnt = 0
                    flag = False
                    while ((idx, t[1], t[2]) in pos_triples_hm
                           or (idx, t[1], t[2]) in neg_triples_hm):
                        idx = np.random.randint(self.tot_entity)
                        break_cnt += 1
                        if break_cnt >= 100:
                            flag = True
                            break
                    if flag:
                        nh.append(last_h)
                        nr.append(last_r)
                        nt.append(last_t)
                    else:
                        nh.append(idx)
                        nr.append(t[1])
                        nt.append(t[2])
                        last_h = idx
                        last_r = t[1]
                        last_t = t[2]
                        neg_triples_hm[(idx, t[1], t[2])] = 1

            counter += 1
            yield ph, pr, pt, nh, nr, nt

            if counter == number_of_batches:
                counter = 0


if __name__ == '__main__':
    data_handler = DataPrep('Freebase15k')
    print("\n----------Train Triple Stats---------------")
    print("Total Training Triples   :", len(data_handler.train_triples))
    print("Total Testing Triples    :", len(data_handler.test_triples))
    print("Total validation Triples :", len(data_handler.validation_triples))
    print("Total Entities           :", data_handler.tot_entity)
    print("Total Relations          :", data_handler.tot_relation)
    print("---------------------------------------------")
    data_handler.dumpdata()
    gen = data_handler.batch_generator_train(batch=8)
    for i in range(10):
        ph, pr, pt, nh, nr, nt = list(next(gen))
        # print("\nph:", ph)
        # print("pr:", pr)
        # print("pt:", pt)
        # print("nh:", nh)
        # print("nr:", nr)
        # print("nt:", nt)
