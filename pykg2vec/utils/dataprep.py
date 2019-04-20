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

from config.global_config import GlobalConfig
import numpy as np
import pickle
import os
from collections import defaultdict
import pprint
from scipy import sparse as sps


class Triple(object):
    def __init__(self, head=None, relation=None, tail=None):
        self.h = head
        self.r = relation
        self.t = tail


class DataInput(object):
    def __init__(self, e1=None, r=None, e2=None, r_rev=None, e2_multi1=None, e2_multi2=None):
        self.e1 = e1
        self.r = r
        self.e2 = e2
        self.r_rev = r_rev
        self.e2_multi1 = e2_multi1
        self.e2_multi2 = e2_multi2


class DataPrep(object):

    def __init__(self, name_dataset='Freebase15k', algo=False):
        '''store the information of database'''

        self.config = GlobalConfig(dataset=name_dataset)
        self.algo = algo
        self.train_triples = []
        self.test_triples = []
        self.validation_triples = []

        self.tot_relation = 0
        self.tot_triple = 0
        self.tot_entity = 0

        self.entity2idx = {}
        self.idx2entity = {}

        self.relation2idx = {}
        self.idx2relation = {}

        self.hr_t = defaultdict(set)
        self.tr_t = defaultdict(set)

        # for ConvE
        self.label_graph = {}
        self.train_graph = {}
        self.train_data = []
        self.test_data = []
        self.valid_data = []
        self.test_triples_no_rev = []
        self.validation_triples_no_rev = []

        # self.train_label
        if not self.algo:
            self.read_triple(['train', 'test', 'valid'])  # TODO: save the triples to prevent parsing everytime
            self.calculate_mapping()  # from entity and relation to indexes.
            self.test_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t]) for t in
                                     self.test_triples]
            self.train_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t]) for t
                                      in
                                      self.train_triples]
            self.validation_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t])
                                           for t
                                           in self.validation_triples]
        else:
            self.read_triple_hr_rt(['train', 'test', 'valid'])
            self.calculate_mapping()  # from entity and relation to indexes.

            for (e, r) in self.train_graph:
                e1_idx = self.entity2idx[e]
                r_idx = self.relation2idx[r]
                e2_multi1 = [self.entity2idx[i] for i in list(self.train_graph[(e, r)])]
                self.train_data.append(DataInput(e1=e1_idx, r=r_idx, e2_multi1=e2_multi1))

            for t in self.test_triples_no_rev:
                e1_idx = self.entity2idx[t.h]
                e2_idx = self.entity2idx[t.t]
                r_idx = self.relation2idx[t.r]
                if type(r_idx) is not int:
                    print(t.r, r_idx)
                r_rev_idx = self.relation2idx[t.r + '_reverse']
                e2_multi1 = [self.entity2idx[i] for i in self.label_graph[(t.h, t.r)]]
                e2_multi2 = [self.entity2idx[i] for i in self.label_graph[(t.t, t.r + '_reverse')]]
                self.test_data.append(DataInput(e1=e1_idx, r=r_idx,
                                                e2=e2_idx, r_rev=r_rev_idx,
                                                e2_multi1=e2_multi1, e2_multi2=e2_multi2))

            for t in self.validation_triples_no_rev:
                e1_idx = self.entity2idx[t.h]
                e2_idx = self.entity2idx[t.t]
                r_idx = self.relation2idx[t.r]
                r_rev_idx = self.relation2idx[t.r + '_reverse']
                e2_multi1 = [self.entity2idx[i] for i in self.label_graph[(t.h, t.r)]]
                e2_multi2 = [self.entity2idx[i] for i in self.label_graph[(t.t, t.r + '_reverse')]]
                if type(r_idx) is not int:
                    print(t.r, r_idx)
                self.valid_data.append(DataInput(e1=e1_idx, r=r_idx,
                                                 e2=e2_idx, r_rev=r_rev_idx,
                                                 e2_multi1=e2_multi1, e2_multi2=e2_multi2))
            self.validation_triples_ids = [
                Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t])
                for t
                in self.validation_triples]

        for t in self.train_triples:
            self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
            self.tr_t[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])

        for t in self.test_triples:
            self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
            self.tr_t[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])

        for t in self.validation_triples:
            self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
            self.tr_t[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])

        if not self.algo:
            self.relation_property_head = {x: [] for x in
                                           range(self.tot_relation)}
            self.relation_property_tail = {x: [] for x in
                                           range(self.tot_relation)}
            for t in self.train_triples_ids:
                self.relation_property_head[t.r].append(t.h)
                self.relation_property_tail[t.r].append(t.t)

            self.relation_property = {x: (len(set(self.relation_property_tail[x]))) / (
                    len(set(self.relation_property_head[x])) + len(set(self.relation_property_tail[x]))) \
                                      for x in
                                      self.relation_property_head.keys()}

    def calculate_mapping(self):
        print("Calculating entity2idx & idx2entity & relation2idx & idx2relation.")

        if self.config.dataset.entity2idx_path.is_file():
            with open(str(self.config.dataset.entity2idx_path), 'rb') as f:
                self.entity2idx = pickle.load(f)

            with open(str(self.config.dataset.idx2entity_path), 'rb') as f:
                self.idx2entity = pickle.load(f)

            with open(str(self.config.dataset.relation2idx_path), 'rb') as f:
                self.relation2idx = pickle.load(f)

            with open(str(self.config.dataset.idx2relation_path), 'rb') as f:
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

        if not os.path.isfile(str(self.config.dataset.entity2idx_path)):
            with open(str(self.config.dataset.entity2idx_path), 'wb') as f:
                pickle.dump(self.entity2idx, f)
        # save idx2entity
        if not os.path.isfile(str(self.config.dataset.idx2entity_path)):
            with open(str(self.config.dataset.idx2entity_path), 'wb') as f:
                pickle.dump(self.idx2entity, f)
        # save relation2idx
        if not os.path.isfile(str(self.config.dataset.relation2idx_path)):
            with open(str(self.config.dataset.relation2idx_path), 'wb') as f:
                pickle.dump(self.relation2idx, f)
        # save idx2relation
        if not os.path.isfile(str(self.config.dataset.idx2relation_path)):
            with open(str(self.config.dataset.idx2relation_path), 'wb') as f:
                pickle.dump(self.idx2relation, f)

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
    def read_triple(self, datatype=None):
        print("Reading Triples", datatype)

        for data in datatype:
            with open(str(self.config.dataset.downloaded_path) + data + '.txt', 'r') as f:
                for l in f.readlines():
                    h, r, t = l.split('\t')
                    triple = Triple(h.strip(), r.strip(), t.strip())
                    if data == 'train':
                        self.train_triples.append(triple)
                    elif data == 'test':
                        self.test_triples.append(triple)
                    elif data == 'valid':
                        self.validation_triples.append(triple)
                    else:
                        continue

    def read_triple_hr_rt(self, datatype=None, reverse_r=True):
        print("Reading Triples", datatype)

        for data in datatype:
            with open(str(self.config.dataset.downloaded_path) + data + '.txt', 'r') as f:
                for l in f.readlines():
                    h, r, t = l.split('\t')
                    h = h.strip()
                    r = r.strip()
                    t = t.strip()

                    r_rev = r + '_reverse'
                    triple = Triple(h, r, t)
                    triple_r = Triple(t, r_rev, h)

                    if (h, r) not in self.label_graph:
                        self.label_graph[(h, r)] = set()
                    self.label_graph[(h, r)].add(t)

                    if (t, r_rev) not in self.label_graph:
                        self.label_graph[(t, r_rev)] = set()
                    self.label_graph[(t, r_rev)].add(h)

                    if data == 'train':
                        self.train_triples.append(triple)
                        self.train_triples.append(triple_r)
                        if (h, r) not in self.train_graph:
                            self.train_graph[(h, r)] = set()
                        if (t, r_rev) not in self.train_graph:
                            self.train_graph[(t, r_rev)] = set()

                        self.train_graph[(h, r)].add(t)
                        self.train_graph[(t, r_rev)].add(h)

                    elif data == 'test':
                        self.test_triples.append(triple)
                        self.test_triples.append(triple_r)
                        self.test_triples_no_rev.append(triple)

                    elif data == 'valid':
                        self.validation_triples.append(triple)
                        self.validation_triples.append(triple_r)
                        self.validation_triples_no_rev.append(triple)

                    else:
                        continue

    def dump(self):
        ''' dump key information'''
        print("\n----------Relation to Indexes--------------")
        pprint.pprint(self.relation2idx)
        print("---------------------------------------------")

        print("\n----------Relation to Indexes--------------")
        pprint.pprint(self.idx2relation)
        print("---------------------------------------------")

        print("\n----------Train Triple Stats---------------")
        print("Total Training Triples   :", len(self.train_triples))
        print("Total Testing Triples    :", len(self.test_triples))
        print("Total validation Triples :", len(self.validation_triples))
        print("Total Entities           :", self.tot_entity)
        print("Total Relations          :", self.tot_relation)
        print("---------------------------------------------")

    def dump_triples(self):
        '''dump all the triples'''
        for idx, triple in enumerate(self.train_triples):
            print(idx, triple.h, triple.r, triple.t)
        for idx, triple in enumerate(self.test_triples):
            print(idx, triple.h, triple.r, triple.t)
        for idx, triple in enumerate(self.validation_triples):
            print(idx, triple.h, triple.r, triple.t)


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
