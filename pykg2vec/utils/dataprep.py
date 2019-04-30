#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for preparing the data
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import progressbar

sys.path.append("../")

from config.global_config import GlobalConfig
import numpy as np
import pickle
import os
from collections import defaultdict
import pprint

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


class DataStats(object):
    def __init__(self, tot_entity=None,
                 tot_relation=None,
                 tot_triple=None,
                 tot_train_triples=None,
                 tot_test_triples=None,
                 tot_valid_triples=None):
        self.tot_triple = tot_triple
        self.tot_valid_triples = tot_valid_triples
        self.tot_test_triples = tot_test_triples
        self.tot_train_triples = tot_train_triples
        self.tot_relation = tot_relation
        self.tot_relation = tot_relation
        self.tot_entity = tot_entity


class DataPrep(object):

    def __init__(self, name_dataset='Freebase15k', sampling="uniform", algo='ConvE'):
        '''store the information of database'''
        self.data_stats = DataStats()
        self.config = GlobalConfig(dataset=name_dataset)
        self.algo = algo
        self.sampling = sampling
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
        self.tr_h = defaultdict(set)

        if not os.path.exists(self.config.tmp_data):
            os.mkdir(self.config.tmp_data)

        if self.algo.lower() in ["conve", "complex", "distmult"]:
            self.label_graph = {}
            self.train_graph = {}
            self.train_data = []
            self.test_data = []
            self.valid_data = []
            self.test_triples_no_rev = []
            self.validation_triples_no_rev = []

            self.read_triple_hr_rt(['train', 'test', 'valid'])
            self.calculate_mapping()

            if not os.path.exists(self.config.tmp_data / 'train_data.pkl'):
                print("\nPreparing Training Data!")
                with progressbar.ProgressBar(max_value=len(self.train_graph)) as bar:
                    for i, (e, r) in enumerate(self.train_graph):
                        e1_idx = self.entity2idx[e]
                        r_idx = self.relation2idx[r]
                        e2_multi1 = [self.entity2idx[i] for i in list(self.train_graph[(e, r)])]
                        self.train_data.append(DataInput(e1=e1_idx, r=r_idx, e2_multi1=e2_multi1))
                        bar.update(i)
                with open(self.config.tmp_data / 'train_data.pkl', 'wb') as f:
                    pickle.dump(self.train_data, f)
            else:
                with open(self.config.tmp_data / 'train_data.pkl', 'rb') as f:
                    self.train_data=pickle.load(f)
            self.data_stats.tot_train_triples = len(self.train_data)

            if not os.path.exists(self.config.tmp_data / 'test_data.pkl'):
                print("\nPreparing Testing Data!")
                with progressbar.ProgressBar(max_value=len(self.test_triples_no_rev)) as bar:
                    for i, t in enumerate(self.test_triples_no_rev):
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
                        bar.update(i)
                with open(self.config.tmp_data / 'test_data.pkl', 'wb') as f:
                    pickle.dump(self.test_data, f)
            else:
                with open(self.config.tmp_data / 'test_data.pkl', 'rb') as f:
                    self.test_data = pickle.load(f)
            self.data_stats.tot_test_triples = len(self.test_data)

            if not os.path.exists(self.config.tmp_data / 'valid_data.pkl'):
                print("\nPreparing Validation Data!")
                with progressbar.ProgressBar(max_value=len(self.validation_triples_no_rev)) as bar:
                    for i, t in enumerate(self.validation_triples_no_rev):
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
                        bar.update(i)
                with open(self.config.tmp_data / 'valid_data.pkl', 'wb') as f:
                    pickle.dump(self.valid_data, f)
            else:
                with open(self.config.tmp_data / 'valid_data.pkl', 'rb') as f:
                    self.valid_data = pickle.load(f)
            self.data_stats.tot_valid_triples = len(self.valid_data)

            self.validation_triples_ids = [
                Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t])
                for t
                in self.validation_triples]

        elif any(self.algo.lower().startswith(x) for x in ['transe', 'transr', 'transh', 'transd', 'transm', 'kg2e',
                                                           'proje', 'rescal',
                                                           'slm', 'sme_bilinear', 'sme_linear', 'ntn', 'rotate']):

            self.read_triple(['train', 'test', 'valid'])  # TODO: save the triples to prevent parsing everytime
            self.calculate_mapping()  # from entity and relation to indexes.

            if not os.path.exists(self.config.tmp_data / 'test_triples_ids.pkl'):
                self.test_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r],
                                                self.entity2idx[t.t]) for t in
                                         self.test_triples]
                with open(self.config.tmp_data / 'test_triples_ids.pkl', 'wb') as f:
                    pickle.dump(self.test_triples_ids, f)
            else:
                with open(self.config.tmp_data / 'test_triples_ids.pkl', 'rb') as f:
                    self.test_triples_ids = pickle.load(f)

            self.data_stats.tot_test_triples = len(self.test_triples_ids)

            if not os.path.exists(self.config.tmp_data / 'train_triples_ids.pkl'):
                self.train_triples_ids = [Triple(self.entity2idx[t.h],
                                                 self.relation2idx[t.r], self.entity2idx[t.t]) for t
                                          in
                                          self.train_triples]
                with open(self.config.tmp_data / 'train_triples_ids.pkl', 'wb') as f:
                    pickle.dump(self.train_triples_ids, f)
            else:
                with open(self.config.tmp_data / 'train_triples_ids.pkl', 'rb') as f:
                    self.train_triples_ids = pickle.load(f)

            self.data_stats.tot_train_triples = len(self.train_triples_ids)

            if not os.path.exists(self.config.tmp_data / 'validation_triples_ids.pkl'):
                self.validation_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r],
                                                      self.entity2idx[t.t])
                                               for t
                                               in self.validation_triples]
                with open(self.config.tmp_data / 'validation_triples_ids.pkl', 'wb') as f:
                    pickle.dump(self.validation_triples_ids, f)
            else:
                with open(self.config.tmp_data / 'validation_triples_ids.pkl', 'rb') as f:
                    self.validation_triples_ids = pickle.load(f)

            self.data_stats.tot_valid_triples = len(self.validation_triples_ids)

            if self.algo.lower().startswith('proje'):
                self.hr_t_ids_train = defaultdict(set)
                self.tr_h_ids_train = defaultdict(set)

                if not os.path.exists(self.config.tmp_data / 'hr_t_ids_train.pkl'):
                    for t in self.train_triples_ids:
                        self.hr_t_ids_train[(t.h, t.r)].add(t.t)
                        self.tr_h_ids_train[(t.t, t.r)].add(t.h)
                    with open(self.config.tmp_data / 'hr_t_ids_train.pkl', 'wb') as f:
                        pickle.dump(self.hr_t_ids_train, f)
                    with open(self.config.tmp_data / 'tr_h_ids_train.pkl', 'wb') as f:
                        pickle.dump(self.tr_h_ids_train, f)

            if self.sampling == "bern":
                import pdb
                pdb.set_trace()
                self.relation_property_head = {x: [] for x in range(self.tot_relation)}
                self.relation_property_tail = {x: [] for x in
                                               range(self.tot_relation)}
                for t in self.train_triples_ids:
                    self.relation_property_head[t.r].append(t.h)
                    self.relation_property_tail[t.r].append(t.t)

                self.relation_property = {x: (len(set(self.relation_property_tail[x]))) / (
                        len(set(self.relation_property_head[x])) + len(set(self.relation_property_tail[x]))) \
                                          for x in
                                          self.relation_property_head.keys()}
                with open(self.config.tmp_data / 'relation_property.pkl', 'wb') as f:
                    pickle.dump(self.relation_property, f)

        else:
            raise NotImplementedError("Data preparation is not implemented for algorithm:", self.algo)

        if not os.path.exists(self.config.tmp_data / 'hr_t.pkl'):
            for t in self.train_triples:
                self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
                self.tr_h[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])

            for t in self.test_triples:
                self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
                self.tr_h[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])

            for t in self.validation_triples:
                self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
                self.tr_h[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])

            with open(self.config.tmp_data / 'hr_t.pkl', 'wb') as f:
                pickle.dump(self.hr_t, f)
            with open(self.config.tmp_data / 'tr_h.pkl', 'wb') as f:
                pickle.dump(self.tr_h, f)
        else:
            with open(self.config.tmp_data / 'hr_t.pkl', 'rb') as f:
                self.hr_t = pickle.load(f)
            with open(self.config.tmp_data / 'tr_h.pkl', 'rb') as f:
                self.tr_h = pickle.load(f)

        if not os.path.exists(self.config.tmp_data / 'data_stats.pkl'):
            with open(self.config.tmp_data / 'data_stats.pkl', 'wb') as f:
                pickle.dump(self.data_stats, f)
        else:
            with open(self.config.tmp_data / 'data_stats.pkl', 'rb') as f:
                self.data_stats = pickle.load(f)

        self.tot_triple = self.data_stats.tot_triple
        self.tot_entity = self.data_stats.tot_entity
        self.tot_relation = self.data_stats.tot_relation

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

            self.data_stats.tot_entity = self.tot_entity
            self.data_stats.tot_relation = self.tot_relation
            self.data_stats.tot_triple = self.tot_triple
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

        self.data_stats.tot_entity = self.tot_entity
        self.data_stats.tot_relation = self.tot_relation
        self.data_stats.tot_triple = self.tot_triple

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


if __name__ == '__main__':
    # test_data_prep()
    pass
