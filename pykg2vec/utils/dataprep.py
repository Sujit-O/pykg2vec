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


class DataInputSimple(object):
    def __init__(self, h=None, r=None, t=None, hr_t=None, rt_h=None):
        self.h = h
        self.r = r
        self.t = t
        self.hr_t = hr_t
        self.rt_h = rt_h


class KGMetaData(object):
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
        self.tot_entity = tot_entity


class DataPrep(object):

    def __init__(self, name_dataset='Freebase15k', sampling="uniform", algo='ConvE'):
        '''store the information of database'''

        self.data_stats = KGMetaData()

        self.config = GlobalConfig(dataset=name_dataset)
        
        self.algo = algo
        self.sampling = sampling
        self.train_triples = []
        self.test_triples = []
        self.validation_triples = []

        self.entity2idx = {}
        self.idx2entity = {}

        self.relation2idx = {}
        self.idx2relation = {}

        self.hr_t = defaultdict(set)
        self.tr_h = defaultdict(set)

    def init_variables(self):
        self.train_triples = None
        self.test_triples = None
        self.validation_triples = None
        self.entities = None 
        self.relations = None

        self.entity2idx = None
        self.idx2entity = None
        self.relation2idx = None
        self.idx2relation = None

        self.test_triples_ids = None
        self.train_triples_ids = None
        self.validation_triples_ids = None

        self.hr_t = None
        self.tr_h = None

        self.hr_t_train = None
        self.tr_h_train = None

    def prepare_data(self):
        '''the ways to prepare data are different across algorithms.'''
        tucker_series = ["tucker"]
        conve_series = ["conve", "complex", "distmult"]
        other_algorithms = ['transe', 'transr', 'transh', 'transd', 'transm', \
        'kg2e', 'proje', 'rescal','slm', 'sme_bilinear', 'sme_linear', 'ntn', 'rotate']

        if self.algo.lower() in tucker_series:
            self.init_variables()
            self.prepare_data_tucker()
        
        elif self.algo.lower() in conve_series:
            self.init_variables()
            self.prepare_data_conve()
        
        elif self.algo.lower() in other_algorithms:
            self.init_variables()
            
            self.entity2idx   = self.get_entity2idx()
            self.idx2entity   = self.get_idx2entity()
            self.relation2idx = self.get_relation2idx() 
            self.idx2relation = self.get_idx2relation()
            self.test_triples_ids       = self.get_test_triples_ids()
            self.train_triples_ids      = self.get_train_triples_ids()
            self.validation_triples_ids = self.get_validation_triples_ids()
            
            self.hr_t = self.get_hr_t(train_only=False)
            self.tr_h = self.get_tr_h(train_only=False)

            if self.algo.lower().startswith('proje'):

                self.hr_t_train = self.get_hr_t(train_only=True)
                self.tr_h_train = self.get_tr_h(train_only=True) 
        
            self.backup_metadata()
        else:
            raise NotImplementedError("Data preparation is not implemented for algorithm:", self.algo)

    def prepare_data_tucker(self):
        
        self.train_graph_rt = {}
        self.train_graph_hr = {}
        self.label_graph_rt = {}
        self.label_graph_hr = {}
        self.train_data = []
        self.test_data = []
        self.valid_data = []

        self.read_triple_hr_rt_simple(['train', 'test', 'valid'])
        self.calculate_mapping()

        if not (self.config.tmp_data / 'train_data.pkl').exists():
            print("\nPreparing Training Data!")
            with progressbar.ProgressBar(max_value=len(self.train_triples)) as bar:
                for i, t in enumerate(self.train_triples):
                    h_idx = self.entity2idx[t.h]
                    r_idx = self.relation2idx[t.r]
                    t_idx = self.entity2idx[t.t]
                    hr_t_idxs = [self.entity2idx[i] for i in list(self.train_graph_hr[(t.h, t.r)])]
                    rt_h_idxs = [self.entity2idx[i] for i in list(self.train_graph_rt[(t.r, t.t)])]
                    self.train_data.append(DataInputSimple(h=h_idx, r=r_idx, t=t_idx, hr_t=hr_t_idxs, rt_h=rt_h_idxs))
                    bar.update(i)
            with open(str(self.config.tmp_data / 'train_data.pkl'), 'wb') as f:
                pickle.dump(self.train_data, f)
        else:
            with open(str(self.config.tmp_data / 'train_data.pkl'), 'rb') as f:
                self.train_data = pickle.load(f)
        self.data_stats.tot_train_triples = len(self.train_data)

        if not (self.config.tmp_data / 'test_data.pkl').exists():
            print("\nPreparing Testing Data!")
            with progressbar.ProgressBar(max_value=len(self.test_triples)) as bar:
                for i, t in enumerate(self.test_triples):
                    h_idx = self.entity2idx[t.h]
                    r_idx = self.relation2idx[t.r]
                    t_idx = self.entity2idx[t.t]
                    self.test_data.append(DataInputSimple(h=h_idx, r=r_idx, t=t_idx))
                    bar.update(i)
            with open(str(self.config.tmp_data / 'test_data.pkl'), 'wb') as f:
                pickle.dump(self.test_data, f)
        else:
            with open(str(self.config.tmp_data / 'test_data.pkl'), 'rb') as f:
                self.test_data = pickle.load(f)
        self.data_stats.tot_test_triples = len(self.test_data)

        if not (self.config.tmp_data / 'valid_data.pkl').exists():
            print("\nPreparing Validation Data!")
            with progressbar.ProgressBar(max_value=len(self.validation_triples)) as bar:
                for i, t in enumerate(self.validation_triples):
                    h_idx = self.entity2idx[t.h]
                    r_idx = self.relation2idx[t.r]
                    t_idx = self.entity2idx[t.t]
                    self.valid_data.append(DataInputSimple(h=h_idx, r=r_idx, t=t_idx))
                    bar.update(i)
            with open(str(self.config.tmp_data / 'valid_data.pkl'), 'wb') as f:
                pickle.dump(self.valid_data, f)
        else:
            with open(str(self.config.tmp_data / 'valid_data.pkl'), 'rb') as f:
                self.valid_data = pickle.load(f)
        self.data_stats.tot_valid_triples = len(self.valid_data)

        self.validation_triples_ids = [
            Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t])
            for t
            in self.validation_triples]

    def prepare_data_conve(self):

        self.label_graph = {}
        self.train_graph = {}
        self.train_data = []
        self.test_data = []
        self.valid_data = []
        self.test_triples_no_rev = []
        self.validation_triples_no_rev = []

        self.read_triple_hr_rt(['train', 'test', 'valid'])
        self.calculate_mapping()

        if not (self.config.tmp_data / 'train_data.pkl').exists():
            print("\nPreparing Training Data!")
            with progressbar.ProgressBar(max_value=len(self.train_graph)) as bar:
                for i, (e, r) in enumerate(self.train_graph):
                    e1_idx = self.entity2idx[e]
                    r_idx = self.relation2idx[r]
                    e2_multi1 = [self.entity2idx[i] for i in list(self.train_graph[(e, r)])]
                    self.train_data.append(DataInput(e1=e1_idx, r=r_idx, e2_multi1=e2_multi1))
                    bar.update(i)
            with open(str(self.config.tmp_data / 'train_data.pkl'), 'wb') as f:
                pickle.dump(self.train_data, f)
        else:
            with open(str(self.config.tmp_data / 'train_data.pkl'), 'rb') as f:
                self.train_data = pickle.load(f)
        self.data_stats.tot_train_triples = len(self.train_data)

        if not (self.config.tmp_data / 'test_data.pkl').exists():
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
            with open(str(self.config.tmp_data / 'test_data.pkl'), 'wb') as f:
                pickle.dump(self.test_data, f)
        else:
            with open(str(self.config.tmp_data / 'test_data.pkl'), 'rb') as f:
                self.test_data = pickle.load(f)
        self.data_stats.tot_test_triples = len(self.test_data)

        if not (self.config.tmp_data / 'valid_data.pkl').exists():
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
            with open(str(self.config.tmp_data / 'valid_data.pkl'), 'wb') as f:
                pickle.dump(self.valid_data, f)
        else:
            with open(str(self.config.tmp_data / 'valid_data.pkl'), 'rb') as f:
                self.valid_data = pickle.load(f)
        self.data_stats.tot_valid_triples = len(self.valid_data)

        self.validation_triples_ids = [
            Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t])
            for t
            in self.validation_triples]
    
    def read_train_triples(self):
        if self.train_triples is None: 
            train_triples = self.config.read_triplets('train')
            return train_triples
        return self.train_triples

    def read_test_triples(self):
        if self.test_triples is None:
            test_triples = self.config.read_triplets('test')
            return test_triples
        return self.test_triples

    def read_valid_triples(self):
        if self.validation_triples is None:
            validation_triples = self.config.read_triplets('valid')
            return validation_triples
        return self.validation_triples

    def get_entities(self):
        if self.entities is not None:
            return self.entities

        entities  = set()

        all_triples = self.read_train_triples() + self.read_test_triples() + self.read_valid_triples()
            
        for triplet in all_triples:
            entities.add(triplet.h)
            entities.add(triplet.t)
        
        return np.sort(list(entities))

    def get_relations(self):
        if self.relations is not None:
            return self.relations

        relations  = set()

        all_triples = self.read_train_triples() + self.read_test_triples() + self.read_valid_triples()
            
        for triplet in all_triples:
            relations.add(triplet.r)
        
        return np.sort(list(relations))

    def prepare_data_others(self):

        self.entity2idx   = self.get_entity2idx()
        self.idx2entity   = self.get_idx2entity()
        self.relation2idx = self.get_relation2idx() 
        self.idx2relation = self.get_idx2relation()
        self.test_triples_ids       = self.get_test_triples_ids()
        self.train_triples_ids      = self.get_train_triples_ids()
        self.validation_triples_ids = self.get_validation_triples_ids()

        self.backup_metadata()
        
        # if self.algo.lower().startswith('proje'):
        #     self.hr_t_ids_train = defaultdict(set)
        #     self.tr_h_ids_train = defaultdict(set)

        #     if (self.config.tmp_data / 'hr_t_ids_train.pkl').exists():
        #         for t in self.train_triples_ids:
        #             self.hr_t_ids_train[(t.h, t.r)].add(t.t)
        #             self.tr_h_ids_train[(t.t, t.r)].add(t.h)
        #         with open(str(self.config.tmp_data / 'hr_t_ids_train.pkl'), 'wb') as f:
        #             pickle.dump(self.hr_t_ids_train, f)
        #         with open(str(self.config.tmp_data / 'tr_h_ids_train.pkl'), 'wb') as f:
        #             pickle.dump(self.tr_h_ids_train, f)

        # if self.sampling == "bern":
        #     import pdb
        #     pdb.set_trace()
        #     self.relation_property_head = {x: [] for x in range(self.tot_relation)}
        #     self.relation_property_tail = {x: [] for x in
        #                                    range(self.tot_relation)}
        #     for t in self.train_triples_ids:
        #         self.relation_property_head[t.r].append(t.h)
        #         self.relation_property_tail[t.r].append(t.t)

        #     self.relation_property = {x: (len(set(self.relation_property_tail[x]))) / (
        #             len(set(self.relation_property_head[x])) + len(set(self.relation_property_tail[x]))) \
        #                               for x in
        #                               self.relation_property_head.keys()}
        #     with open(str(self.config.tmp_data / 'relation_property.pkl'), 'wb') as f:
        #         pickle.dump(self.relation_property, f)

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

    def read_triple_hr_rt_simple(self, datatype=None):
        print("Reading Triples", datatype)

        for data in datatype:
            with open(str(self.config.dataset.downloaded_path) + data + '.txt', 'r') as f:
                for l in f.readlines():
                    h, r, t = l.split('\t')
                    h = h.strip()
                    r = r.strip()
                    t = t.strip()

                    triple = Triple(h, r, t)

                    if data == 'train':
                        self.train_triples.append(triple)
                        if (h, r) not in self.train_graph_hr:
                            self.train_graph_hr[(h, r)] = set()
                        if (r, t) not in self.train_graph_rt:
                            self.train_graph_rt[(r, t)] = set()

                        self.train_graph_hr[(h, r)].add(t)
                        self.train_graph_rt[(r, t)].add(h)

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

    def get_entity2idx(self):
        ''' entity2idx should be backup in dataset folder'''
        if self.entity2idx: 
            return self.entity2idx

        entity2idx = None 
        entity2idx_path = self.config.dataset.entity2idx_path

        if entity2idx_path.exists():
            with open(str(entity2idx_path), 'rb') as f:
                entity2idx = pickle.load(f)
            return entity2idx

        entity2idx = {v: k for k, v in enumerate(self.get_entities())} ##
       
        with open(str(entity2idx_path), 'wb') as f:
            pickle.dump(entity2idx, f)

        return entity2idx

    def get_idx2entity(self):
        ''' idx2entity should be backup in dataset folder'''
        if self.idx2entity: 
            return self.idx2entity

        idx2entity = None 
        idx2entity_path = self.config.dataset.idx2entity_path

        if idx2entity_path.exists():
            with open(str(idx2entity_path), 'rb') as f:
                idx2entity = pickle.load(f)
            return idx2entity

        idx2entity = {v: k for k, v in self.get_entity2idx().items()} ##
       
        with open(str(idx2entity_path), 'wb') as f:
            pickle.dump(idx2entity, f)

        return idx2entity

    def get_relation2idx(self):
        ''' relation2idx should be backup in dataset folder'''
        if self.relation2idx: 
            return self.relation2idx

        relation2idx = None 
        relation2idx_path = self.config.dataset.relation2idx_path

        if relation2idx_path.exists():
            with open(str(relation2idx_path), 'rb') as f:
                relation2idx = pickle.load(f)
            return relation2idx
       
        relation2idx = {v: k for k, v in enumerate(self.get_relations())} ##
       
        with open(str(relation2idx_path), 'wb') as f:
            pickle.dump(relation2idx, f)

        return relation2idx

    def get_idx2relation(self):
        ''' relation2idx should be backup in dataset folder'''
        if self.idx2relation: 
            return self.idx2relation

        idx2relation = None 
        idx2relation_path = self.config.dataset.idx2relation_path

        if idx2relation_path.exists():
            with open(str(idx2relation_path), 'rb') as f:
                idx2relation = pickle.load(f)
            return idx2relation
        
        idx2relation = {v: k for k, v in self.get_relation2idx().items()} ##

        with open(str(idx2relation_path), 'wb') as f:
            pickle.dump(idx2relation, f)

        return idx2relation

    def get_test_triples_ids(self):
        if self.test_triples_ids: 
            return self.test_triples_ids

        test_triples_ids = [] 
        test_triples_ids_path = self.config.dataset.testing_triples_id_path

        if test_triples_ids_path.exists(): 
            with open(str(test_triples_ids_path), 'rb') as f:
                test_triples_ids = pickle.load(f)
            return test_triples_ids
        e2i = self.get_entity2idx()
        r2i = self.get_relation2idx()
        test_triples_ids = [Triple(e2i[t.h], r2i[t.r], e2i[t.t]) for t in self.read_test_triples()]

        with open(str(test_triples_ids_path), 'wb') as f:
            pickle.dump(test_triples_ids, f)
        
        return test_triples_ids

    def get_train_triples_ids(self):
        if self.train_triples_ids: 
            return self.train_triples_ids

        train_triples_ids = [] 
        train_triples_ids_path = self.config.dataset.training_triples_id_path

        if train_triples_ids_path.exists(): 
            with open(str(train_triples_ids_path), 'rb') as f:
                train_triples_ids = pickle.load(f)
            return train_triples_ids

        train_triples_ids = [Triple(self.get_entity2idx()[t.h], self.get_relation2idx()[t.r],
                                   self.get_entity2idx()[t.t]) for t in self.read_train_triples()]

        with open(str(train_triples_ids_path), 'wb') as f:
            pickle.dump(train_triples_ids, f)
        
        return train_triples_ids

    def get_validation_triples_ids(self):
        if self.validation_triples_ids: 
            return self.validation_triples_ids

        validation_triples_ids = [] 
        validation_triples_ids_path = self.config.dataset.validating_triples_id_path

        if validation_triples_ids_path.exists(): 
            with open(str(validation_triples_ids_path), 'rb') as f:
                validation_triples_ids = pickle.load(f)
            return validation_triples_ids

        validation_triples_ids = [Triple(self.get_entity2idx()[t.h], self.get_relation2idx()[t.r],
                                   self.get_entity2idx()[t.t]) for t in self.read_valid_triples()]

        with open(str(validation_triples_ids_path), 'wb') as f:
            pickle.dump(validation_triples_ids, f)
        
        return validation_triples_ids
    
    def get_hr_t(self, train_only=False):
        if train_only:
            if self.hr_t_train is not None:
                return self.hr_t_train
        else:
            if self.hr_t is not None:
                return self.hr_t

        hr_t = defaultdict(set)
        
        if train_only:
            hrt_path = self.config.dataset.hrt_train_path
        else:
            hrt_path = self.config.dataset.hrt_path

        if hrt_path.exists():
            with open(str(hrt_path), 'rb') as f:
                hr_t = pickle.load(f)
            return hr_t

        for t in self.get_test_triples_ids():
            hr_t[(t.h, t.r)].add(t.t)

        if not train_only:
            for t in self.get_train_triples_ids():
                hr_t[(t.h, t.r)].add(t.t)

            for t in self.get_validation_triples_ids():
                hr_t[(t.h, t.r)].add(t.t)

        with open(str(hrt_path), 'wb') as f:
            pickle.dump(hr_t, f)

        return hr_t

    def get_tr_h(self, train_only=False):
        if train_only:
            if self.tr_h_train is not None:
                return self.tr_h_train
        else:
            if self.tr_h is not None:
                return self.tr_h

        tr_h = defaultdict(set)
        if train_only:
            trh_path = self.config.dataset.trh_train_path
        else:
            trh_path = self.config.dataset.trh_path

        if trh_path.exists():
            with open(str(trh_path), 'rb') as f:
                tr_h = pickle.load(f) 
            return tr_h

        for t in self.get_test_triples_ids():
            tr_h[(t.t, t.r)].add(t.h)

        if not train_only:
            for t in self.get_train_triples_ids():
                tr_h[(t.t, t.r)].add(t.h)

            for t in self.get_validation_triples_ids():
                tr_h[(t.t, t.r)].add(t.h)

        with open(str(trh_path), 'wb') as f:
            pickle.dump(tr_h, f)

        return tr_h

    def backup_metadata(self):
        kg_meta = self.data_stats

        kg_meta.tot_entity   = len(self.entity2idx)
        kg_meta.tot_relation = len(self.relation2idx)
        kg_meta.tot_test_triples = len(self.test_triples_ids)
        kg_meta.tot_train_triples = len(self.train_triples_ids)
        kg_meta.tot_valid_triples = len(self.validation_triples_ids)
        kg_meta.tot_triple   = kg_meta.tot_test_triples + kg_meta.tot_train_triples + kg_meta.tot_valid_triples

        with open(str(self.config.dataset.metadata_path), 'wb') as f:
            pickle.dump(self.data_stats, f)

    def dump(self):
        ''' dump key information'''
        print("\n----------Relation to Indexes--------------")
        pprint.pprint(self.relation2idx)
        print("---------------------------------------------")

        print("\n----------Relation to Indexes--------------")
        pprint.pprint(self.idx2relation)
        print("---------------------------------------------")

        print("\n----------Train Triple Stats---------------")
        print("Total Training Triples   :", len(self.train_triples_ids))
        print("Total Testing Triples    :", len(self.test_triples_ids))
        print("Total validation Triples :", len(self.validation_triples_ids))
        print("Total Entities           :", self.data_stats.tot_entity)
        print("Total Relations          :", self.data_stats.tot_relation)
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
    data_handler = DataPrep('Freebase15k', sampling="uniform", algo='transe')
    data_handler.prepare_data()
    data_handler.dump()


if __name__ == '__main__':
    test_data_prep()