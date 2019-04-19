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

class Triple(object):
    def __init__(self, head=None, relation=None, tail=None):
        self.h = head
        self.r = relation
        self.t = tail

class DataPrep(object):

    def __init__(self, name_dataset='Freebase15k'):
        '''store the information of database'''
        self.config = GlobalConfig(dataset=name_dataset)

        self.train_triples = []
        self.test_triples = []
        self.validation_triples = []
        
        self.tot_relation = 0
        self.tot_triple   = 0
        self.tot_entity   = 0

        self.entity2idx   = {}
        self.idx2entity   = {}

        self.relation2idx = {}
        self.idx2relation = {}

        self.hr_t = defaultdict(set)
        self.tr_t = defaultdict(set)      

        self.read_triple(['train','test','valid']) #TODO: save the triples to prevent parsing everytime
        self.calculate_mapping() # from entity and relation to indexes.

        self.test_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t]) for t in self.test_triples]
        self.train_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t]) for t in self.train_triples]
        self.validation_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t]) for t in self.validation_triples]

        for t in self.test_triples:
            self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
            self.tr_t[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])
        for t in self.train_triples:
            self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
            self.tr_t[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])
        for t in self.validation_triples:
            self.hr_t[(self.entity2idx[t.h], self.relation2idx[t.r])].add(self.entity2idx[t.t])
            self.tr_t[(self.entity2idx[t.t], self.relation2idx[t.r])].add(self.entity2idx[t.h])

        if self.config.negative_sample =='bern':          
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

    def batch_generator_train(self, src_triples=None, batch_size=128):

        batch_size = batch_size // 2
        
        if src_triples is None:
            #TODO: add parameter for specifying the source of triple
            src_triples = self.train_triples_ids

        observed_triples = {(t.h, t.r, t.t): 1 for t in src_triples} 
        # 1 as positive, 0 as negative

        array_rand_ids = np.random.permutation(len(src_triples))
        number_of_batches = len(src_triples) // batch_size

        # print("Number of batches:", number_of_batches)

        batch_idx = 0
        last_h=0
        last_r=0
        last_t=0

        while True:
            
            pos_triples = np.asarray([[src_triples[x].h,
                                       src_triples[x].r,
                                       src_triples[x].t] for x in array_rand_ids[batch_size*batch_idx:batch_size*(batch_idx+1)]])
            
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

                if np.random.random()>prob:
                    idx_replace_tail = np.random.randint(self.tot_entity)

                    break_cnt = 0
                    while ((t[0], t[1], idx_replace_tail) in observed_triples
                           or (t[0], t[1], idx_replace_tail) in observed_triples):
                        idx_replace_tail = np.random.randint(self.tot_entity)
                        break_cnt += 1
                        if break_cnt >= 100:
                            break

                    if break_cnt >= 100: # can not find new negative triple.
                        nh.append(last_h)
                        nr.append(last_r)
                        nt.append(last_t)
                    else:
                        nh.append(t[0])
                        nr.append(t[1])
                        nt.append(idx_replace_tail)
                        last_h=t[0]
                        last_r=t[1]
                        last_t=idx_replace_tail

                        observed_triples[(t[0],t[1],idx_replace_tail)] = 0

                else:
                    idx_replace_head = np.random.randint(self.tot_entity)
                    break_cnt = 0
                    while ((idx_replace_head, t[1], t[2]) in observed_triples
                           or (idx_replace_head, t[1], t[2]) in observed_triples):
                        idx_replace_head = np.random.randint(self.tot_entity)
                        break_cnt += 1
                        if break_cnt >= 100:
                            break

                    if break_cnt >= 100: # can not find new negative triple.
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
        print("Reading Triples",datatype)

        for data in datatype:
            with open(str(self.config.dataset.downloaded_path)+data+'.txt','r') as f:
                for l in f.readlines():
                    h, r, t = l.split('\t')
                    triple = Triple(h.strip(),r.strip(),t.strip())

                    if data == 'train':
                        self.train_triples.append(triple)
                    elif data == 'test':
                        self.test_triples.append(triple)
                    elif data == 'valid':
                        self.validation_triples.append(triple)
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
            print(idx, triple.h,triple.r,triple.t)
        for idx, triple in enumerate(self.test_triples):
            print(idx, triple.h,triple.r,triple.t)
        for idx, triple in enumerate(self.validation_triples):
            print(idx, triple.h,triple.r,triple.t)

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

if __name__=='__main__':
    # test_data_prep()
    test_data_prep_generator()
    
