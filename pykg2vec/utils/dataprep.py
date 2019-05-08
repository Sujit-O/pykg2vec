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

from config.global_config import GlobalConfig, Triple, DataInputSimple
import numpy as np
import pickle
from collections import defaultdict
import pprint

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
        self.config = GlobalConfig(dataset=name_dataset)
        
        self.algo = algo
        self.sampling = sampling
        
    def init_variables(self):

        self.data_stats = KGMetaData()

        self.train_data = None
        self.test_data = None
        self.valid_data = None

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

        self.relation_property = None

    def prepare_data(self):
        '''the ways to prepare data are different across algorithms.'''
        tucker_series = ["tucker", "conve", "complex", "distmult"]
        other_algorithms = ['transe', 'transr', 'transh', 'transd', 'transm', \
        'kg2e', 'proje', 'rescal','slm', 'sme_bilinear', 'sme_linear', 'ntn', 'rotate']

        # check if the algorithm is out of the list of supporting algorithms. 
        if not self.algo.lower() in tucker_series and not self.algo.lower() in other_algorithms:
            raise NotImplementedError("Data preparation is not implemented for algorithm:", self.algo)

        self.init_variables()
        
        self.entities               = self.get_entities()
        self.relations              = self.get_relations()
        self.entity2idx             = self.get_entity2idx()
        self.idx2entity             = self.get_idx2entity()
        self.relation2idx           = self.get_relation2idx() 
        self.idx2relation           = self.get_idx2relation()
        self.test_triples_ids       = self.get_test_triples_ids()
        self.train_triples_ids      = self.get_train_triples_ids()
        self.validation_triples_ids = self.get_validation_triples_ids()
        
        self.hr_t = self.get_hr_t(train_only=False)
        self.tr_h = self.get_tr_h(train_only=False)

        if self.algo.lower().startswith('proje'):
            self.hr_t_train = self.get_hr_t(train_only=True)
            self.tr_h_train = self.get_tr_h(train_only=True) 
    
        if self.sampling == "bern":
            self.relation_property = self.get_relation_property()

        if self.algo.lower() in tucker_series:
            self.train_data = self.get_train_data()
            self.test_data = self.test_triples_ids
            self.valid_data = self.validation_triples_ids
        
        self.backup_metadata()
    
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
        if self.train_triples_ids is not None: 
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

    def get_train_data(self):
        if self.train_data is not None: 
            return self.train_data 
        
        train_data = []
        train_data_path = self.config.dataset.hrt_hr_rt_train

        if train_data_path.exists():
            with open(str(train_data_path), 'rb') as f:
                train_data = pickle.load(f)
            return train_data
        
        train_triples_ids = self.get_train_triples_ids()
        hr_t_train = self.get_hr_t(train_only=True)
        tr_h_train = self.get_tr_h(train_only=True)

        with progressbar.ProgressBar(max_value=len(train_triples_ids)) as bar:
            for i, t in enumerate(train_triples_ids):
                hr_t_idxs = list(hr_t_train[(t.h, t.r)])
                rt_h_idxs = list(tr_h_train[(t.r, t.t)])
                train_data.append(DataInputSimple(h=t.h, r=t.r, t=t.t, hr_t=hr_t_idxs, rt_h=rt_h_idxs))
                bar.update(i)
        
        with open(str(train_data_path), 'wb') as f:
            pickle.dump(train_data, f)

        return train_data

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

    def get_relation_property(self):
        if self.relation_property is not None: 
            return self.relation_property
        
        relation_property_path = self.config.dataset.relation_property_path
        relation_property_head = {x: [] for x in range(len(self.get_relations()))}
        relation_property_tail = {x: [] for x in range(len(self.get_relations()))}

        for t in self.get_train_triples_ids():
            relation_property_head[t.r].append(t.h)
            relation_property_tail[t.r].append(t.t)

        relation_property = {x: (len(set(relation_property_tail[x]))) / ( \
                len(set(relation_property_head[x])) + len(set(relation_property_tail[x]))) \
                                  for x in relation_property_head.keys()}

        with open(str(relation_property_path), 'wb') as f:
            pickle.dump(relation_property, f)

        return relation_property

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

def test_data_prep_tucker():
    data_handler = DataPrep('Freebase15k', sampling="uniform", algo='tucker')
    data_handler.prepare_data()
    # data_handler.dump()


if __name__ == '__main__':
    test_data_prep_tucker()