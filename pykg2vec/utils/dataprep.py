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
    def __init__(self, head=None,relation=None,tail=None):
        self.h = head
        self.r = relation
        self.t = tail

class DataPrep(object):
    
    def __init__(self, name_dataset='Freebase15k'):

        '''store the information of database'''
        self.config = GlobalConfig(dataset=name_dataset)
        
        self.train_triples      = []
        self.test_triples       = []
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
        
        self.relation_property_head = None
        self.relation_property_tail = None
        self.relation_property = None

        self.read_triple(['train','test','valid']) #TODO: save the triples to prevent parsing everytime
        self.calculate_mapping() # from entity and relation to indexes.

        self.test_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t]) for t in self.test_triples]
        self.train_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t]) for t in self.train_triples]
        self.validation_triples_ids = [Triple(self.entity2idx[t.h], self.relation2idx[t.r], self.entity2idx[t.t]) for t in self.validation_triples]

        if self.config.negative_sample =='bern':
            self.negative_sampling()

    def read_triple(self, datatype=None):
        print("Reading Triples",datatype)

        for data in datatype:
            with open( str(self.config.dataset.downloaded_path)+data+'.txt','r') as f:
                lines=f.readlines()
                for l in lines:
                    h, r, t = l.split('\t')
                    h = h.split(' ')[0].strip()
                    r = r.split(' ')[0].strip()
                    t = t.split(' ')[0].strip()
                    triple = Triple(h,r,t)
                    if data == 'train':
                        self.train_triples.append(triple)
                    elif data =='test':
                        self.test_triples.append(triple)
                    elif data == 'valid':
                        self.validation_triples.append(triple)
                    else:
                        continue

    def calculate_mapping(self):
        print("Calculating entity2idx & idx2entity & relation2idx & idx2relation.")

        if self.config.dataset.entity2idx_path.is_file():
            with open(self.config.dataset.entity2idx_path, 'rb') as f:
                self.entity2idx = pickle.load(f)

            with open(self.config.dataset.idx2entity_path, 'rb') as f:
                self.idx2entity = pickle.load(f)

            with open(self.config.dataset.relation2idx_path, 'rb') as f:
                self.relation2idx = pickle.load(f)

            with open(self.config.dataset.idx2relation_path, 'rb') as f:
                self.idx2relation = pickle.load(f)

            self.tot_entity=len(self.entity2idx)
            self.tot_relation = len(self.relation2idx)
            self.tot_triple = len(self.train_triples) + \
                              len(self.test_triples)+\
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
        self.tot_relation =  len(relations)

        self.entity2idx = {v:k for k,v in enumerate(entities)}
        self.idx2entity = {v:k for k, v in self.entity2idx.items()}

        self.relation2idx = {v: k for k, v in enumerate(relations)}
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}

        # pdb.set_trace()
        #save entity2idx
        if not os.path.isfile(self.config.dataset.entity2idx_path):
            with open(self.config.dataset.entity2idx_path, 'wb') as f:
                pickle.dump(self.entity2idx, f)
        #save idx2entity
        if not os.path.isfile(self.config.dataset.idx2entity_path):
            with open(self.config.dataset.idx2entity_path, 'wb') as f:
                pickle.dump(self.idx2entity, f)
        #save relation2idx
        if not os.path.isfile(self.config.dataset.relation2idx_path):
            with open(self.config.dataset.relation2idx_path, 'wb') as f:
                pickle.dump(self.relation2idx, f)
        #save idx2relation
        if not os.path.isfile(self.config.dataset.idx2relation_path):
            with open(self.config.dataset.idx2relation_path, 'wb') as f:
                pickle.dump(self.idx2relation, f)

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

    def batch_generator_train(self, batch=128):
        pos_triples_hm = {}
        neg_triples_hm = {}

        for t in self.train_triples_ids:
            pos_triples_hm[(t.h,t.r,t.t)] = 1

        rand_ids = np.random.permutation(len(self.train_triples_ids))
        number_of_batches = len(self.train_triples_ids) // batch
        print("Number of batches:", number_of_batches)

        counter = 0

        while True:
            pos_triples = np.asarray([[self.train_triples_ids[x].h,
                                       self.train_triples_ids[x].r,
                                       self.train_triples_ids[x].t] for x in rand_ids[batch*counter:batch*(counter + 1)]])
            # print("triples:",pos_triples)
            ph = pos_triples[:, 0]
            pr = pos_triples[:, 1]
            pt = pos_triples[:, 2]
            nh = []
            nr = []
            nt = []
            for t in pos_triples:
                # print("r:",t[1])
                if self.config.negative_sample == 'uniform':
                    prob = 0.5
                elif self.config.negative_sample == 'bern':
                    prob = self.relation_property[t[1]]
                else:
                    raise NotImplementedError("%s sampling not supported!" % self.config.negative_sample)
                last_h=0
                last_r=0
                last_t=0
                if np.random.random()>prob:
                    idx = np.random.randint(self.tot_entity)
                    break_cnt = 0
                    flag = False

                    while ((t[0], t[1], idx) in pos_triples_hm
                           or (t[0], t[1], idx) in neg_triples_hm):
                        idx = np.random.randint(self.tot_entity)
                        break_cnt += 1
                        if break_cnt >= 100:
                            flag =True
                            break
                    if flag:
                        nh.append(last_h)
                        nr.append(last_r)
                        nt.append(last_t)
                    else:
                        nh.append(t[0])
                        nr.append(t[1])
                        nt.append(idx)
                        last_h=t[0]
                        last_r=t[1]
                        last_t=idx
                        neg_triples_hm[(t[0],t[1],idx)] = 1
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
    
    def dump(self):
        ''' dump key information'''
        print("\n----------Relation to Indexes---------------")
        pprint.pprint(self.relation2idx)
        print("---------------------------------------------")
        
        print("\n----------Relation to Indexes---------------")
        pprint.pprint(self.idx2relation)
        print("---------------------------------------------")

        print("\n----------Train Triple Stats---------------")
        print("Total Training Triples   :", len(self.train_triples))
        print("Total Testing Triples    :", len(self.test_triples))
        print("Total validation Triples :", len(self.validation_triples))
        print("Total Training Triples   :", len(self.train_triples_ids), "(from train_triples_ids)")
        print("Total Testing Triples    :", len(self.test_triples_ids), "(from test_triples_ids)")
        print("Total validation Triples :", len(self.validation_triples_ids), "(from validation_triples_ids)")
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
    gen = data_handler.batch_generator_train(batch=8)
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
    


"""
    def prepare_data(self):

        unseen_entities = []
        removed_triples = []

        pos_triples_cnt = 0
        neg_triples_cnt = 0
        for data in ['train', 'valid', 'test']:
            if os.path.isfile(self.config.dataset.prepared_data_path + '%s_head_pos.pkl' % data) \
                    and os.path.isfile(self.config.dataset.prepared_data_path + '%s_head_neg.pkl' % data):
                return
            with open(self.config.dataset.downloaded_path + "%s.txt" % data, 'r') as f:
                lines = f.readlines()

                head_list = []
                rel_list  = []
                tail_list = []

                head_list_neg = []
                rel_list_neg  = []
                tail_list_neg = []
                pos_triples = {}
                neg_triples = {}
                print("\nProcessing: %s dataset"%data)
                with progressbar.ProgressBar(max_value=len(lines)) as bar:
                    for i,line in enumerate(lines):
                        triple = parse_line(line)
                        if triple.h in self.entity2idx and triple.t in self.entity2idx and triple.r in self.entity2idx:
                            head_list.append(self.entity2idx[triple.h])
                            rel_list.append(self.entity2idx[triple.r])
                            tail_list.append(self.entity2idx[triple.t])

                            pos_triples[(self.entity2idx[triple.h],
                                           self.entity2idx[triple.r],
                                           self.entity2idx[triple.t])] = 1
                        else:
                            if triple.h in self.entity2idx:
                                unseen_entities += [triple.h]
                            if triple.r in self.entity2idx:
                                unseen_entities += [triple.r]
                            if triple.t in self.entity2idx:
                                unseen_entities += [triple.t]
                            removed_triples += [line]
                        bar.update(i)

                    print("\npos_heads:",head_list[:5])
                    print("pos_tails:",tail_list[:5])
                    print("pos_rels:",rel_list[:5])

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_pos.pkl'% data):
                        with open(self.config.dataset.prepared_data_path+'%s_head_pos.pkl'% data, 'wb') as g:
                            pickle.dump(head_list, g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_tail_pos.pkl'% data):
                        with open(self.config.dataset.prepared_data_path+'%s_tail_pos.pkl'% data, 'wb') as g:
                            pickle.dump(tail_list, g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_rel_pos.pkl' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_rel_pos.pkl' % data, 'wb') as g:
                            pickle.dump(rel_list, g)

                pos_triples_cnt+=len(pos_triples)

                print("\nGenerating Corrupted Data: %s dataset" % data)
                #TODO: Check when (train, test, validate) and how much to corrupt the data
                with progressbar.ProgressBar(max_value=len(lines)) as bar:
                    for i, triple in enumerate(list(pos_triples.keys())):
                        # rand_num=randint(0,900)
                        corrupt_head_prob = np.random.binomial(1, 0.5)
                        if corrupt_head_prob:
                            # Corrupt Tail
                            idx = choice(tail_list)
                            break_cnt=0
                            flag=False
                            while ((triple[0], triple[1], idx) in pos_triples
                                   or (triple[0], triple[1], idx) in neg_triples):
                                idx = choice(tail_list)
                                break_cnt+=1
                                if break_cnt>=100:
                                    flag=True
                                    break
                            if flag:
                                continue

                            head_list_neg.append(triple[0])
                            rel_list_neg.append(triple[1])
                            tail_list_neg.append(idx)
                            neg_triples[(triple[0],
                                           triple[1],
                                           idx)] = 1

                        else:
                            #Corrupt Head
                            idx = choice(head_list)
                            break_cnt = 0
                            flag = False
                            while ((idx, triple[1], triple[2]) in pos_triples or
                            (idx, triple[1], triple[2]) in neg_triples):
                                idx = choice(head_list)
                                break_cnt += 1
                                if break_cnt >= 100:
                                    flag = True
                                    break
                            if flag:
                                continue

                            head_list_neg.append(idx)
                            rel_list_neg.append(triple[1])
                            tail_list_neg.append(triple[2])
                            neg_triples[(idx,
                                             triple[1],
                                             triple[2])] = 1
                        # else:
                        #     #Corrupt relation
                        #     idx = choice(rel_list)
                        #     break_cnt = 0
                        #     flag = False
                        #     while (triple[0], idx, triple[2]) in pos_triples or\
                        #            (triple[0], idx, triple[2]) in neg_triples:
                        #         idx = choice(rel_list)
                        #         break_cnt += 1
                        #         if break_cnt >= 100:
                        #             flag = True
                        #             break
                        #     if flag:
                        #         continue
                        #
                        #     head_list_neg.append(triple[0])
                        #     rel_list_neg.append(idx)
                        #     tail_list_neg.append(triple[2])
                        #     neg_triples[(triple[0],
                        #                      idx,
                        #                      triple[2])] = 1

                        bar.update(i)

                    print("\nneg_heads:", head_list_neg[:5])
                    print("neg_tails:", tail_list_neg[:5])
                    print("neg_rels:", rel_list_neg[:5])

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_neg.pkl' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_head_neg.pkl' % data, 'wb') as g:
                            pickle.dump(head_list_neg, g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_tail_neg.pkl' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_tail_neg.pkl' % data, 'wb') as g:
                            pickle.dump(tail_list_neg, g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_rel_neg.pkl' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_rel_neg.pkl' % data, 'wb') as g:
                            pickle.dump(rel_list_neg, g)

                    neg_triples_cnt+=len(neg_triples)

        print("\n----------Data Prep Results--------------")
        print("Total Positive Triples :", pos_triples_cnt)
        print("Total Negative Triples :", neg_triples_cnt)
        print("Total Unseen Triples   :", len(list(set(removed_triples))))
        print("Total Unseen Entities  :", len(list(set(unseen_entities))))
        print('-------------------------------------------')
"""