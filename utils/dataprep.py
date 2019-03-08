#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for preparing the data
"""
import numpy as np
from config.config import GlobalConfig
import pickle
import os
import scipy.sparse as sp
import time
import progressbar
import urllib.request
import shutil
import tarfile


class Triple(object):
    def __init__(self, head=None,relation=None,tail=None):
        self.h = head
        self.r = relation
        self.t = tail

# def cmp_head(a, b):
#     return (a.h < b.h) or (a.h == b.h and a.r < b.r) or (a.h == b.h and a.r == b.r and a.t < b.t)
#
# def cmp_tail(a,b):
#     return (a.t < b.t) or (a.t == b.t and a.r < b.r) or (a.t == b.t and a.r == b.r and a.h < b.h)
#
# def minimal(a,b):
#     return a if a<b else b
#
# def cmp_list(a,b):
#     return minimal(a.h, a.t) > minimal(b.h, b.t)

def parse_line(line):
    h, r, t = line.split('\t')
    h = h.split(' ')[0]
    r = r.split(' ')[0]
    t = t.split(' ')[0]
    return Triple(h,r,t)

def extract(tar_path, extract_path='.'):
    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])

class DataPrep(object):
    def __init__(self, conf = None):
        # self.id = conf.id
        # self.batch_h = conf.batch_h
        # self.batch_t = conf.batch_t
        # self.batch_r = conf.batch_r
        # self.batch_y = conf.batch_y
        # self.batchSize  = conf.batchSize
        # self.negRate    = conf.negRate
        # self.negRelRate = conf.negRelRate
        con = GlobalConfig()
        if not os.path.exists('../dataset/Freebase/'):
            os.mkdir('../dataset/Freebase/')

            with urllib.request.urlopen(con.url_FB15) as response, open('../dataset/Freebase/FB15k.tgz', 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            try:
                extract('../dataset/Freebase/FB15k.tgz','../dataset/Freebase/')
            except Exception as e:
                print("Could not extract the tgz file!")
                print(type(e),e.args)

        if conf is None:
            self.path_FB15  = con.path_FB15
        else:
            self.path_FB15  = conf.path

        self.trainTriple = []
        self.testTriple  = []
        self.validTriple = []
        self.entity2idx   = {}
        self.idx2entity   = {}
        self.relation2idx = {}
        self.idx2relation = {}
        self.tot_only_h = 0
        self.tot_only_t = 0
        self.tot_shared = 0
        self.tot_r      = 0
        self.tot_triple = 0

    def read_triple(self, datatype=None):
        if datatype is None:
            datatype = ['train']
        for data in datatype:
            with open(self.path_FB15+data+'.txt','r') as f:
                lines=f.readlines()
                for l in lines:
                    if data == 'train':
                        self.trainTriple.append(parse_line(l))
                    elif data =='test':
                        self.testTriple.append(parse_line(l))
                    elif data == 'valid':
                        self.validTriple.append(parse_line(l))
                    else:
                        continue

    def print_triple(self):
        for triple in self.trainTriple:
            print(triple.h,triple.r,triple.t)
        for triple in self.testTriple:
            print(triple.h,triple.r,triple.t)
        for triple in self.validTriple:
            print(triple.h,triple.r,triple.t)

    def triple_idx_and_stats(self):
        heads = []
        tails = []
        relations = []

        if not self.trainTriple:
            self.read_triple()

        for triple in self.trainTriple:
            heads += [triple.h]
            tails += [triple.t]
            relations += [triple.r]
            self.tot_triple+=1

        only_head = np.sort(list(set(heads) - set(tails)))
        shared    = np.sort(list(set(heads) & set(tails)))
        only_tail = np.sort(list(set(tails) - set(heads)))
        relation_set   = np.sort(list(set(relations)))

        idx = 0
        for i in only_head:
            self.entity2idx[i] = idx
            self.idx2entity[idx] = i
            idx += 1

        self.tot_only_h = idx

        for i in shared:
            self.entity2idx[i] = idx
            self.idx2entity[idx] = i
            idx += 1
        self.tot_shared = idx - self.tot_only_h

        for i in only_tail:
            self.entity2idx[i] = idx
            self.idx2entity[idx] = i
            idx += 1
        self.tot_only_t = idx - (self.tot_shared + self.tot_only_h)

        for i in relation_set:
            self.entity2idx[i] = idx
            self.idx2entity[idx] = i
            idx += 1
        self.tot_r = idx - (self.tot_shared + self.tot_only_h + self.tot_only_t)

        if not os.path.isfile('../dataset/Freebase/FB15k/FB15k_entity2idx.pkl'):
            with open('../dataset/Freebase/FB15k/FB15k_entity2idx.pkl', 'wb') as f:
                pickle.dump(self.entity2idx, f)
        if not os.path.isfile('../dataset/Freebase/FB15k/FB15k_idx2entity.pkl'):
            with open('../dataset/Freebase/FB15k/FB15k_idx2entity.pkl', 'wb') as f:
                pickle.dump(self.idx2entity, f)

        print("\n----------Train Triple Stats---------------")
        print("Total Training Triples   :", self.tot_triple)
        print("Total Head Only Entities :", self.tot_only_h)
        print("Total Tail Only Entities :", self.tot_only_t)
        print("Total Shared Entities    :", self.tot_shared)
        print("Total Relations          :", self.tot_r)
        print("-------------------------------------------")

    def corrupt_head(self):
        if not self.trainTriple:
            self.read_triple()
        #TODO: Corrupt the head for the training data
        pass

    def corrupt_tail(self):
        if not self.trainTriple:
            self.read_triple()
        #TODO: Corrupt the tail for the training data
        pass

    def prepare_data(self):
        if not self.entity2idx:
            if os.path.isfile('../dataset/Freebase/FB15k/FB15k_entity2idx.pkl'):
                with open('../dataset/Freebase/FB15k/FB15k_entity2idx.pkl', 'rb') as f:
                    self.entity2idx=pickle.load(f)
            else:
                self.triple_idx_and_stats()
        unseen_entities = []
        removed_triples = []
        for data in ['train', 'valid', 'test']:
            with open(self.path_FB15 + "%s.txt" % data, 'r') as f:
                lines = f.readlines()

                head_list = sp.lil_matrix((np.max(list(self.entity2idx.values())) + 1, len(lines)),
                                     dtype='float32')
                rel_list = sp.lil_matrix((np.max(list(self.entity2idx.values())) + 1, len(lines)),
                                     dtype='float32')
                tail_list = sp.lil_matrix((np.max(list(self.entity2idx.values())) + 1, len(lines)),
                                     dtype='float32')

                ct = 0
                print("\nProcessing: %s dataset"%data)
                with progressbar.ProgressBar(max_value=len(lines)) as bar:
                    for i,line in enumerate(lines):
                        triple = parse_line(line)
                        if triple.h in self.entity2idx and triple.t in self.entity2idx and triple.r in self.entity2idx:
                            head_list[self.entity2idx[triple.h], ct] = 1
                            rel_list[self.entity2idx[triple.r], ct]  = 1
                            tail_list[self.entity2idx[triple.t], ct] = 1
                            ct += 1
                        else:
                            if triple.h in self.entity2idx:
                                unseen_entities += [triple.h]
                            if triple.r in self.entity2idx:
                                unseen_entities += [triple.r]
                            if triple.t in self.entity2idx:
                                unseen_entities += [triple.t]
                            removed_triples += [line]
                        bar.update(i)

                    if not os.path.isfile('../dataset/Freebase/FB15k/FB15k_%s_head.pkl'% data):
                        with open('../dataset/Freebase/FB15k/FB15k_%s_head.pkl'% data, 'wb') as g:
                            pickle.dump(head_list.tocsr(), g)

                    if not os.path.isfile('../dataset/Freebase/FB15k/FB15k_%s_tail.pkl'% data):
                        with open('../dataset/Freebase/FB15k/FB15k_%s_tail.pkl'% data, 'wb') as g:
                            pickle.dump(tail_list.tocsr(), g)

                    if not os.path.isfile('../dataset/Freebase/FB15k/FB15k_%s_rel.pkl' % data):
                        with open('../dataset/Freebase/FB15k/FB15k_%s_rel.pkl' % data, 'wb') as g:
                            pickle.dump(rel_list.tocsr(), g)

        print("\n----------Data Prep Results--------------")
        print("Total Unseen Triples   :", len(list(set(removed_triples))))
        print("Total Unseen Entities  :", len(list(set(unseen_entities))))
        print('-------------------------------------------')

    def batch_generator(self, data="train"):
        if not os.path.isfile('../dataset/Freebase/FB15k/FB15k_%s_head.pkl' % data):
            self.prepare_data()

        with open('../dataset/Freebase/FB15k/FB15k_%s_head.pkl' % data, 'rb') as g:
            head_list = pickle.load(g)

        with open('../dataset/Freebase/FB15k/FB15k_%s_tail.pkl' % data, 'rb') as g:
            tail_list =pickle.dump(g)

        with open('../dataset/Freebase/FB15k/FB15k_%s_rel.pkl' % data, 'rb') as g:
            rel_list = pickle.dump(g)

        #TODO: Corrupt the head list
        #TODO: Corrupt the tail list
        #TODO: Yield the batch size
        pass

if __name__=='__main__':
    data_handler = DataPrep()
    data_handler.prepare_data()







