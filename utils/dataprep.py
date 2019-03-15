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
from random import randint, choice


class Triple(object):
    def __init__(self, head=None,relation=None,tail=None):
        self.h = head
        self.r = relation
        self.t = tail

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
    def __init__(self, conf = None, dataset='Freebase'):
        # self.id = conf.id
        # self.batch_h = conf.batch_h
        # self.batch_t = conf.batch_t
        # self.batch_r = conf.batch_r
        # self.batch_y = conf.batch_y
        # self.batchSize  = conf.batchSize
        # self.negRate    = conf.negRate
        # self.negRelRate = conf.negRelRate
        self.config = GlobalConfig(dataset=dataset)

        if not os.path.exists(self.config.dataset.root_path):
            os.mkdir(self.config.dataset.root_path)

            with urllib.request.urlopen(self.config.dataset.url)\
                    as response, open(self.config.dataset.tar, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            try:
                extract(self.config.dataset.tar,self.config.dataset.root_path)
            except Exception as e:
                print("Could not extract the tgz file!")
                print(type(e),e.args)

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
            with open(self.config.dataset.donwnloaded_path +data+'.txt','r') as f:
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

        if not os.path.isfile(self.config.dataset.entity2idx_path):
            with open(self.config.dataset.entity2idx_path, 'wb') as f:
                pickle.dump(self.entity2idx, f)
        if not os.path.isfile(self.config.dataset.idx2entity_path):
            with open(self.config.dataset.idx2entity_path, 'wb') as f:
                pickle.dump(self.idx2entity, f)

        print("\n----------Train Triple Stats---------------")
        print("Total Training Triples   :", self.tot_triple)
        print("Total Head Only Entities :", self.tot_only_h)
        print("Total Tail Only Entities :", self.tot_only_t)
        print("Total Shared Entities    :", self.tot_shared)
        print("Total Relations          :", self.tot_r)
        print("-------------------------------------------")

    def prepare_data(self):
        if not self.entity2idx:
            if os.path.isfile(self.config.dataset.entity2idx_path):
                with open(self.config.dataset.entity2idx_path, 'rb') as f:
                    self.entity2idx=pickle.load(f)
            else:
                self.triple_idx_and_stats()
        unseen_entities = []
        removed_triples = []

        pos_triples_cnt = 0
        neg_triples_cnt = 0
        for data in ['train', 'valid', 'test']:
            with open(self.config.dataset.downloaded_path + "%s.txt" % data, 'r') as f:
                lines = f.readlines()

                head_list = sp.lil_matrix((np.max(list(self.entity2idx.values())) + 1, len(lines)),
                                     dtype='float32')
                rel_list  = sp.lil_matrix((np.max(list(self.entity2idx.values())) + 1, len(lines)),
                                     dtype='float32')
                tail_list = sp.lil_matrix((np.max(list(self.entity2idx.values())) + 1, len(lines)),
                                     dtype='float32')

                head_list_neg = sp.lil_matrix((np.max(list(self.entity2idx.values())) + 1, len(lines)),
                                          dtype='float32')
                rel_list_neg  = sp.lil_matrix((np.max(list(self.entity2idx.values())) + 1, len(lines)),
                                         dtype='float32')
                tail_list_neg = sp.lil_matrix((np.max(list(self.entity2idx.values())) + 1, len(lines)),
                                          dtype='float32')
                pos_triples = {}
                neg_triples = {}
                head_idx = []
                tail_idx = []
                rel_idx  = []
                ct = 0
                ct_neg = 0
                print("\nProcessing: %s dataset"%data)
                with progressbar.ProgressBar(max_value=len(lines)) as bar:
                    for i,line in enumerate(lines):
                        triple = parse_line(line)
                        if triple.h in self.entity2idx and triple.t in self.entity2idx and triple.r in self.entity2idx:
                            head_list[self.entity2idx[triple.h], ct] = 1
                            head_idx.append(self.entity2idx[triple.h])
                            rel_list[self.entity2idx[triple.r], ct]  = 1
                            rel_idx.append(self.entity2idx[triple.r])
                            tail_list[self.entity2idx[triple.t], ct] = 1
                            tail_idx.append(self.entity2idx[triple.t])
                            pos_triples[(self.entity2idx[triple.h],
                                           self.entity2idx[triple.r],
                                           self.entity2idx[triple.t])] = 1
                            ct += 1
                        else:
                            head_list_neg[self.entity2idx[triple.h], ct_neg] = 1
                            rel_list_neg[self.entity2idx[triple.r], ct_neg] = 1
                            tail_list_neg[self.entity2idx[triple.t], ct_neg] = 1
                            neg_triples[(self.entity2idx[triple.h],
                                           self.entity2idx[triple.r],
                                           self.entity2idx[triple.t])] = 1
                            ct_neg += 1
                            # if triple.h in self.entity2idx:
                            #     unseen_entities += [triple.h]
                            # if triple.r in self.entity2idx:
                            #     unseen_entities += [triple.r]
                            # if triple.t in self.entity2idx:
                            #     unseen_entities += [triple.t]
                            # removed_triples += [line]
                        bar.update(i)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_pos.pkl'% data):
                        with open(self.config.dataset.prepared_data_path+'%s_head_pos.pkl'% data, 'wb') as g:
                            pickle.dump(head_list.tocsr(), g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_tail_pos.pkl'% data):
                        with open(self.config.dataset.prepared_data_path+'%s_tail_pos.pkl'% data, 'wb') as g:
                            pickle.dump(tail_list.tocsr(), g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_rel_pos.pkl' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_rel_pos.pkl' % data, 'wb') as g:
                            pickle.dump(rel_list.tocsr(), g)

                pos_triples_cnt+=len(pos_triples)

                print("\nGenerating Corrupted Data: %s dataset" % data)
                #TODO: Check when (train, test, validate) and how much to corrupt the data
                with progressbar.ProgressBar(max_value=len(lines)) as bar:
                    for i, triple in enumerate(list(pos_triples.keys())):
                        rand_num=randint(0,900)
                        if rand_num<300:
                            # Corrupt Tail
                            idx = choice(tail_idx)
                            break_cnt=0
                            flag=False
                            while ((triple[0], triple[1], idx) in pos_triples
                                   or (triple[0], triple[1], idx) in neg_triples):
                                idx = choice(tail_idx)
                                break_cnt+=1
                                if break_cnt>=100:
                                    flag=True
                                    break
                            if flag:
                                continue

                            head_list_neg[triple[0], ct_neg] = 1
                            rel_list_neg[triple[1], ct_neg]  = 1
                            tail_list_neg[idx, ct_neg]       = 1
                            neg_triples[(triple[0],
                                           triple[1],
                                           idx)] = 1
                            ct_neg += 1

                        elif 300<=rand_num<600:
                            #Corrupt Head
                            idx = choice(head_idx)
                            break_cnt = 0
                            flag = False
                            while ((idx, triple[1], triple[2]) in pos_triples or
                            (idx, triple[1], triple[2]) in neg_triples):
                                idx = choice(head_idx)
                                break_cnt += 1
                                if break_cnt >= 100:
                                    flag = True
                                    break
                            if flag:
                                continue

                            head_list_neg[idx, ct_neg] = 1
                            rel_list_neg[triple[1], ct_neg] = 1
                            tail_list_neg[triple[2], ct_neg] = 1
                            neg_triples[(idx,
                                             triple[1],
                                             triple[2])] = 1
                            ct_neg += 1
                        else:
                            #Corrupt relation
                            idx = choice(rel_idx)
                            break_cnt = 0
                            flag = False
                            while (triple[0], idx, triple[2]) in pos_triples or\
                                   (triple[0], idx, triple[2]) in neg_triples:
                                idx = choice(rel_idx)
                                break_cnt += 1
                                if break_cnt >= 100:
                                    flag = True
                                    break
                            if flag:
                                continue

                            head_list_neg[triple[0], ct_neg] = 1
                            rel_list_neg[idx, ct_neg] = 1
                            tail_list_neg[triple[2], ct_neg] = 1
                            neg_triples[(triple[0],
                                             idx,
                                             triple[2])] = 1
                            ct_neg += 1

                        bar.update(i)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_neg.pkl' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_head_neg.pkl' % data, 'wb') as g:
                            pickle.dump(head_list_neg.tocsr(), g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_tail_neg.pkl' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_tail_neg.pkl' % data, 'wb') as g:
                            pickle.dump(tail_list_neg.tocsr(), g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_rel_neg.pkl' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_rel_neg.pkl' % data, 'wb') as g:
                            pickle.dump(rel_list_neg.tocsr(), g)

                    neg_triples_cnt+=len(neg_triples)

        print("\n----------Data Prep Results--------------")
        print("Total Positive Triples :", pos_triples_cnt)
        print("Total Negative Triples :", neg_triples_cnt)
        print("Total Unseen Triples   :", len(list(set(removed_triples))))
        print("Total Unseen Entities  :", len(list(set(unseen_entities))))
        print('-------------------------------------------')

    def batch_generator(self, data="train"):
        if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_pos.pkl' % data)\
                or not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_neg.pkl' % data):
            self.prepare_data()

        with open(self.config.dataset.prepared_data_path+'%s_head_pos.pkl' % data, 'rb') as g:
            head_list_pos = pickle.load(g)

        with open(self.config.dataset.prepared_data_path+'%s_tail_pos.pkl' % data, 'rb') as g:
            tail_list_pos = pickle.load(g)

        with open(self.config.dataset.prepared_data_path+'%s_rel_pos.pkl' % data, 'rb') as g:
            rel_list_pos = pickle.load(g)

        with open(self.config.dataset.prepared_data_path+'%s_head_neg.pkl' % data, 'rb') as g:
            head_list_neg = pickle.load(g)

        with open(self.config.dataset.prepared_data_path+'%s_tail_neg.pkl' % data, 'rb') as g:
            tail_list_neg = pickle.load(g)

        with open(self.config.dataset.prepared_data_path+'%s_rel_neg.pkl' % data, 'rb') as g:
            rel_list_neg = pickle.load(g)

        #TODO: Yield the batch size
        pass

if __name__=='__main__':
    data_handler = DataPrep('Freebase')
    data_handler.prepare_data()







