#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for preparing the data
"""
import numpy as np
from config.config import GlobalConfig
import pickle
import os
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
    def __init__(self, dataset='Freebase'):

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
        self.tot_entity = 0


    def read_triple(self, datatype=None):
        if datatype is None:
            datatype = ['train']
        for data in datatype:
            with open(self.config.dataset.downloaded_path +data+'.txt','r') as f:
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

        self.tot_entity = self.tot_only_h+self.tot_only_t-self.tot_shared

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

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_pos.npz'% data):
                        with open(self.config.dataset.prepared_data_path+'%s_head_pos.pkl'% data, 'wb') as g:
                            pickle.dump(head_list, g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_tail_pos.npz'% data):
                        with open(self.config.dataset.prepared_data_path+'%s_tail_pos.pkl'% data, 'wb') as g:
                            pickle.dump(tail_list, g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_rel_pos.npz' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_rel_pos.pkl' % data, 'wb') as g:
                            pickle.dump(rel_list, g)

                pos_triples_cnt+=len(pos_triples)

                print("\nGenerating Corrupted Data: %s dataset" % data)
                #TODO: Check when (train, test, validate) and how much to corrupt the data
                with progressbar.ProgressBar(max_value=len(lines)) as bar:
                    for i, triple in enumerate(list(pos_triples.keys())):
                        rand_num=randint(0,900)
                        if rand_num<300:
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

                        elif 300<=rand_num<600:
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
                        else:
                            #Corrupt relation
                            idx = choice(rel_list)
                            break_cnt = 0
                            flag = False
                            while (triple[0], idx, triple[2]) in pos_triples or\
                                   (triple[0], idx, triple[2]) in neg_triples:
                                idx = choice(rel_list)
                                break_cnt += 1
                                if break_cnt >= 100:
                                    flag = True
                                    break
                            if flag:
                                continue

                            head_list_neg.append(triple[0])
                            rel_list_neg.append(idx)
                            tail_list_neg.append(triple[2])
                            neg_triples[(triple[0],
                                             idx,
                                             triple[2])] = 1

                        bar.update(i)

                    print("\nneg_heads:", head_list_neg[:5])
                    print("neg_tails:", tail_list_neg[:5])
                    print("neg_rels:", rel_list_neg[:5])

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_neg.npz' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_head_neg.pkl' % data, 'wb') as g:
                            pickle.dump(head_list_neg, g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_tail_neg.npz' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_tail_neg.pkl' % data, 'wb') as g:
                            pickle.dump(tail_list_neg, g)

                    if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_rel_neg.npz' % data):
                        with open(self.config.dataset.prepared_data_path+'%s_rel_neg.pkl' % data, 'wb') as g:
                            pickle.dump(rel_list_neg, g)

                    neg_triples_cnt+=len(neg_triples)

        print("\n----------Data Prep Results--------------")
        print("Total Positive Triples :", pos_triples_cnt)
        print("Total Negative Triples :", neg_triples_cnt)
        print("Total Unseen Triples   :", len(list(set(removed_triples))))
        print("Total Unseen Entities  :", len(list(set(unseen_entities))))
        print('-------------------------------------------')

    def batch_generator(self, batch=128, data="train"):
        if not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_pos.pkl' % data)\
                or not os.path.isfile(self.config.dataset.prepared_data_path+'%s_head_neg.pkl' % data):
            self.prepare_data()

        with open(self.config.dataset.prepared_data_path+'%s_head_pos.pkl' % data, 'rb') as g:
            head_list_pos = (pickle.load(g))

        with open(self.config.dataset.prepared_data_path+'%s_tail_pos.pkl' % data, 'rb') as g:
            tail_list_pos = (pickle.load(g))

        with open(self.config.dataset.prepared_data_path+'%s_rel_pos.pkl' % data, 'rb') as g:
            rel_list_pos = (pickle.load(g))

        with open(self.config.dataset.prepared_data_path+'%s_head_neg.pkl' % data, 'rb') as g:
            head_list_neg = (pickle.load(g))

        with open(self.config.dataset.prepared_data_path+'%s_tail_neg.pkl' % data, 'rb') as g:
            tail_list_neg = (pickle.load(g))

        with open(self.config.dataset.prepared_data_path+'%s_rel_neg.pkl' % data, 'rb') as g:
            rel_list_neg = (pickle.load(g))

        number_of_batches = (len(head_list_pos)) // batch
        print("Number of bacthes:", number_of_batches)
        counter = 0
        while True:
            ph = head_list_pos[batch*counter:batch*(counter + 1)]
            pr = rel_list_pos[batch * counter:batch * (counter + 1)]
            pt = tail_list_pos[batch * counter:batch * (counter + 1)]

            nh = head_list_neg[batch * counter:batch * (counter + 1)]
            nr = rel_list_neg[batch * counter:batch * (counter + 1)]
            nt = tail_list_neg[batch * counter:batch * (counter + 1)]

            counter += 1
            if data=='Train':
                yield ph, pr, pt, nh, nr, nt
            else:
                yield ph, pr, pt
            if counter == number_of_batches:
                counter = 0


if __name__=='__main__':
    data_handler = DataPrep('Freebase')
    data_handler.prepare_data()
    gen = data_handler.batch_generator()
    for i in range(5):
        ph, pr, pt, nh, nr, nt = list(next(gen))
        print("\nph:", ph)
        print("pr:", pr)
        print("pt:", pt)
        print("nh:", nh)
        print("nr:", nr)
        print("nt:", nt)







