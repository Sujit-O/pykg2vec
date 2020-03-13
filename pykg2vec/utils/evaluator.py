#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for evaluating the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import timeit
from multiprocessing import Process, Queue
import tensorflow as tf
from pykg2vec.core.KGMeta import EvaluationMeta


class MetricCalculator:
    '''
        MetricCalculator aims to 
        1) address all the statistic tasks.
        2) provide interfaces for querying results.

        MetricCalculator is expected to be used by "evaluation_process".
    '''
    def __init__(self, config):
        self.config = config 

        self.hr_t = config.knowledge_graph.read_cache_data('hr_t')
        self.tr_h = config.knowledge_graph.read_cache_data('tr_h')

        # (f)mr  : (filtered) mean rank
        # (f)mrr : (filtered) mean reciprocal rank
        # (f)hit : (filtered) hit-k ratio
        self.mr   = {}
        self.fmr  = {}
        self.mrr  = {}
        self.fmrr = {}
        self.hit  = {}
        self.fhit = {}

        self.reset()

    def reset(self):
        # temporarily used buffers and indexes.
        self.rank_head = []
        self.rank_tail = []
        self.f_rank_head = []
        self.f_rank_tail = []
        self.epoch = None
        self.start_time = timeit.default_timer()

    def append_result(self, result):
        predict_tail = result[0]
        predict_head = result[1]

        h,r,t = result[2], result[3], result[4]

        self.epoch = result[5]

        t_rank, f_t_rank = self.get_tail_rank(predict_tail, h, r, t)
        h_rank, f_h_rank = self.get_head_rank(predict_head, h, r, t)

        self.rank_head.append(h_rank)
        self.rank_tail.append(t_rank)
        self.f_rank_head.append(f_h_rank)
        self.f_rank_tail.append(f_t_rank)

    def get_tail_rank(self, tail_candidate, h, r, t):
        """Function to evaluate the tail rank.
           
           Args:
               id_replace_tail (list): List of the predicted tails for the given head, relation pair
               h (int): head id
               r (int): relation id
               t (int): tail id
               hr_t (dict): list of tails for the given hwS and relation pari.

            Returns:
                Tensors: Returns tail rank and filetered tail rank
        """
        trank = 0
        ftrank = 0

        for j in range(len(tail_candidate)):
            val = tail_candidate[-j - 1]
            if val == t:
                break
            else:
                trank += 1
                ftrank += 1
                if val in self.hr_t[(h, r)]:
                    ftrank -= 1
        
        return trank, ftrank

    def get_head_rank(self, head_candidate, h, r, t):
        """Function to evaluate the head rank.
               
           Args:
               head_candidate (list): List of the predicted head for the given tail, relation pair
               h (int): head id
               r (int): relation id
               t (int): tail id

            Returns:
                Tensors: Returns head  rank and filetered head rank
        """
        hrank = 0
        fhrank = 0

        for j in range(len(head_candidate)):
            val = head_candidate[-j - 1]
            if val == h:
                break
            else:
                hrank += 1
                fhrank += 1
                if val in self.tr_h[(t, r)]:
                    fhrank -= 1

        return hrank, fhrank

    def settle(self):
        head_ranks  = np.asarray(self.rank_head, dtype=np.float32)+1
        tail_ranks  = np.asarray(self.rank_tail, dtype=np.float32)+1
        head_franks = np.asarray(self.f_rank_head, dtype=np.float32)+1
        tail_franks = np.asarray(self.f_rank_tail, dtype=np.float32)+1

        ranks  = np.concatenate((head_ranks, tail_ranks)) 
        franks = np.concatenate((head_franks, tail_franks))

        self.mr[self.epoch]   = np.mean(ranks)
        self.mrr[self.epoch]  = np.mean(np.reciprocal(ranks))
        self.fmr[self.epoch]  = np.mean(franks)
        self.fmrr[self.epoch] = np.mean(np.reciprocal(franks))

        for hit in self.config.hits:
            self.hit[(self.epoch, hit)] = np.mean(ranks<=hit, dtype=np.float32)
            self.fhit[(self.epoch, hit)] = np.mean(franks<=hit, dtype=np.float32)

    def get_curr_score(self):
        return self.mr[self.epoch]



    def save_test_summary(self, model_name):
        """Function to save the test of the summary.
               
            Args:
                model_name (str): specify the name of the model. 

        """
        files = os.listdir(str(self.config.path_result))
        l = len([f for f in files if model_name in f if 'Testing' in f])
        with open(str(self.config.path_result / (model_name + '_summary_' + str(l) + '.txt')), 'w') as fh:
            fh.write('----------------SUMMARY----------------\n')
            for key, val in self.config.__dict__.items():
                if 'gpu' in key:
                    continue
                if 'kg_meta' in key or 'knowledge_graph' in key:
                    continue
                if not isinstance(val, str):
                    if isinstance(val, list):
                        v_tmp = '['
                        for i, v in enumerate(val):
                            if i == 0:
                                v_tmp += str(v)
                            else:
                                v_tmp += ',' + str(v)
                        v_tmp += ']'
                        val = v_tmp
                    else:
                        val = str(val)
                fh.write(key + ':' + val + '\n')
            fh.write('-----------------------------------------\n')
            fh.write("\n----------Metadata Info for Dataset:%s----------------" % self.config.knowledge_graph.dataset_name)
            fh.write("Total Training Triples   :%d\n"%self.config.kg_meta.tot_train_triples)
            fh.write("Total Testing Triples    :%d\n"%self.config.kg_meta.tot_test_triples)
            fh.write("Total validation Triples :%d\n"%self.config.kg_meta.tot_valid_triples)
            fh.write("Total Entities           :%d\n"%self.config.kg_meta.tot_entity)
            fh.write("Total Relations          :%d\n"%self.config.kg_meta.tot_relation)
            fh.write("---------------------------------------------")

        columns = ['Epoch', 'Mean Rank', 'Filtered Mean Rank', 'Mean Reciprocal Rank', 'Filtered Mean Reciprocal Rank']
        for hit in self.config.hits:
            columns += ['Hit-%d Ratio'%hit, 'Filtered Hit-%d Ratio'%hit]

        results = []
        for epoch in self.mr.keys():
            res_tmp = [epoch, self.mr[epoch], self.fmr[epoch], self.mrr[epoch], self.fmrr[epoch]]

            for hit in self.config.hits:
                res_tmp.append(self.hit[(epoch, hit)])
                res_tmp.append(self.fhit[(epoch, hit)])

            results.append(res_tmp)

        df = pd.DataFrame(results, columns=columns)

        with open(str(self.config.path_result / (model_name + '_Testing_results_' + str(l) + '.csv')),'a') as fh:
            df.to_csv(fh)

    def display_summary(self):
        """Function to print the test summary."""
        kg = self.config.knowledge_graph
        stop_time = timeit.default_timer()
        print("------Test Results for %s: Epoch: %d --- time: %.2f------------" % (kg.dataset_name, self.epoch, stop_time - self.start_time))
        print('--# of entities, # of relations: %d, %d'%(kg.kg_meta.tot_entity, kg.kg_meta.tot_relation) )
        print('--mr,  filtered mr             : %.4f, %.4f'%(self.mr[self.epoch], self.fmr[self.epoch]))
        print('--mrr, filtered mrr            : %.4f, %.4f'%(self.mrr[self.epoch], self.fmrr[self.epoch]))
        for hit in self.config.hits:
            print('--hits%d                        : %.4f ' % (hit, (self.hit[(self.epoch, hit)])))
            print('--filtered hits%d               : %.4f ' % (hit, (self.fhit[(self.epoch, hit)])))
        print("---------------------------------------------------------")


class Evaluator(EvaluationMeta):
    """Class to perform evaluation of the model.

        Args:
            model (object): Model object
            data_type (str): evaluating 'test' or 'valid'
            tuning (bool): Flag to denoting tuning if True

        Examples:
            >>> from pykg2vec.utils.evaluator import Evaluator
            >>> evaluator = Evaluator(model=model, tuning=True)
            >>> evaluator.test_batch(Session(), 0)
            >>> acc = evaluator.output_queue.get()
            >>> evaluator.stop()
    """
    TEST_BATCH_START = "Start!"
    TEST_BATCH_STOP = "Stop!"
    TEST_BATCH_EARLY_STOP = "EarlyStop!"

    def __init__(self, model=None, data_type='valid', tuning=False):
        
        self.model = model
        self.tuning = tuning
        self.result_path = self.model.config.path_result

        if data_type == 'test':
            self.eval_data = self.model.config.knowledge_graph.read_cache_data('triplets_test')
        elif data_type == 'valid':
            self.eval_data = self.model.config.knowledge_graph.read_cache_data('triplets_valid')
        else:
            raise NotImplementedError("%s datatype is not available!" % data_type)

        tot_rows_data = len(self.eval_data)

        '''
            n_test: number of triplets to be tested
            1) if n_test == 0, test all the triplets. 
            2) if n_test >= # of testable triplets, then set n_test to # of testable triplets
        '''
        self.n_test = self.model.config.test_num
        if self.n_test == 0:
            self.n_test = tot_rows_data
        else:
            self.n_test = min(self.n_test, tot_rows_data)

        self.metric_calculator = MetricCalculator(self.model.config)

    @tf.function
    def test_tail_rank_multiclass(self, h, r, topk=-1):
        rank = self.model.predict_tail(h, r, topk=topk)
        return tf.squeeze(rank, 0)

    @tf.function
    def test_head_rank_multiclass(self, r, t, topk=-1):
        rank = self.model.predict_head(t, r, topk=topk)
        return tf.squeeze(rank, 0)

    @tf.function
    def test_tail_rank(self, h, r, topk=-1):
        tot_ent = self.model.config.kg_meta.tot_entity
        
        h_batch = tf.tile([h], [tot_ent])
        r_batch = tf.tile([r], [tot_ent])
        entity_array = tf.range(tot_ent)

        return self.model.predict(h_batch, r_batch, entity_array, topk=topk)

    @tf.function
    def test_head_rank(self, r, t, topk=-1):
        tot_ent = self.model.config.kg_meta.tot_entity
        
        entity_array = tf.range(tot_ent)
        r_batch = tf.tile([r], [tot_ent])
        t_batch = tf.tile([t], [tot_ent])
    
        return self.model.predict(entity_array, r_batch, t_batch, topk=topk)

    @tf.function
    def test_rel_rank(self, h, t, topk=-1):
        tot_rel = self.model.config.kg_meta.tot_relation
    
        h_batch = tf.tile([h], [tot_rel])
        rel_array = tf.range(tot_rel)
        t_batch = tf.tile([t], [tot_rel])
    
        return self.model.predict(h_batch, rel_array, t_batch, topk=topk)

    def test(self, epoch=None):
        print("Testing [%d/%d] Triples" % (self.n_test, len(self.eval_data)))

        progress_bar = tf.keras.utils.Progbar(self.n_test)

        self.metric_calculator.reset()

        for i in range(self.n_test):
            h, r, t = self.eval_data[i].h, self.eval_data[i].r, self.eval_data[i].t
            
            # generate head batch and predict heads. Tensorflow handles broadcasting.
            h_tensor = tf.convert_to_tensor(h, dtype=tf.int32)
            r_tensor = tf.convert_to_tensor(r, dtype=tf.int32)
            t_tensor = tf.convert_to_tensor(t, dtype=tf.int32)

            if self.model.model_name.lower() in ["tucker", "tucker_v2", "conve", "proje_pointwise"]:
                hrank = self.test_head_rank_multiclass(r_tensor, t_tensor, self.model.config.kg_meta.tot_entity)
                trank = self.test_tail_rank_multiclass(h_tensor, r_tensor, self.model.config.kg_meta.tot_entity)
            else:
                hrank = self.test_head_rank(r_tensor, t_tensor, self.model.config.kg_meta.tot_entity)
                trank = self.test_tail_rank(h_tensor, r_tensor, self.model.config.kg_meta.tot_entity)
            
            result_data = [trank.numpy(), hrank.numpy(), h, r, t, epoch]

            self.metric_calculator.append_result(result_data)

            progress_bar.add(1)

        self.metric_calculator.settle()
        self.metric_calculator.display_summary()

        if self.metric_calculator.epoch >= self.model.config.epochs - 1:
            self.metric_calculator.save_test_summary(self.model.model_name)

        return self.metric_calculator.get_curr_score()


