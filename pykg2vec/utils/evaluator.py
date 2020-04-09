#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for evaluating the results
"""
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import pandas as pd
import timeit
from multiprocessing import Process, Queue
import tensorflow as tf
from pykg2vec.core.KGMeta import EvaluationMeta
from pykg2vec.utils.generator import TrainingStrategy
from pykg2vec.utils.logger import Logger


class MetricCalculator:
    '''
        MetricCalculator aims to 
        1) address all the statistic tasks.
        2) provide interfaces for querying results.

        MetricCalculator is expected to be used by "evaluation_process".
    '''
    _logger = Logger().get_logger(__name__)

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

    def get_curr_scores(self):
        scores = {'mr': self.mr[self.epoch], 
                  'fmr':self.fmr[self.epoch],
                  'mrr':self.mrr[self.epoch], 
                  'fmrr':self.fmrr[self.epoch]}
        return scores


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
        test_results = []
        test_results.append('')
        test_results.append("------Test Results for %s: Epoch: %d --- time: %.2f------------" % (kg.dataset_name, self.epoch, stop_time - self.start_time))
        test_results.append('--# of entities, # of relations: %d, %d'%(kg.kg_meta.tot_entity, kg.kg_meta.tot_relation) )
        test_results.append('--mr,  filtered mr             : %.4f, %.4f'%(self.mr[self.epoch], self.fmr[self.epoch]))
        test_results.append('--mrr, filtered mrr            : %.4f, %.4f'%(self.mrr[self.epoch], self.fmrr[self.epoch]))
        for hit in self.config.hits:
            test_results.append('--hits%d                        : %.4f ' % (hit, (self.hit[(self.epoch, hit)])))
            test_results.append('--filtered hits%d               : %.4f ' % (hit, (self.fhit[(self.epoch, hit)])))
        test_results.append("---------------------------------------------------------")
        test_results.append('')
        self._logger.info("\n".join(test_results))


class Evaluator(EvaluationMeta):
    """Class to perform evaluation of the model.

        Args:
            model (object): Model object
            tuning (bool): Flag to denoting tuning if True

        Examples:
            >>> from pykg2vec.utils.evaluator import Evaluator
            >>> evaluator = Evaluator(model=model, tuning=True)
            >>> evaluator.test_batch(Session(), 0)
            >>> acc = evaluator.output_queue.get()
            >>> evaluator.stop()
    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, model, tuning=False):
        self.model = model
        self.tuning = tuning
        self.test_data = self.model.config.knowledge_graph.read_cache_data('triplets_test')
        self.eval_data = self.model.config.knowledge_graph.read_cache_data('triplets_valid')
        self.metric_calculator = MetricCalculator(self.model.config)

    @tf.function
    def test_tail_rank(self, h, r, topk=-1):
        if hasattr(self.model, 'predict_tail_rank'):
            h = tf.expand_dims(h, 0)
            r = tf.expand_dims(r, 0)
            rank = self.model.predict_tail_rank(h, r, topk=topk)
            return tf.squeeze(rank, 0)

        if hasattr(self.model, 'predict'):
            h_batch = tf.tile([h], [self.model.config.kg_meta.tot_entity])
            r_batch = tf.tile([r], [self.model.config.kg_meta.tot_entity])
            entity_array = tf.range(self.model.config.kg_meta.tot_entity)

            preds = self.model.forward(h_batch, r_batch, entity_array)
            _, rank = tf.nn.top_k(preds, k=topk)
            return rank

        raise NotImplementedError("Neither %s nor %s has been implemented" % ("predict_tail_rank", "predict_rank"))

    @tf.function
    def test_head_rank(self, r, t, topk=-1):
        if hasattr(self.model, 'predict_head_rank'):
            t = tf.expand_dims(t, 0)
            r = tf.expand_dims(r, 0)
            rank = self.model.predict_head_rank(t, r, topk=topk)
            return tf.squeeze(rank, 0)

        if hasattr(self.model, 'predict'):
            entity_array = tf.range(self.model.config.kg_meta.tot_entity)
            r_batch = tf.tile([r], [self.model.config.kg_meta.tot_entity])
            t_batch = tf.tile([t], [self.model.config.kg_meta.tot_entity])

            preds = self.model.forward(entity_array, r_batch, t_batch)
            _, rank = tf.nn.top_k(preds, k=topk)
            return rank

        raise NotImplementedError("Neither %s nor %s has been implemented" % ("predict_head_rank", "predict_rank"))

    @tf.function
    def test_rel_rank(self, h, t, topk=-1):
        if hasattr(self.model, 'predict_rel_rank'):
            h = tf.expand_dims(h, 0)
            t = tf.expand_dims(t, 0)
            rank = self.model.predict_rel_rank(h, t, topk=topk)
            return tf.squeeze(rank, 0)

        if hasattr(self.model, 'predict'):
            h_batch = tf.tile([h], [self.model.config.kg_meta.tot_relation])
            rel_array = tf.range(self.model.config.kg_meta.tot_relation)
            t_batch = tf.tile([t], [self.model.config.kg_meta.tot_relation])
            
            preds = self.model.forward(h_batch, rel_array, t_batch)
            _, rank = tf.nn.top_k(preds, k=topk)
            return rank

        raise NotImplementedError("Neither %s nor %s has been implemented" % ("predict_rel_rank", "predict_rank"))

    def mini_test(self, epoch=None):
        if self.model.config.test_num == 0:
            tot_valid_to_test = len(self.eval_data)
        else:
            tot_valid_to_test = min(self.model.config.test_num, len(self.eval_data))
        if self.model.config.debug: 
            tot_valid_to_test = 10

        self._logger.info("Mini-Testing on [%d/%d] Triples in the valid set." % (tot_valid_to_test, len(self.eval_data)))
        return self.test(self.eval_data, tot_valid_to_test, epoch=epoch)

    def full_test(self, epoch=None):
        tot_valid_to_test = len(self.test_data)
        if self.model.config.debug:
            tot_valid_to_test  = 10

        self._logger.info("Full-Testing on [%d/%d] Triples in the test set." % (tot_valid_to_test, len(self.test_data)))
        return self.test(self.test_data, tot_valid_to_test, epoch=epoch)

    def test(self, data, num_of_test, epoch=None):
        self.metric_calculator.reset()
        
        progress_bar = tf.keras.utils.Progbar(num_of_test)

        for i in range(num_of_test):
            h, r, t = data[i].h, data[i].r, data[i].t
            
            # generate head batch and predict heads. Tensorflow handles broadcasting.
            h_tensor = tf.convert_to_tensor(h, dtype=tf.int32)
            r_tensor = tf.convert_to_tensor(r, dtype=tf.int32)
            t_tensor = tf.convert_to_tensor(t, dtype=tf.int32)

            hrank = self.test_head_rank(r_tensor, t_tensor, self.model.config.kg_meta.tot_entity)
            trank = self.test_tail_rank(h_tensor, r_tensor, self.model.config.kg_meta.tot_entity)
            
            result_data = [trank.numpy(), hrank.numpy(), h, r, t, epoch]

            self.metric_calculator.append_result(result_data)

            progress_bar.add(1)

        self.metric_calculator.settle()
        self.metric_calculator.display_summary()

        if self.metric_calculator.epoch >= self.model.config.epochs - 1:
            self.metric_calculator.save_test_summary(self.model.model_name)

        return self.metric_calculator.get_curr_scores()