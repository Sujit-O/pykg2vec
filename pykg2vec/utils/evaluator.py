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
import progressbar

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

    def append_result(self, result):
        
        id_replace_tail = result[0]
        id_replace_head = result[1]
        h_list = result[2]
        r_list = result[3]
        t_list = result[4]
        self.epoch = result[5]
        
        total_test = len(h_list)
        
        for triple_id in range(total_test):
            h, r, t = h_list[triple_id], r_list[triple_id], t_list[triple_id]

            t_rank, f_t_rank = self.get_tail_rank(id_replace_tail[triple_id], h, r, t)
            h_rank, f_h_rank = self.get_head_rank(id_replace_head[triple_id], h, r, t)

            self.rank_head.append(h_rank)
            self.rank_tail.append(t_rank)
            self.f_rank_head.append(f_h_rank)
            self.f_rank_tail.append(f_t_rank)

    def append_result_new(self, result):
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

    def reset(self):
        # temporarily used buffers and indexes.
        self.rank_head = []
        self.rank_tail = []
        self.f_rank_head = []
        self.f_rank_tail = []
        self.epoch = None
        self.start_time = timeit.default_timer()

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

def evaluation_process(result_queue, output_queue, config, model_name, tuning):
    """ The process that coordinates the tasks of evaluation.
           
        Args:
            result_queue (Queue): Multiprocessing queue to acquire inference result
            output_queue (Queue): Multiprocessing queue to store the evaluation result
            config (object):Model configuration object instance
            model_name (str): Name of the model
            tuning (bool): Check if tuning or performing full test.

    """

    calculator = MetricCalculator(config)

    while True:
        result = result_queue.get()
        
        if result == Evaluator.TEST_BATCH_START:
            calculator.reset()
            
        elif result == Evaluator.TEST_BATCH_STOP:
            calculator.settle()
            calculator.display_summary()

            if calculator.epoch >= config.epochs - 1:
                calculator.save_test_summary(model_name)

                if tuning:
                    score = calculator.get_curr_score()
                    output_queue.put(score)

                break
        elif result == Evaluator.TEST_BATCH_EARLY_STOP:
            break
        else:
            if config.batch_size_testing == 1:
                calculator.append_result_new(result)
            else:
                calculator.append_result(result)

class Evaluator(EvaluationMeta):
    """Class to perform evaluation of the model.

        Args:
            model (object): Model object
            debug (bool): Flag to check if its debugging
            data_type (str): evaluating 'test' or 'valid'
            tuning (bool): Flag to denoting tuning if True

        Examples:
            >>> from pykg2vec.utils.evaluator import Evaluator
            >>> evaluator = Evaluator(model=model, debug=False, tuning=True)
            >>> evaluator.test_batch(Session(), 0)
            >>> acc = evaluator.output_queue.get()
            >>> evaluator.stop()
    """
    TEST_BATCH_START = "Start!"
    TEST_BATCH_STOP = "Stop!"
    TEST_BATCH_EARLY_STOP = "EarlyStop!"

    def __init__(self, model=None, debug=False, data_type='valid', tuning=False, session=None):
        
        self.session = session 
        self.model = model
        self.debug = debug
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
        self.n_test = model.config.test_num
        if self.n_test == 0:
            self.n_test = tot_rows_data
        else:
            self.n_test = min(self.n_test, tot_rows_data)

        ''' 
            loop_len: the # of loops to perform batch evaluation. 
            if debug mode is turned on, then set to only 2. 
        ''' 
        if self.n_test < self.model.config.batch_size_testing:
            self.loop_len = 1
        else:
            self.loop_len = (self.n_test // self.model.config.batch_size_testing) if not self.debug else 2
        
        self.n_test = self.model.config.batch_size_testing * self.loop_len

        '''
            create the process that manages the batched evaluating results.
            result_queue: stores the results for each batch. 
            output_queue: stores the result for a trial, used by bayesian_optimizer.
        '''
        self.result_queue = Queue()
        self.output_queue = Queue()
        self.rank_calculator = Process(target=evaluation_process,
                                       args=(self.result_queue, self.output_queue, 
                                             self.model.config, self.model.model_name, self.tuning))
        self.rank_calculator.start()

    def stop(self):
        """Function that stops the evaluation process"""
        self.rank_calculator.join()
        self.rank_calculator.terminate()

    def test_batch(self, epoch=None):
        """Function that performs the batch testing"""
        
        print("Testing [%d/%d] Triples" % (self.n_test, len(self.eval_data)))

        size_per_batch = self.model.config.batch_size_testing
        head_rank, tail_rank = self.model.test_batch()

        widgets = ['Inferring for Evaluation: ', progressbar.AnimatedMarker(), " Done:",
                   progressbar.Percentage(), " ", progressbar.AdaptiveETA()]

        self.result_queue.put(self.TEST_BATCH_START)
        with progressbar.ProgressBar(max_value=self.loop_len, widgets=widgets) as bar:
            for i in range(self.loop_len):
                data = np.asarray([[self.eval_data[x].h, self.eval_data[x].r, self.eval_data[x].t]
                                   for x in range(size_per_batch * i, size_per_batch * (i + 1))])
                h = data[:, 0]
                r = data[:, 1]
                t = data[:, 2]

                feed_dict = {
                    self.model.test_h_batch: h,
                    self.model.test_r_batch: r,
                    self.model.test_t_batch: t}

                head_tmp, tail_tmp = np.squeeze(self.session.run([head_rank, tail_rank], feed_dict))
                
                result_data = [tail_tmp, head_tmp, h, r, t, epoch]
                self.result_queue.put(result_data)

                bar.update(i)

        self.result_queue.put(self.TEST_BATCH_STOP)

    def test_per_sample(self, epoch=None):

        print("New Testing [%d/%d] Triples" % (self.n_test, len(self.eval_data)))

        rank = self.model.def_predict()

        widgets = ['Inferring for Evaluation: ', progressbar.AnimatedMarker(), " Done:",
                   progressbar.Percentage(), " ", progressbar.AdaptiveETA()]

        self.result_queue.put(self.TEST_BATCH_START)
        with progressbar.ProgressBar(max_value=self.n_test, widgets=widgets) as bar:
            entity_array = np.arange(self.model.config.kg_meta.tot_entity)
            
            for i in range(self.n_test):
                h, r, t = self.eval_data[i].h, self.eval_data[i].r, self.eval_data[i].t
                
                # generate head batch and predict heads. Tensorflow handles broadcasting.
                
                h_batch = np.tile([h], self.model.config.kg_meta.tot_entity)
                r_batch = np.tile([r], self.model.config.kg_meta.tot_entity)
                t_batch = np.tile([t], self.model.config.kg_meta.tot_entity)

                feed_dict = {
                    self.model.test_h_batch: entity_array,
                    self.model.test_r_batch: r_batch,
                    self.model.test_t_batch: t_batch}

                head_tmp = np.squeeze(self.session.run([rank], feed_dict))
                
                feed_dict = {
                    self.model.test_h_batch: h_batch,
                    self.model.test_r_batch: r_batch,
                    self.model.test_t_batch: entity_array}
                
                tail_tmp = np.squeeze(self.session.run([rank], feed_dict))

                result_data = [tail_tmp, head_tmp, h, r, t, epoch]

                self.result_queue.put(result_data)

                bar.update(i)

        self.result_queue.put(self.TEST_BATCH_STOP)

    def save_training_result(self, losses):
        """Function that saves training result"""
        files = os.listdir(str(self.result_path))
        l = len([f for f in files if self.model.model_name in f if 'Training' in f])
        df = pd.DataFrame(losses, columns=['Epochs', 'Loss'])
        with open(str(self.result_path / (self.model.model_name + '_Training_results_' + str(l) + '.csv')),
                  'w') as fh:
            df.to_csv(fh)
