import unittest

import sys
sys.path.append("../")

from core.TransE import TransE
from core.TransH import TransH
from core.TransR import TransR
from config.config import TransEConfig, TransHConfig, TransHConfig, TransRConfig
from utils.dataprep import DataPrep
from utils.trainer import Trainer

import tensorflow as tf
import numpy as np
import timeit
class KnowledgeGraphDataPrepTestcase(unittest.TestCase):
    
    def setUp(self):
        print('setup')
        
    def test_freebase15k(self):
        knowledge_graph = DataPrep("Freebase15k")
        knowledge_graph.dump()
        
    def test_generator(self):
        data_handler = DataPrep('Freebase15k')

        data_handler.start_multiprocess()
        start_time = timeit.default_timer()
        gen = data_handler.batch_generator_train_proje(batch_size=100)

        batch_counts = 0
        for i in range(8):
            triples = next(gen)
            data_handler.raw_training_data_queue.put((i, triples))
            batch_counts += 1
        
        print('[%.4f sec] spent on sending tasks to queue' % (timeit.default_timer() - start_time))    

        start_time = timeit.default_timer()
        while batch_counts > 0:
            batch_counts -= 1
            hr_hr, hr_t, tr_tr, tr_h = data_handler.training_data_queue.get()

        print('[%.4f sec] spent on completing all tasks from queue' % (timeit.default_timer() - start_time))
        data_handler.end()

    def tearDown(self):
        print('teardown')
        tf.reset_default_graph()

def suite():
    suite = unittest.TestSuite()
    suite.addTest(KnowledgeGraphDataPrepTestcase('test_freebase15k'))
    suite.addTest(KnowledgeGraphDataPrepTestcase('test_generator'))
    return suite

if __name__ == '__main__':
    """ Execute whole test case as a whole """
    runner = unittest.TextTestRunner()
    runner.run(suite())