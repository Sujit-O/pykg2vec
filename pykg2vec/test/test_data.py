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

class KnowledgeGraphDataPrepTestcase(unittest.TestCase):
    
    def setUp(self):
        print('setup')
        
    def test_freebase15k(self):
        self.data_handler = DataPrep("Freebase15k")

    def tearDown(self):
        print('teardown')
        tf.reset_default_graph()

def suite():
    suite = unittest.TestSuite()
    suite.addTest(KnowledgeGraphDataPrepTestcase('test_freebase15k'))
    return suite

if __name__ == '__main__':
    """ Execute whole test case as a whole """
    runner = unittest.TextTestRunner()
    runner.run(suite())