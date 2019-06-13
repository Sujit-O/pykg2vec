"""
global_config.py
====================================
It stores the global configuration of the files and consists of modules for processing data.
"""


from pathlib import Path


class GeneratorConfig(object):
    """Configuration for Generator

        Args:
          batch_size (int): int value for batch size
          loss_type (str) : string 'distance' or 'entropy' to show
                     type of algorithm used
          data_path (object) : path from where data is read and batch is
                     generated
          sampling (str) :  type of sampling used for generating negative
                     triples
          raw_queue_size (int): size of the generator queue for storing raw triples
          processed_queue_size (int): size of the generator queue for storing processed triples
          queue_size (int): size of the generator queue from where the data is fed to the algorithms
          process_num (int): total number of processes for retrieving and
                       preparing the data
          neg_rate (int): This rate determines how many negative triples are generated per positive triple
          data (str): Either train, test or valid.
          algo (str): The algorithm for which the data is being generated.
    """

    def __init__(self, batch_size=128,
                 loss_type='entropy',
                 data_path=Path('..') / 'data',
                 sampling='uniform',
                 queue_size=50,
                 raw_queue_size=50,
                 processed_queue_size=50,
                 process_num=2,
                 data='train',
                 algo='ConvE',
                 neg_rate=2
                 ):
        self.neg_rate = neg_rate
        self.process_num = process_num
        self.raw_queue_size = raw_queue_size
        self.processed_queue_size = processed_queue_size
        self.algo = algo
        self.data = data
        self.queue_size = queue_size
        self.sampling = sampling
        self.data_path = data_path
        self.loss_type = loss_type
        self.batch_size = batch_size