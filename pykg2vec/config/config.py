#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
We store the base configuration of the models here
"""


class Freebase(object):

	def __init__(self):
		self.root_path = '../dataset/Freebase/'
		self.tar  = '../dataset/Freebase/FB15k.tgz'
		self.url  =  "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz"
		self.downloaded_path    = '../dataset/Freebase/FB15k/freebase_mtr100_mte100-'
		self.prepared_data_path = '../dataset/Freebase/FB15k/FB15k_'
		self.entity2idx_path    = '../dataset/Freebase/FB15k/FB15k_entity2idx.pkl'
		self.idx2entity_path    = '../dataset/Freebase/FB15k/FB15k_idx2entity.pkl'
		self.relation2idx_path = '../dataset/Freebase/FB15k/FB15k_relation2idx.pkl'
		self.idx2relation_path = '../dataset/Freebase/FB15k/FB15k_idx2relation.pkl'


class GlobalConfig(object):

	def __init__(self, dataset = 'Freebase', negative_sample = 'uniform'):
		if dataset =='Freebase':
			self.dataset = Freebase()
		else:
			raise NotImplementedError("%s dataset config not found!" % dataset)

		self.negative_sample = negative_sample


class TransEConfig(object):

	def __init__(self,
				 learning_rate  = 0.001,
				 test_flag      = False,
				 l1_flag        = True,
				 hidden_size    = 100,
				 load_from_data = False,
				 batch_size     = 128,
				 epochs = 1000,
				 margin = 1.0,
				 data   = 'Freebase',
				 optimizer  = 'gradient',
				 test_step  = 100,
				 test_num   = 300,
				 triple_num = 5):

		self.learning_rate = learning_rate
		self.testFlag      = test_flag
		self.L1_flag       = l1_flag
		self.loadFromData  = load_from_data
		self.hidden_size   = hidden_size
		self.batch_size    = batch_size
		self.epochs        = epochs
		self.margin        = margin
		self.data          = data
		self.optimizer     = optimizer
		self.test_step     = test_step
		self.test_num 	   = test_num
		self.disp_triple_num = triple_num






