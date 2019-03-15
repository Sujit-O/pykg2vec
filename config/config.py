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

class GlobalConfig(object):
	def __init__(self, dataset = 'Freebase', batch = 128):
		if dataset =='Freebase':
			self.dataset = Freebase()
		else:
			print("Invalid Dataset!")
			return

		self.batch = batch


class TransEConfig(object):
	def __init__(self):
		self.learning_rate = 0.001
		self.testFlag      = False
		self.loadFromData  = False
		self.L1_flag       = True
		self.hidden_size   = 100
		self.nbatches      = 100
		self.entity        = 0
		self.relation      = 0
		self.trainTimes    = 1000
		self.margin        = 1.0





