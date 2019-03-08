#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
We store the base configuration of the models here
"""
class GlobalConfig(object):
	def __init__(self):
		self.url_FB15  =  "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz"
		self.path_FB15 = '../dataset/Freebase/FB15k/freebase_mtr100_mte100-'


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





