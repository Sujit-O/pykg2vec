#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Abstract class for the Knowledge graph models"""
from abc import ABCMeta

class KGMeta:
	__metaclass__ = ABCMeta

	def __init__(self):
		"""Initializing the and create the model"""		
		pass

	def train(self):
		"""function to train the model"""
		pass

	def test(self):
		"""function to test the model"""
		pass

	def embed(self,h, r, t):
		"""function to get the embedding value"""
		pass

	def display(self):
		"""function to display embedding"""
		pass

	def save_model(self, sess):
		"""function to save the model"""
		pass

	def load_model(self, sess):
		"""function to load the model"""
		pass

	def summary(self):
		"""function to print the summary"""
		pass


class Evaluation:
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	def relation_prediction(self):
		pass

	def entity_classification(self):
		pass

	def relation_classification(self):
		pass

	def triple_classification(self):
		pass

	def entity_completion(self):
		pass
