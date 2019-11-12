#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Graph Meta Class
====================================
It provides Abstract class for the Knowledge graph models.
"""

from abc import ABCMeta, abstractmethod


class ModelMeta:
	""" Meta Class for knowledge graph embedding algorithms"""

	__metaclass__ = ABCMeta

	def __init__(self):
		"""Initialize and create the model to be trained and inferred"""
		pass

	@abstractmethod
	def def_inputs(self):
		"""Function to define the inputs for the model"""
		pass

	@abstractmethod
	def def_parameters(self):
		"""Function to define the parameters for the model"""
		pass

	@abstractmethod
	def def_loss(self):
		"""Function to define how loss is calculated in the model"""
		pass

	@abstractmethod
	def embed(self,h, r, t):
		"""Function to get the embedding value"""
		pass

	@abstractmethod
	def get_embed(self,h, r, t):
		"""Function to get the embedding value in numpy"""
		pass

	@abstractmethod
	def get_proj_embed(self,h, r, t):
		"""Function to get the projected embedding value"""
		pass


class TrainerMeta:
	""" Meta Class for Trainer Module"""
	__metaclass__ = ABCMeta

	def __init__(self):
		"""Initializing and create the model to be trained and inferred"""
		pass

	@abstractmethod
	def build_model(self):
		"""function to compile the model"""
		pass

	@abstractmethod
	def train_model(self):
		"""function to train the model"""
		pass

	@abstractmethod
	def save_model(self, sess):
		"""function to save the model"""
		pass

	@abstractmethod
	def load_model(self, sess):
		"""function to load the model"""
		pass


class VisualizationMeta:
	""" Meta Class for Visualization Module"""
	__metaclass__ = ABCMeta
	
	def __init__(self):
		"""Initializing and create the model to be trained and inferred"""
		pass

	@abstractmethod
	def display(self):
		"""function to display embedding"""
		pass

	@abstractmethod
	def summary(self):
		"""function to print the summary"""
		pass


class EvaluationMeta:
	""" Meta Class for Evaluation Module"""
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def relation_prediction(self):
		"""Function for evaluating link prediction"""
		pass

	@abstractmethod
	def entity_classification(self):
		"""Function for evaluating entity classification"""
		pass

	@abstractmethod
	def relation_classification(self):
		"""Function for evaluating relation classification"""
		pass

	@abstractmethod
	def triple_classification(self):
		"""Function for evaluating triple classificaiton"""
		pass

	@abstractmethod
	def entity_completion(self):
		"""Function for evaluating entity completion"""
		pass

class InferenceMeta:
	""" Meta Class for inference based on distance measure"""
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	@abstractmethod
	def def_parameters(self):
		"""Function to define the parameters for the model"""
		pass

	@abstractmethod
	def distance(self, h, r, t, axis):
		"""Function to calculate distance measure in embedding space"""
		pass

	@abstractmethod
	def infer_tails(self, h, r, topk):
		"""Function to infer top k tails for given head and relation"""
		pass

	@abstractmethod
	def infer_heads(self, r, t, topk):
		"""Function to infer top k head for given relation and tail"""
		pass

	@abstractmethod
	def infer_rels(self, h, t, topk):
		"""Function to infer top k relations for given head and tail"""
		pass
