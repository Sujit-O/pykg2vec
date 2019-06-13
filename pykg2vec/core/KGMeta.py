#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Graph Meta Class
====================================
It provides Abstract class for the Knowledge graph models.
"""

from abc import ABCMeta


class ModelMeta:
	""" Meta Class for knowledge graph embedding algorithms"""

	__metaclass__ = ABCMeta

	def __init__(self):
		"""Initialize and create the model to be trained and inferred"""
		pass

	def def_inputs(self):
		"""Function to define the inputs for the model"""
		pass

	def def_parameters(self):
		"""Function to define the parameters for the model"""
		pass

	def def_loss(self):
		"""Function to define how loss is calculated in the model"""
		pass
		
	def embed(self,h, r, t):
		"""Function to get the embedding value"""
		pass

	def get_embed(self,h, r, t):
		"""Function to get the embedding value in numpy"""
		pass

	def get_proj_embed(self,h, r, t):
		"""Function to get the projected embedding value"""
		pass


class TrainerMeta:
	""" Meta Class for Trainer Module"""
	__metaclass__ = ABCMeta

	def __init__(self):
		"""Initializing and create the model to be trained and inferred"""
		pass

	def build_model(self):
		"""function to compile the model"""
		pass

	def train_model(self):
		"""function to train the model"""
		pass

	def save_model(self, sess):
		"""function to save the model"""
		pass

	def load_model(self, sess):
		"""function to load the model"""
		pass


class VisualizationMeta:
	""" Meta Class for Visualization Module"""
	__metaclass__ = ABCMeta
	
	def __init__(self):
		"""Initializing and create the model to be trained and inferred"""
		pass
		
	def display(self):
		"""function to display embedding"""
		pass

	def summary(self):
		"""function to print the summary"""
		pass


class EvaluationMeta:
	""" Meta Class for Evaluation Module"""
	__metaclass__ = ABCMeta

	def __init__(self):
		pass

	def relation_prediction(self):
		"""Function for evaluating link prediction"""
		pass

	def entity_classification(self):
		"""Function for evaluating entity classification"""
		pass

	def relation_classification(self):
		"""Function for evaluating relation classification"""
		pass

	def triple_classification(self):
		"""Function for evaluating triple classificaiton"""
		pass

	def entity_completion(self):
		"""Function for evaluating entity completion"""
		pass
