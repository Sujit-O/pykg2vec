#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Abstract class for the Knowledge graph models"""
from abc import ABCMeta

class KGMeta:
	__metaclass__ = ABCMeta

	def __init__(self):
		"""Initializing the class"""
		pass

	def readKGraph(self):
		"""function to read Knowledge graph"""
		pass

	def train(self):
		"""function to train the model"""
		pass

	def test(self):
		"""function to test the model"""
		pass

	def embed(self):
		"""function to get the embedding value"""
		pass

	def display(self):
		"""function to display embedding"""
		pass

	def saveModel(self):
		"""function to save the model"""
		pass

	def loadModel(self):
		"""function to load the model"""
		pass

	def summary(self):
		"""function to print the summary"""
		pass