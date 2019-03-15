#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
------------------Paper Title-----------------------------
Translating Embeddings for Modeling Multi-relational Data
------------------Paper Authors---------------------------
Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran
Universite de Technologie de Compiegne – CNRS
Heudiasyc UMR 7253
Compiegne, France
{bordesan, nusunier, agarciad}@utc.fr
Jason Weston, Oksana Yakhnenko
Google
111 8th avenue
New York, NY, USA
{jweston, oksana}@google.com
------------------Summary---------------------------------
TransE is an energy based model which represents the
relationships as translations in the embedding space. Which
means that if (h,l,t) holds then the embedding of the tail
't' should be close to the embedding of head entity 'h'
plus some vector that depends on the relationship 'l'.
Both entities and relations are vectors in the same space.
|        ......>.
|      .     .
|    .    .
|  .  .
|_________________
Code Based on https://github.com/thunlp/OpenKE/blob/master/models/TransE.py
"""
from KGMeta import KGMeta
import numpy as np
import tensorflow as tf
import os
from config.config import TransEConfig
from utils.dataprep import DataPrep

class  TransE(KGMeta):
	def __init__(self, config=None):
		""" TransE Models
		Args:
		-----Inputs-------
		"""
		if not config:
			self.model_config = TransEConfig()
		else:
			self.model_config = config

		self.pos_h = tf.placeholder(tf.int32, [None])
		self.pos_t = tf.placeholder(tf.int32, [None])
		self.pos_r = tf.placeholder(tf.int32, [None])

		self.neg_h = tf.placeholder(tf.int32, [None])
		self.neg_t = tf.placeholder(tf.int32, [None])
		self.neg_r = tf.placeholder(tf.int32, [None])

		with tf.name_scope("embedding"):
			self.ent_embeddings = tf.get_variable(name = "ent_embedding",\
			shape = [self.model_config.entity, self.model_config.hidden_size],\
			initializer = tf.contrib.layers.xavier_initializer(uniform = False))

			self.rel_embeddings = tf.get_variable(name = "rel_embedding",\
			shape = [self.model_config.relation, self.model_config.hidden_size],\
			initializer = tf.contrib.layers.xavier_initializer(uniform = False))

			pos_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
			pos_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
			pos_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
			neg_h_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
			neg_t_e = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
			neg_r_e = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

		if self.model_config.L1_flag:
			pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
			neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
			self.predict = pos
		else:
			pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
			neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)
			self.predict = pos

		with tf.name_scope("output"):
			self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.model_config.margin, 0))

	def readKGraph(self, logdir=None):
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

def main(_):
	data_handler = DataPrep('Freebase')
	data_handler.prepare_data()
	# gen = data_handler.batch_generator()
	config = TransEConfig()
	if config.testFlag:
		config.relation = data_handler.tot_r
		config.entity = data_handler.tot_entity

	else:
		config.relation = data_handler.tot_r
		config.entity = data_handler.tot_entity
		config.batches = data_handler.tot_triple // config.batch_size
	
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			initializer = tf.contrib.layers.xavier_initializer(uniform = False)
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				trainModel = TransE(config = config)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.GradientDescentOptimizer(config.learning_rate)
			grads_and_vars = optimizer.compute_gradients(trainModel.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())
			if config.loadFromData:
				saver.restore(sess, './intermediate/TransEModel.vec')

			def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
				feed_dict = {
					trainModel.pos_h: pos_h_batch,
					trainModel.pos_t: pos_t_batch,
					trainModel.pos_r: pos_r_batch,
					trainModel.neg_h: neg_h_batch,
					trainModel.neg_t: neg_t_batch,
					trainModel.neg_r: neg_r_batch
				}
				_, step, loss = sess.run(
					[train_op, global_step, trainModel.loss], feed_dict)
				return loss

			def test_step(pos_h_batch, pos_t_batch, pos_r_batch):
				feed_dict = {
					trainModel.pos_h: pos_h_batch,
					trainModel.pos_t: pos_t_batch,
					trainModel.pos_r: pos_r_batch,
				}
				step, predict = sess.run(
					[global_step, trainModel.predict], feed_dict)
				return predict

			if not config.testFlag:
				gen_train = data_handler.batch_generator(batch=config.batch_size)

				for times in range(config.epochs):
					res = 0.0
					for i in range(config.batches):
						ph, pt, pr, nh, nt, nr = list(next(gen_train))
						res += train_step(ph, pt, pr, nh, nt, nr)
						current_step = tf.train.global_step(sess, global_step)
					print(times)
					print(res)
				if not os.path.exists('./intermediate'):
					os.mkdir('./intermediate')
				saver.save(sess, './intermediate/TransEModel.vec')
			else:
				total = data_handler.testTriple
				gen_test = data_handler.batch_generator(batch=config.batch_size, data='test')
				for times in range(total):
					ph, pt, pr = list(next(gen_test))
					res = test_step(ph, pt, pr)
					print(res)
				# 	if (times % 50 == 0):
				# 		test_lib.test()
				# test_lib.test()

if __name__ == "__main__":
	tf.app.run()
		
