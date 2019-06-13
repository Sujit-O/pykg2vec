#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for training process.
"""
import tensorflow as tf
import timeit

import numpy as np

from pykg2vec.core.KGMeta import TrainerMeta
from pykg2vec.utils.evaluation import Evaluation
from pykg2vec.utils.visualization import Visualization
from pykg2vec.utils.generator import Generator
from pykg2vec.config.global_config import GeneratorConfig
from pykg2vec.utils.kgcontroller import KGMetaData, KnowledgeGraph

def get_sparse_mat(data, bs, te):
    """Function to get the sparse matrix.
           
        Args:
            data (list): List containing the positive triple integer ID.
            bs (int): Batch size.
            te (int): Total entities.
            
        Returns:
            Matrix: Returns the numpy matrix
    """
    mat = np.zeros(shape=(bs, te), dtype=np.int16)
    for i in range(bs):
        for j in range(len(data[i])):
            mat[i][j] = 1
    return mat


class Trainer(TrainerMeta):
    """Class for handling the training of the algorithms.

        Args:
            model (object): Model object
            debug (bool): Flag to check if its debugging
            tuning (bool): Flag to denoting tuning if True

        Examples:
            >>> from pykg2vec.utils.trainer import Trainer
            >>> from pykg2vec.core.TransE import TransE
            >>> trainer = Trainer(model=TransE(), debug=False)
            >>> trainer.build_model()
            >>> trainer.train_model()
    """

    def __init__(self, model, debug=False, tuning=False):
        self.debug = debug
        self.model = model
        self.config = self.model.config
        self.training_results = []
        self.gen_train = None
        self.tuning=tuning

    def build_model(self):
        """function to build the model"""
        self.model.def_inputs()
        self.model.def_parameters()
        if getattr(self.model, "def_layer", None):
            self.model.def_layer()
        self.model.def_loss()

        if not self.debug:
            self.sess = tf.Session(config=self.config.gpu_config)
        else:
            self.sess = tf.InteractiveSession()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        if self.config.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.config.learning_rate)
        else:
            raise NotImplementedError("No support for %s optimizer" % self.config.optimizer)

        grads = optimizer.compute_gradients(self.model.loss)
        self.op_train = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())

        if not self.tuning:
            self.summary()
            self.summary_hyperparameter()

    ''' Training related functions:'''

    def train_model(self):
        """Function to train the model."""
        loss = 0

        if self.config.loadFromData:
            self.load_model()
        else:
            generator_config = GeneratorConfig(data='train', algo=self.model.model_name,
                                               batch_size=self.model.config.batch_size)
            self.gen_train = Generator(config=generator_config, model_config=self.model.config)

            if not self.tuning:
                self.evaluator = Evaluation(model=self.model, debug=self.debug)

            for n_iter in range( self.config.epochs):
                loss = self.train_model_epoch(n_iter)
                if not self.tuning:
                    self.test(n_iter)

            self.gen_train.stop()

            if not self.tuning:
                self.evaluator.save_training_result(self.training_results)
                self.evaluator.stop()

            if self.config.save_model:
                self.save_model()

        if self.config.disp_result:
            self.display()

        if self.config.disp_summary:
            self.summary()
            self.summary_hyperparameter()

        self.sess.close()
        tf.reset_default_graph() # clean the tensorflow for the next training task.

        return loss

    def tune_model(self):
        """Function to tune the model."""
        acc = 0

        generator_config = GeneratorConfig(data='train', algo=self.model.model_name,
                                           batch_size=self.model.config.batch_size)
        self.gen_train = Generator(config=generator_config, model_config=self.model.config)

        self.evaluator = Evaluation(model=self.model, debug=self.debug, tuning=True)
       
        for n_iter in range( self.config.epochs):
            self.train_model_epoch(n_iter)

        self.gen_train.stop()
        self.evaluator.test_batch(self.sess, n_iter)
        acc = self.evaluator.output_queue.get()
        self.evaluator.stop()
        self.sess.close()
        tf.reset_default_graph() # clean the tensorflow for the next training task.

        return acc

    def train_model_epoch(self, epoch_idx):
        """Function to train the model for one epoch."""
        acc_loss = 0

        num_batch = self.model.config.kg_meta.tot_train_triples // self.config.batch_size if not self.debug else 10

        start_time = timeit.default_timer()

        for batch_idx in range(num_batch):
            data = list(next(self.gen_train))
            if self.model.model_name.lower() in ["tucker", "tucker_v2", "conve", "complex", "distmult", "proje_pointwise"]:
                h = data[0]
                r = data[1]
                t = data[2]
                hr_t = data[3]
                rt_h = data[4]

                feed_dict = {
                    self.model.h: h,
                    self.model.r: r,
                    self.model.t: t,
                    self.model.hr_t: hr_t,
                    self.model.rt_h: rt_h
                }
            else:
                ph = data[0]
                pr = data[1]
                pt = data[2]
                nh = data[3]
                nr = data[4]
                nt = data[5]

                feed_dict = {
                    self.model.pos_h: ph,
                    self.model.pos_t: pt,
                    self.model.pos_r: pr,
                    self.model.neg_h: nh,
                    self.model.neg_t: nt,
                    self.model.neg_r: nr
                }

            _, step, loss = self.sess.run([self.op_train, self.global_step, self.model.loss], feed_dict)

            acc_loss += loss

            if not self.tuning:
                print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time,
                                                            batch_idx, num_batch, loss), end='\r')
        if not self.tuning:
            print('iter[%d] ---Train Loss: %.5f ---time: %.2f' % (
                epoch_idx, acc_loss, timeit.default_timer() - start_time))

        self.training_results.append([epoch_idx, acc_loss])

        return acc_loss

    ''' Testing related functions:'''

    def test(self, curr_epoch):
        """function to test the model.
           
           Args:
                curr_epoch (int): The current epoch number.
        """

        if not self.config.full_test_flag and (curr_epoch % self.config.test_step == 0 or
                                               curr_epoch == 0 or
                                               curr_epoch == self.config.epochs - 1):
            self.evaluator.test_batch(self.sess, curr_epoch)
        else:
            if curr_epoch == self.config.epochs - 1:
                self.evaluator.test_batch(self.sess, curr_epoch)

    ''' Procedural functions:'''

    def save_model(self):
        """Function to save the model."""
        saved_path = self.config.tmp / self.model.model_name
        saved_path.mkdir(parents=True, exist_ok=True)

        saver = tf.train.Saver(self.model.parameter_list)
        saver.save(self.sess, str(saved_path / 'model.vec'))

    def load_model(self):
        """Function to load the model."""
        saved_path = self.config.tmp / self.model.model_name
        if saved_path.exists():
            saver = tf.train.Saver(self.model.parameter_list)
            saver.restore(self.sess, str(saved_path / 'model.vec'))

    def display(self):
        """Function to display embedding."""
        options = {"ent_only_plot": True,
                    "rel_only_plot": not self.config.plot_entity_only,
                    "ent_and_rel_plot": not self.config.plot_entity_only}

        if self.config.plot_embedding:
            viz = Visualization(model=self.model, vis_opts = options)

            viz.plot_embedding(sess=self.sess,
                               resultpath=self.config.figures,
                               algos=self.model.model_name,
                               show_label=False)

        if self.config.plot_training_result:
            viz = Visualization(model=self.model)
            viz.plot_train_result()

        if self.config.plot_testing_result:
            viz = Visualization(model=self.model)
            viz.plot_test_result()

    def summary(self):
        """Function to print the summary."""
        print("\n------------------Global Setting--------------------")
        # Acquire the max length and add four more spaces
        maxspace = len(max([k for k in self.config.__dict__.keys()])) +20
        for key, val in self.config.__dict__.items():
            if key in self.config.__dict__['hyperparameters']:
                continue

            if isinstance(val, (KGMetaData, KnowledgeGraph)) or key.startswith('gpu') or key.startswith('hyperparameters'):
                continue

            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            print("%s : %s"%(key, val))
        print("---------------------------------------------------")

    def summary_hyperparameter(self):
        """Function to print the hyperparameter summary."""
        print("\n-----------%s Hyperparameter Setting-------------"%(self.model.model_name))
        maxspace = len(max([k for k in self.config.hyperparameters.keys()])) + 15
        for key,val in self.config.hyperparameters.items():
            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            print("%s : %s" % (key, val))
        print("---------------------------------------------------")
