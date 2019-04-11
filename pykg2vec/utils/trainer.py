import sys
sys.path.append("../")
from core.KGMeta import ModelMeta, TrainerMeta
import pandas as pd
import tensorflow as tf
import timeit
from argparse import ArgumentParser
import os
from utils.evaluation import Evaluation


class Trainer(TrainerMeta):

    def __init__(self, model):
        self.model = model
        self.config = self.model.config
        self.data_handler = self.model.data_handler

        self.training_results = []
    
    def build_model(self):
        self.sess = tf.Session(config=self.model.config.gpu_config)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if self.config.optimizer == 'gradient':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        else:
            raise NotImplementedError("No support for %s optimizer" % self.config.optimizer)
        grads = optimizer.compute_gradients(self.model.loss)
        self.op_train = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())
    
    def train_model(self):
        """function to train the model"""

        if self.config.loadFromData:
            self.model.load_model()

        evaluate = Evaluation(model=self.model, test_data='test')

        for n_iter in range(self.config.epochs):

            acc_loss = 0
            batch = 0
            num_batch = 5  # len(self.data_handler.train_triples_ids) // self.config.batch_size
            start_time = timeit.default_timer()
            
            gen_train = self.data_handler.batch_generator_train(batch_size=self.config.batch_size)

            for i in range(num_batch):
                ph, pr, pt, nh, nr, nt = list(next(gen_train))

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
                batch += 1
                print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time,
                                                            batch,
                                                            num_batch,
                                                            loss), end='\r')

            print('iter[%d] ---Train Loss: %.5f ---time: %.2f' % (
                n_iter, acc_loss, timeit.default_timer() - start_time))
            
            self.training_results.append([n_iter, acc_loss])
            if n_iter % self.config.test_step == 0 or n_iter == 0 or n_iter == self.config.epochs - 1:
                evaluate.test(self.sess, n_iter)
                evaluate.print_test_summary(n_iter)

        evaluate.save_test_summary(algo=self.model.model_name)
        evaluate.save_training_result(self.training_results)

        if self.config.save_model:
            self.model.save_model(self.sess)
               
        if self.config.disp_result:
            self.model.display(self.sess)

        if self.config.disp_summary:
            self.model.summary()

    def save_model(self):
        """function to save the model"""
        if not os.path.exists(self.config.tmp):
            os.mkdir('../intermediate')
        # TODO
        # saver = tf.train.Saver()
        # saver.save(self.sess, self.config.tmp + '/TransEModel.vec')

    def load_model(self):
        """function to load the model"""
        if not os.path.exists(self.config.tmp):
            os.mkdir('../intermediate')
        # TODO
        # saver = tf.train.Saver()
        # saver.restore(self.sess, self.config.tmp + '/TransEModel.vec')
    