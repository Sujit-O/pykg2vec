import tensorflow as tf
import timeit
import os
import sys

sys.path.append("../")
from core.KGMeta import TrainerMeta
from utils.evaluation import Evaluation
from utils.visualization import Visualization
from utils.generator import Generator
from config.global_config import GeneratorConfig
from utils.dataprep import DataInput, DataInputSimple, DataStats
import numpy as np
import pickle
from scipy import sparse as sps


def get_sparse_mat(data, bs, te):
    mat = np.zeros(shape=(bs, te), dtype=np.int16)
    for i in range(bs):
        for j in range(len(data[i])):
            mat[i][j] = 1
    return mat


class Trainer(TrainerMeta):

    def __init__(self, model, debug=False):
        self.debug = debug
        self.model = model
        self.config = self.model.config
        self.evaluator = Evaluation(model=model, debug=self.debug)
        self.training_results = []
        self.gen_train = None

    def build_model(self):
        """function to build the model"""
        self.sess = tf.Session(config=self.config.gpu_config)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        if self.config.optimizer == 'gradient':
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

    ''' Training related functions:'''

    def train_model(self):
        """function to train the model"""
        if self.config.loadFromData:
            self.load_model()
        else:

            # self.gen_train = Generator(config=GeneratorConfig(data='train', algo=self.model.model_name,
            #                                                   batch_size=self.model.config.batch_size))

            for n_iter in range(self.config.epochs):
                if self.model.model_name == "ProjE_pointwise":
                    self.train_model_epoch_proje(n_iter)
                    self.tiny_test_proje(n_iter)
                elif self.model.model_name.lower() in ["tucker"]:
                    self.train_model_epoch_simple(n_iter)
                    self.tiny_test_simple(n_iter)
                elif self.model.model_name.lower() in ['conve', 'complex', "distmult"]:
                    self.train_model_epoch_conve(n_iter)
                    self.tiny_test_conve(n_iter)
                else:
                    self.train_model_epoch(n_iter)
                    self.tiny_test(n_iter)

            self.gen_train.stop()

            self.evaluator.save_test_summary()
            self.evaluator.save_training_result(self.training_results)

            if self.config.save_model:
                self.save_model()

        if self.config.disp_result:
            self.display()

        if self.config.disp_summary:
            self.summary()

    def train_model_epoch_proje(self, epoch_idx):
        acc_loss = 0

        num_batch = self.gen_train.tot_train_data // self.config.batch_size if not self.debug else 5

        start_time = timeit.default_timer()

        # gen_train = self.data_handler.batch_generator_train_proje(batch_size=self.config.batch_size)
        batch_counts = 0

        for batch_idx in range(num_batch):
            data = list(next(self.gen_train))
            hr_hr = data[0]
            hr_t = data[1]
            tr_tr = data[2]
            tr_h = data[3]

            feed_dict = {
                self.model.hr_h: hr_hr[:, 0],
                self.model.hr_r: hr_hr[:, 1],
                self.model.hr_t: hr_t,
                self.model.tr_t: tr_tr[:, 0],
                self.model.tr_r: tr_tr[:, 1],
                self.model.tr_h: tr_h
            }

            _, step, loss = self.sess.run([self.op_train, self.global_step, self.model.loss], feed_dict)

            acc_loss += loss

            print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time,
                                                        batch_counts, num_batch, loss), end='\r')
            batch_counts += 1
        print('iter[%d] ---Train Loss: %.5f ---time: %.2f' % (
            epoch_idx, acc_loss, timeit.default_timer() - start_time))

        self.training_results.append([epoch_idx, acc_loss])

    def train_model_epoch(self, epoch_idx):
        acc_loss = 0

        num_batch = self.gen_train.tot_train_data // self.config.batch_size if not self.debug else 100

        start_time = timeit.default_timer()

        # if self.config.sampling == "uniform":
        #     gen_train = self.data_handler.batch_generator_train(batch_size=self.config.batch_size)
        # elif self.config.sampling == "bern":
        #     gen_train = self.data_handler.batch_generator_bern(batch_size=self.config.batch_size)

        for batch_idx in range(num_batch):
            data = list(next(self.gen_train))
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

            print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time,
                                                        batch_idx, num_batch, loss), end='\r')

        print('iter[%d] ---Train Loss: %.5f ---time: %.2f' % (
            epoch_idx, acc_loss, timeit.default_timer() - start_time))

        self.training_results.append([epoch_idx, acc_loss])

    def train_model_epoch_simple(self, epoch_idx):
        acc_loss = 0

        with open(str(self.config.tmp_data / 'data_stats.pkl'), 'rb') as f:
            data_stats = pickle.load(f)

        with open(str(self.config.tmp_data / 'train_data.pkl'), 'rb') as f:
            train_data = pickle.load(f)
            rand_ids_train = np.random.permutation(len(train_data))

        # num_batch = self.gen_train.tot_train_data // self.config.batch_size if not self.debug else 10
        num_batch = len(train_data) // self.config.batch_size if not self.debug else 10
        start_time = timeit.default_timer()

        for batch_idx in range(num_batch):
            data = np.asarray([[train_data[x].h,
                                train_data[x].r,
                                train_data[x].t,
                                train_data[x].hr_t
                                ] for x in
                               rand_ids_train[
                               self.config.batch_size * batch_idx: self.config.batch_size * (batch_idx + 1)]])

            h = data[:, 0]
            r = data[:, 1]
            t = data[:, 2]
            hr_t = get_sparse_mat(data[:, 3], self.config.batch_size, data_stats.tot_entity)
            # rt_h = data[4]

            feed_dict = {
                self.model.h: h,
                self.model.r: r,
                self.model.t: t,
                self.model.hr_t: hr_t
                # self.model.rt_h: rt_h
            }

            _, step, loss = self.sess.run([self.op_train, self.global_step, self.model.loss], feed_dict)

            acc_loss += loss

            print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time,
                                                        batch_idx, num_batch, loss), end='\r')

        print('iter[%d] ---Train Loss: %.5f ---time: %.2f' % (
            epoch_idx, acc_loss, timeit.default_timer() - start_time))

        self.training_results.append([epoch_idx, acc_loss])

    def train_model_epoch_conve(self, epoch_idx):
        acc_loss = 0

        num_batch = self.gen_train.tot_train_data // self.config.batch_size if not self.debug else 10

        start_time = timeit.default_timer()

        # gen_train = self.data_handler.batch_generator_train_hr_tr(batch_size=self.config.batch_size)

        for batch_idx in range(num_batch):
            data = list(next(self.gen_train))
            e1 = data[0]
            r = data[1]
            e2_multi1 = data[2]

            feed_dict = {
                self.model.e1: e1,
                self.model.r: r,
                self.model.e2_multi1: e2_multi1
            }

            _, step, loss = self.sess.run([self.op_train, self.global_step, self.model.loss], feed_dict)

            acc_loss += loss

            print('[%.2f sec](%d/%d): -- loss: %.5f' % (timeit.default_timer() - start_time,
                                                        batch_idx, num_batch, loss), end='\r')

        print('iter[%d] ---Train Loss: %.5f ---time: %.2f' % (
            epoch_idx, acc_loss, timeit.default_timer() - start_time))

        self.training_results.append([epoch_idx, acc_loss])

    ''' Testing related functions:'''

    def tiny_test(self, curr_epoch):
        start_time = timeit.default_timer()

        if self.config.test_step == 0:
            return

        if curr_epoch % self.config.test_step == 0 or \
                curr_epoch == 0 or \
                curr_epoch == self.config.epochs - 1:
            # self.evaluator.test(self.sess, curr_epoch)
            self.evaluator.test_step(self.sess, curr_epoch)
            self.evaluator.print_test_summary(curr_epoch)

            print('iter[%d] ---Testing ---time: %.2f' % (curr_epoch, timeit.default_timer() - start_time))

    def tiny_test_simple(self, curr_epoch):
        start_time = timeit.default_timer()

        if self.config.test_step == 0:
            return

        if curr_epoch % self.config.test_step == 0 or \
                curr_epoch == 0 or \
                curr_epoch == self.config.epochs - 1:
            self.evaluator.test_simple(self.sess, curr_epoch)
            self.evaluator.print_test_summary(curr_epoch)

            print('iter[%d] ---Testing ---time: %.2f' % (curr_epoch, timeit.default_timer() - start_time))

    def tiny_test_conve(self, curr_epoch):
        start_time = timeit.default_timer()

        if self.config.test_step == 0:
            return

        if curr_epoch % self.config.test_step == 0 or \
                curr_epoch == 0 or \
                curr_epoch == self.config.epochs - 1:
            self.evaluator.test_conve(self.sess, curr_epoch)
            self.evaluator.print_test_summary(curr_epoch)

            print('iter[%d] ---Testing ---time: %.2f' % (curr_epoch, timeit.default_timer() - start_time))

    def tiny_test_proje(self, curr_epoch):
        start_time = timeit.default_timer()

        if self.config.test_step == 0:
            return

        if curr_epoch % self.config.test_step == 0 or \
                curr_epoch == 0 or \
                curr_epoch == self.config.epochs - 1:
            self.evaluator.test_proje(self.sess, curr_epoch)
            self.evaluator.print_test_summary(curr_epoch)

            print('iter[%d] ---Testing ---time: %.2f' % (curr_epoch, timeit.default_timer() - start_time))

    def full_test(self):
        self.evaluator.test(self.sess, self.config.epochs)
        self.evaluator.print_test_summary(self.config.epochs)
        self.evaluator.save_test_summary(algo=self.model.model_name)

    ''' Procedural functions:'''

    def save_model(self):
        """function to save the model"""
        if not os.path.exists(self.config.tmp):
            os.mkdir('../intermediate')
        saver = tf.train.Saver(self.model.parameter_list)
        saver.save(self.sess, str(self.config.tmp) + '/%s.vec' % self.model.model_name)

    def load_model(self):
        """function to load the model"""
        if not os.path.exists(self.config.tmp):
            os.mkdir('../intermediate')
        saver = tf.train.Saver(self.model.parameter_list)
        saver.restore(self.sess, str(self.config.tmp) + '/%s.vec' % self.model.model_name)

    def display(self):
        """function to display embedding"""
        if self.config.plot_embedding:
            if self.config.plot_entity_only:
                viz = Visualization(model=self.model,
                                    ent_only_plot=True,
                                    rel_only_plot=False,
                                    ent_and_rel_plot=False)
            else:
                viz = Visualization(model=self.model,
                                    ent_only_plot=True,
                                    rel_only_plot=True,
                                    ent_and_rel_plot=True)

            viz.plot_embedding(sess=self.sess, resultpath=self.config.figures, algos=self.model.model_name,
                               show_label=False)

        if self.config.plot_training_result:
            viz = Visualization()
            viz.plot_train_result(path=self.config.result,
                                  result=self.config.figures,
                                  algo=['TransE', 'TransR', 'TransH', 'SLM'],
                                  data=['Freebase15k'])

        if self.config.plot_testing_result:
            viz = Visualization()
            viz.plot_test_result(path=self.config.result,
                                 result=self.config.figures,
                                 algo=['TransE', 'TransR', 'TransH'],
                                 data=['Freebase15k'], paramlist=None, hits=self.config.hits)

    def summary(self):
        """function to print the summary"""
        print("\n----------SUMMARY----------")
        # Acquire the max length and add four more spaces
        maxspace = len(max([k for k in self.config.__dict__.keys()])) + 15
        for key, val in self.config.__dict__.items():
            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            print(key, ":", val)
        print("---------------------------")
