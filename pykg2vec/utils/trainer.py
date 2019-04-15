import tensorflow as tf
import timeit
import os
from pykg2vec.core.KGMeta import TrainerMeta
from pykg2vec.utils.evaluation import Evaluation
from pykg2vec.utils.visualization import Visualization


class Trainer(TrainerMeta):

    def __init__(self, model):
        self.model = model
        self.config = self.model.config
        self.data_handler = self.model.data_handler

        self.evaluator = Evaluation(model=model, test_data='test')
        self.training_results = []
    
    def build_model(self):
        """function to build the model"""

        self.model.def_inputs()
        self.model.def_parameters()
        self.model.def_loss()
        
        self.sess = tf.Session(config=self.config.gpu_config)
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
    
    ''' Training related functions:'''
    def train_model(self):
        """function to train the model"""
        if self.config.loadFromData:
            self.load_model() 
        else:
            for n_iter in range(self.config.epochs):
                self.train_model_epoch(n_iter)                
                self.tiny_test(n_iter)

            self.evaluator.save_test_summary(algo=self.model.model_name)
            self.evaluator.save_training_result(self.training_results)

            if self.config.save_model:
                self.save_model()
                   
        if self.config.disp_result:
            self.display()

        if self.config.disp_summary:
            self.summary()

    def train_model_epoch(self, epoch_idx, debug=False):
        acc_loss = 0
        num_batch = len(self.data_handler.train_triples_ids) // self.config.batch_size if not debug else 5

        start_time = timeit.default_timer()
        
        gen_train = self.data_handler.batch_generator_train(batch_size=self.config.batch_size)

        for batch_idx in range(num_batch):

            ph, pr, pt, nh, nr, nt = next(gen_train)

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

    ''' Testing related functions:'''
    def tiny_test(self, curr_epoch):
        if self.config.test_step == 0:
            return 
            
        if curr_epoch % self.config.test_step == 0 or \
           curr_epoch == 0 or \
           curr_epoch == self.config.epochs - 1:

            self.evaluator.test(self.sess, curr_epoch)
            self.evaluator.print_test_summary(curr_epoch)

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
        saver.save(self.sess, self.config.tmp + '/%s.vec' % self.model.model_name)

    def load_model(self):
        """function to load the model"""
        if not os.path.exists(self.config.tmp):
            os.mkdir('../intermediate')
        saver = tf.train.Saver(self.model.parameter_list)
        saver.restore(self.sess, self.config.tmp + '/%s.vec' % self.model.model_name)

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
                                  algo=['TransE', 'TransR', 'TransH'],
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
        maxspace = len(max([k for k in self.config.__dict__.keys()])) + 4
        for key, val in self.config.__dict__.items():
            if len(key) < maxspace:
                for i in range(maxspace - len(key)):
                    key = ' ' + key
            print(key, ":", val)
        print("---------------------------")