#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for training process.
"""
import tensorflow as tf
import timeit

import numpy as np
import os 

from pykg2vec.core.KGMeta import TrainerMeta
from pykg2vec.utils.evaluation import Evaluation
from pykg2vec.utils.visualization import Visualization
from pykg2vec.utils.generator import Generator
from pykg2vec.config.global_config import GeneratorConfig
from pykg2vec.utils.kgcontroller import KGMetaData, KnowledgeGraph


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

    def __init__(self, model, trainon='train',teston='valid',debug=False, tuning=False):
        self.debug = debug
        self.model = model
        self.config = self.model.config
        self.training_results = []
        self.gen_train = None
        self.tuning=tuning
        self.trainon = trainon
        self.teston = teston

    def enter_interactive_mode(self):
        self.build_model()
        self.load_model()

        print("The training/loading of the model has finished!\nNow enter interactive mode :)")
        print("-----")
        print("Example 1: trainer.infer_tails(1,10,topk=5)")
        self.infer_tails(1,10,topk=5)

        print("-----")
        print("Example 2: trainer.infer_heads(10,20,topk=5)")
        self.infer_heads(10,20,topk=5)
    
    def exit_interactive_mode(self):
        self.sess.close()
        tf.reset_default_graph() # clean the tensorflow for the next training task.

        print("Thank you for trying out inference interactive script :)")

    def infer_tails(self,h,r,topk=5):
        tails_op = self.model.infer_tails(h,r,topk)
        tails = self.sess.run(tails_op)
        print("\n(head, relation)->({},{}) :: Inferred tails->({})\n".format(h,r,",".join([str(i) for i in tails])))
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')
        print("head: %s" % idx2ent[h])
        print("relation: %s" % idx2rel[r])

        for idx, tail in enumerate(tails):
            print("%dth predicted tail: %s" % (idx, idx2ent[tail]))

    def infer_heads(self,r,t,topk=5):
        heads_op = self.model.infer_heads(r,t,topk)
        heads = self.sess.run(heads_op)
        
        print("\n(relation,tail)->({},{}) :: Inferred heads->({})\n".format(t,r,",".join([str(i) for i in heads])))
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')
        print("tail: %s" % idx2ent[t])
        print("relation: %s" % idx2rel[r])

        for idx, head in enumerate(heads):
            print("%dth predicted head: %s" % (idx, idx2ent[head]))

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
        self.saver =  tf.train.Saver()
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
            generator_config = GeneratorConfig(data=self.trainon, algo=self.model.model_name,
                                               batch_size=self.model.config.batch_size)
            self.gen_train = Generator(config=generator_config, model_config=self.model.config)

            if not self.tuning:
                self.evaluator = Evaluation(model=self.model, data_type=self.teston, debug=self.debug, session=self.sess)

            for n_iter in range(self.config.epochs):
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

        if not os.path.exists("./tmp"):
            os.mkdir("./tmp")

        self.export_embeddings()

        self.sess.close()
        tf.reset_default_graph() # clean the tensorflow for the next training task.

        return loss

    def tune_model(self):
        """Function to tune the model."""
        acc = 0

        generator_config = GeneratorConfig(data=self.trainon, algo=self.model.model_name,
                                           batch_size=self.model.config.batch_size)
        self.gen_train = Generator(config=generator_config, model_config=self.model.config)

        self.evaluator = Evaluation(model=self.model,data_type=self.teston, debug=self.debug, tuning=True, session=self.sess)
       
        for n_iter in range( self.config.epochs):
            self.train_model_epoch(n_iter)

        self.gen_train.stop()
        self.evaluator.test_batch(n_iter)
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
            if self.model.model_name.lower() in ["tucker", "tucker_v2", "conve","convkb", "complex", "distmult", "proje_pointwise"]:
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
            self.evaluator.test_batch(curr_epoch)
        else:
            if curr_epoch == self.config.epochs - 1:
                self.evaluator.test_batch(curr_epoch)

    ''' Procedural functions:'''

    def save_model(self):
        """Function to save the model."""
        saved_path = self.config.path_tmp / self.model.model_name
        saved_path.mkdir(parents=True, exist_ok=True)

        saver = tf.train.Saver(self.model.parameter_list)
        saver.save(self.sess, str(saved_path / 'model.vec'))

    def load_model(self):
        """Function to load the model."""
        saved_path = self.config.path_tmp / self.model.model_name
        if saved_path.exists():
            saver = tf.train.Saver(self.model.parameter_list)
            saver.restore(self.sess, str(saved_path / 'model.vec'))

    def display(self):
        """Function to display embedding."""
        options = {"ent_only_plot": True,
                    "rel_only_plot": not self.config.plot_entity_only,
                    "ent_and_rel_plot": not self.config.plot_entity_only}

        if self.config.plot_embedding:
            viz = Visualization(model=self.model, vis_opts = options, sess=self.sess)

            viz.plot_embedding(resultpath=self.config.figures,
                               algos=self.model.model_name,
                               show_label=False)

        if self.config.plot_training_result:
            viz = Visualization(model=self.model, sess=self.sess)
            viz.plot_train_result()

        if self.config.plot_testing_result:
            viz = Visualization(model=self.model, sess=self.sess)
            viz.plot_test_result()
    
    def export_embeddings(self):
        """
            Export embeddings in tsv format. 
            With tsvs (both label, vector files), you can:
            1) Use those pretained embeddings for your applications.  
            2) Visualize the embeddings in this website to gain insights. (https://projector.tensorflow.org/)
        """
        save_path = self.config.path_embeddings / self.model.model_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')

        with open(str(save_path / "ent_labels.tsv"), 'w') as l_export_file:
            for label in idx2ent.values():
                l_export_file.write(label + "\n")

        with open(str(save_path / "rel_labels.tsv"), 'w') as l_export_file:
            for label in idx2rel.values():
                l_export_file.write(label + "\n")

        for parameter in self.model.parameter_list:
            all_ids = list(range(0, int(parameter.shape[0])))
            stored_name = parameter.name.split(':')[0]
            # import pdb; pdb.set_trace()

            if len(parameter.shape) == 2:
                op_get_all_embs = tf.nn.embedding_lookup(parameter, all_ids)
                all_embs = self.sess.run(op_get_all_embs)
                with open(str(save_path / ("%s.tsv" % stored_name)), 'w') as v_export_file:
                    for idx in all_ids:
                        v_export_file.write("\t".join([str(x) for x in all_embs[idx]]) + "\n")
                    
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
