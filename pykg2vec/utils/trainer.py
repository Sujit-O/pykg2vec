#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for training process.
"""
import timeit, os
import tensorflow as tf
import pandas as pd

from pykg2vec.core.KGMeta import TrainerMeta
from pykg2vec.utils.evaluator import Evaluator
from pykg2vec.utils.visualization import Visualization
from pykg2vec.utils.generator import Generator, TrainingStrategy
from pykg2vec.utils.kgcontroller import KnowledgeGraph

tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU') 
try:
    for gpu in physical_devices: 
        tf.config.experimental.set_memory_growth(gpu, True) 
except: 
  # Invalid device or cannot modify virtual devices once initialized. 
  pass 
  
class Trainer(TrainerMeta):
    """Class for handling the training of the algorithms.

        Args:
            model (object): Model object
            debug (bool): Flag to check if its debugging
            tuning (bool): Flag to denoting tuning if True
            patience (int): Number of epochs to wait before early stopping the training on no improvement.
            No early stopping if it is a negative number (default: {-1}).

        Examples:
            >>> from pykg2vec.utils.trainer import Trainer
            >>> from pykg2vec.core.TransE import TransE
            >>> trainer = Trainer(TransE())
            >>> trainer.build_model()
            >>> trainer.train_model()
    """
    def __init__(self, model):
        self.model = model
        self.config = model.config

        self.training_results = []

        self.evaluator = None
        self.generator = None

    def build_model(self):
        """function to build the model"""
        if self.config.optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'rms':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.config.learning_rate)
        elif self.config.optimizer == 'adadelta':
            self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.config.learning_rate)
        else:
            raise NotImplementedError("No support for %s optimizer" % self.config.optimizer)
        
        # For optimizer that has not supported gpu computation in TF2, place parameters in cpu. 
        if self.config.optimizer in ['rms', 'adagrad', 'adadelta']:
            with tf.device('cpu:0'):
                self.model.def_parameters()
        else:
            self.model.def_parameters()

        self.config.summary()
        self.config.summary_hyperparameter(self.model.model_name)

    ''' Training related functions:'''
    @tf.function
    def train_step_pairwise(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        with tf.GradientTape() as tape:
            loss = self.model.get_loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def train_step_projection(self, h, r, t, hr_t, rt_h):
        with tf.GradientTape() as tape:
            loss = self.model.get_loss(h, r, t, hr_t, rt_h)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def train_step_pointwise(self, h, r, t, y):
        with tf.GradientTape() as tape:
            loss = self.model.get_loss(h, r, t, y)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train_model(self):
        """Function to train the model."""
        ### Early Stop Mechanism
        loss = previous_loss = float("inf")
        patience_left = self.config.patience
        ### Early Stop Mechanism

        self.generator = Generator(self.model)
        self.evaluator = Evaluator(self.model)

        if self.config.loadFromData:
            self.load_model()
        
        for cur_epoch_idx in range(self.config.epochs):
            print("Epoch[%d/%d]"%(cur_epoch_idx,self.config.epochs))
            loss = self.train_model_epoch(cur_epoch_idx)

            if cur_epoch_idx % self.config.test_step == 0:
                mini_result = self.evaluator.mini_test(cur_epoch_idx)
            
            ### Early Stop Mechanism
            ### start to check if the loss is still decreasing after an interval. 
            ### Example, if early_stop_epoch == 50, the trainer will check loss every 50 epoche.
            ### TODO: change to support different metrics.
            if ((cur_epoch_idx + 1) % self.config.early_stop_epoch) == 0: 
                if patience_left > 0 and previous_loss <= loss:
                    patience_left -= 1
                    print('%s more chances before the trainer stops the training. (prev_loss, curr_loss): (%.f, %.f)' % \
                        (patience_left, previous_loss, loss))

                elif patience_left == 0 and previous_loss <= loss:
                    print('Stop the training.')
                    break
                else:
                    patience_left = self.config.patience

            previous_loss = loss
            ### Early Stop Mechanism

        self.evaluator.full_test(cur_epoch_idx)

        self.generator.stop()
        self.save_training_result()

        if self.config.save_model:
            self.save_model()

        if self.config.disp_result:
            self.display()

        if self.config.disp_summary:
            self.config.summary()
            self.config.summary_hyperparameter(self.model.model_name)

        self.export_embeddings()

        return cur_epoch_idx # the runned epoches.

    def tune_model(self):
        """Function to tune the model."""
        acc = 0
        ### Early Stop Mechanism
        loss = previous_loss = float("inf")
        patience_left = self.config.patience
        ### Early Stop Mechanism

        self.generator = Generator(self.model)
        self.evaluator = Evaluator(self.model, tuning=True)
       
        for cur_epoch_idx in range(self.config.epochs):
            loss = self.train_model_epoch(cur_epoch_idx, tuning=True)
            ### Early Stop Mechanism
            ### start to check if the loss is still decreasing after an interval. 
            ### Example, if early_stop_epoch == 50, the trainer will check loss every 50 epoche.
            ### TODO: change to support different metrics.
            if ((cur_epoch_idx + 1) % self.config.early_stop_epoch) == 0: 
                if patience_left > 0 and previous_loss <= loss:
                    patience_left -= 1
                    print('%s more chances before the trainer stops the training. (prev_loss, curr_loss): (%.f, %.f)' % \
                        (patience_left, previous_loss, loss))

                elif patience_left == 0 and previous_loss <= loss:
                    print('Stop the training.')
                    break
                else:
                    patience_left = self.config.patience

            previous_loss = loss

        acc = self.evaluator.full_test(cur_epoch_idx)

        self.generator.stop()
        
        return acc

    def train_model_epoch(self, epoch_idx, tuning=False):
        """Function to train the model for one epoch."""
        acc_loss = 0

        num_batch = self.model.config.kg_meta.tot_train_triples // self.config.batch_size if not self.config.debug else 10
       
        metrics_names = ['acc_loss', 'loss'] 
        progress_bar = tf.keras.utils.Progbar(num_batch, stateful_metrics=metrics_names)

        for batch_idx in range(num_batch):
            data = list(next(self.generator))
            
            if self.model.training_strategy == TrainingStrategy.PROJECTION_BASED:
                h = tf.convert_to_tensor(data[0], dtype=tf.int32)
                r = tf.convert_to_tensor(data[1], dtype=tf.int32)
                t = tf.convert_to_tensor(data[2], dtype=tf.int32)
                hr_t = data[3]
                rt_h = data[4]
                loss = self.train_step_projection(h, r, t, hr_t, rt_h)
            elif self.model.training_strategy == TrainingStrategy.POINTWISE_BASED:
                h = tf.convert_to_tensor(data[0], dtype=tf.int32)
                r = tf.convert_to_tensor(data[1], dtype=tf.int32)
                t = tf.convert_to_tensor(data[2], dtype=tf.int32)
                y = tf.convert_to_tensor(data[3], dtype=tf.float32)
                loss = self.train_step_pointwise(h, r, t, y)
            else:
                ph = tf.convert_to_tensor(data[0], dtype=tf.int32)
                pr = tf.convert_to_tensor(data[1], dtype=tf.int32)
                pt = tf.convert_to_tensor(data[2], dtype=tf.int32)
                nh = tf.convert_to_tensor(data[3], dtype=tf.int32)
                nr = tf.convert_to_tensor(data[4], dtype=tf.int32)
                nt = tf.convert_to_tensor(data[5], dtype=tf.int32)
                loss = self.train_step_pairwise(ph, pr, pt, nh, nr, nt)

            acc_loss += loss

            if not tuning:
                progress_bar.add(1, values=[('acc_loss', acc_loss), ('loss', loss)])

        self.training_results.append([epoch_idx, acc_loss.numpy()])

        return acc_loss.numpy()
   
    def enter_interactive_mode(self):
        self.build_model()
        self.load_model()

        self.evaluator = Evaluator(self.model)
        print("The training/loading of the model has finished!\nNow enter interactive mode :)")
        print("-----")
        print("Example 1: trainer.infer_tails(1,10,topk=5)")
        self.infer_tails(1,10,topk=5)

        print("-----")
        print("Example 2: trainer.infer_heads(10,20,topk=5)")
        self.infer_heads(10,20,topk=5)

        print("-----")
        print("Example 3: trainer.infer_rels(1,20,topk=5)")
        self.infer_rels(1,20,topk=5)

    def exit_interactive_mode(self):
        print("Thank you for trying out inference interactive script :)")

    def infer_tails(self,h,r,topk=5):
        tails = self.evaluator.test_tail_rank(h,r,topk).numpy()
        print("\n(head, relation)->({},{}) :: Inferred tails->({})\n".format(h,r,",".join([str(i) for i in tails])))
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')
        print("head: %s" % idx2ent[h])
        print("relation: %s" % idx2rel[r])

        for idx, tail in enumerate(tails):
            print("%dth predicted tail: %s" % (idx, idx2ent[tail]))

        return {tail: idx2ent[tail] for tail in tails}

    def infer_heads(self,r,t,topk=5):
        heads = self.evaluator.test_head_rank(r,t,topk).numpy()
        print("\n(relation,tail)->({},{}) :: Inferred heads->({})\n".format(t,r,",".join([str(i) for i in heads])))
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')
        print("tail: %s" % idx2ent[t])
        print("relation: %s" % idx2rel[r])

        for idx, head in enumerate(heads):
            print("%dth predicted head: %s" % (idx, idx2ent[head]))

        return {head: idx2ent[head] for head in heads}

    def infer_rels(self, h, t, topk=5):
        rels = self.evaluator.test_rel_rank(h,t,topk).numpy()
        print("\n(head,tail)->({},{}) :: Inferred rels->({})\n".format(h, t, ",".join([str(i) for i in rels])))
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')
        print("head: %s" % idx2ent[h])
        print("tail: %s" % idx2ent[t])

        for idx, rel in enumerate(rels):
            print("%dth predicted rel: %s" % (idx, idx2rel[rel]))

        return {rel: idx2rel[rel] for rel in rels}
    
    ''' Procedural functions:'''

    def save_model(self):
        """Function to save the model."""
        saved_path = self.config.path_tmp / self.model.model_name
        saved_path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(saved_path / 'model.vec'))

    def load_model(self):
        """Function to load the model."""
        saved_path = self.config.path_tmp / self.model.model_name
        if saved_path.exists():
            self.model.load_weights(str(saved_path / 'model.vec'))

    def display(self):
        """Function to display embedding."""
        options = {"ent_only_plot": True,
                    "rel_only_plot": not self.config.plot_entity_only,
                    "ent_and_rel_plot": not self.config.plot_entity_only}

        if self.config.plot_embedding:
            viz = Visualization(model=self.model, vis_opts = options)

            viz.plot_embedding(resultpath=self.config.figures, algos=self.model.model_name, show_label=False)

        if self.config.plot_training_result:
            viz = Visualization(model=self.model)
            viz.plot_train_result()

        if self.config.plot_testing_result:
            viz = Visualization(model=self.model)
            viz.plot_test_result()
    
    def export_embeddings(self):
        """
            Export embeddings in tsv and pandas pickled format. 
            With tsvs (both label, vector files), you can:
            1) Use those pretained embeddings for your applications.  
            2) Visualize the embeddings in this website to gain insights. (https://projector.tensorflow.org/)

            Pandas dataframes can be read with pd.read_pickle('desired_file.pickle')
        """
        save_path = self.config.path_embeddings / self.model.model_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')


        series_ent = pd.Series(idx2ent)
        series_rel = pd.Series(idx2rel)
        series_ent.to_pickle(save_path / "ent_labels.pickle")
        series_rel.to_pickle(save_path / "rel_labels.pickle")

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
                all_embs = parameter.numpy()
                with open(str(save_path / ("%s.tsv" % stored_name)), 'w') as v_export_file:
                    for idx in all_ids:
                        v_export_file.write("\t".join([str(x) for x in all_embs[idx]]) + "\n")

                df = pd.DataFrame(all_embs)
                df.to_pickle(save_path / ("%s.pickle" % stored_name))

    def save_training_result(self):
        """Function that saves training result"""
        files = os.listdir(str(self.model.config.path_result))
        l = len([f for f in files if self.model.model_name in f if 'Training' in f])
        df = pd.DataFrame(self.training_results, columns=['Epochs', 'Loss'])
        with open(str(self.model.config.path_result / (self.model.model_name + '_Training_results_' + str(l) + '.csv')),
                  'w') as fh:
            df.to_csv(fh)
