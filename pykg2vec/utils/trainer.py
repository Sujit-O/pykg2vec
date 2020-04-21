#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for training process.
"""
import timeit, os
import tensorflow as tf
import pandas as pd

from enum import Enum
from pykg2vec.core.KGMeta import TrainerMeta
from pykg2vec.utils.evaluator import Evaluator
from pykg2vec.utils.visualization import Visualization
from pykg2vec.utils.generator import Generator, TrainingStrategy
from pykg2vec.utils.logger import Logger
from pykg2vec.utils.kgcontroller import KnowledgeGraph

import warnings
warnings.filterwarnings('ignore')

tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU') 
try:
    for gpu in physical_devices: 
        tf.config.experimental.set_memory_growth(gpu, True) 
except: 
    # Invalid device or cannot modify virtual devices once initialized. 
    pass


class Monitor(Enum):
    MEAN_RANK = "mr"
    FILTERED_MEAN_RANK = "fmr"
    MEAN_RECIPROCAL_RANK = "mrr"
    FILTERED_MEAN_RECIPROCAL_RANK = "fmrr"


class EarlyStopper:
    
    _logger = Logger().get_logger(__name__)

    def __init__(self, patience, monitor):

        self.monitor = monitor
        self.patience = patience
        
        # controlling variables.
        self.previous_metrics = None
        self.patience_left = patience

    def should_stop(self, curr_metrics):
        should_stop = False
        value, name = self.monitor.value, self.monitor.name

        if self.previous_metrics is not None:
            if self.monitor == Monitor.MEAN_RANK or self.monitor == Monitor.FILTERED_MEAN_RANK:
                is_worse = self.previous_metrics[value] < curr_metrics[value]
            else:
                is_worse = self.previous_metrics[value] > curr_metrics[value]

            if self.patience_left > 0 and is_worse:
                self.patience_left -= 1
                self._logger.info(
                    '%s more chances before the trainer stops the training. (prev_%s, curr_%s): (%.4f, %.4f)' %
                    (self.patience_left, name, name, self.previous_metrics[value], curr_metrics[value]))
            
            elif self.patience_left == 0 and is_worse:
                self._logger.info('Stop the training.')
                should_stop = True
            
            else:
                self._logger.info('Reset the patience count to %d' % (self.patience))
                self.patience_left = self.patience
                
        self.previous_metrics = curr_metrics

        return should_stop


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
    _logger = Logger().get_logger(__name__)

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
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.config.learning_rate, initial_accumulator_value=0.0, epsilon=1e-08)
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

        self.early_stopper = EarlyStopper(self.config.patience, Monitor.FILTERED_MEAN_RANK)

    ''' Training related functions:'''
    @tf.function
    def train_step_pairwise(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        with tf.GradientTape() as tape:
            pos_preds = self.model.forward(pos_h, pos_r, pos_t)
            neg_preds = self.model.forward(neg_h, neg_r, neg_t)
            
            if self.config.sampling == 'adversarial_negative_sampling':
                # RotatE: Adversarial Nnegative Sampling and alpha is the temperature.
                pos_preds = -pos_preds
                neg_preds = -neg_preds
                pos_preds = tf.math.log_sigmoid(pos_preds)
                neg_preds = tf.reshape(neg_preds, [-1, self.config.neg_rate])
                softmax = tf.stop_gradient(tf.nn.softmax(neg_preds*self.config.alpha, axis=1))
                neg_preds = tf.reduce_sum(softmax * (tf.math.log_sigmoid(-neg_preds)), axis=-1)
                loss = -tf.reduce_mean(neg_preds) - tf.reduce_mean(pos_preds)
            else:
                # others that use margin-based & pairwise loss function. (unif or bern)
                loss = tf.reduce_sum(tf.maximum(pos_preds + self.config.margin - neg_preds, 0))
            
            if hasattr(self.model, 'get_reg'):
                # now only NTN uses regularizer, 
                # other pairwise based KGE methods use normalization to regularize parameters.
                loss += self.model.get_reg()

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def train_step_projection(self, h, r, t, hr_t, tr_h):
        with tf.GradientTape() as tape:
            hr_t = tf.cast(tf.sparse.to_dense(tf.sparse.reorder(hr_t)), dtype=tf.float32)
            tr_h = tf.cast(tf.sparse.to_dense(tf.sparse.reorder(tr_h)), dtype=tf.float32)
           
            if self.model.model_name.lower() == "conve" or self.model.model_name.lower() == "tucker":   
                if hasattr(self.config, 'label_smoothing'):
                    hr_t = hr_t * (1.0 - self.config.label_smoothing) + 1.0 / self.config.kg_meta.tot_entity
                    tr_h = tr_h * (1.0 - self.config.label_smoothing) + 1.0 / self.config.kg_meta.tot_entity

                pred_tails = self.model.forward(h, r, direction="tail") # (h, r) -> hr_t forward
                pred_heads = self.model.forward(t, r, direction="head") # (t, r) -> tr_h backward

                loss_tails = tf.reduce_mean(tf.keras.backend.binary_crossentropy(hr_t, pred_tails))
                loss_heads = tf.reduce_mean(tf.keras.backend.binary_crossentropy(tr_h, pred_heads))

                loss = loss_tails + loss_heads
            
            else:
                loss_tails = self.model.forward(h, r, hr_t, direction="tail") # (h, r) -> hr_t forward
                loss_heads = self.model.forward(t, r, tr_h, direction="head") # (t, r) -> tr_h backward

                loss = loss_tails + loss_heads

                if hasattr(self.model, 'get_reg'):
                    # now only complex distmult uses regularizer in algorithms, 
                    loss += self.model.get_reg()


        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def train_step_pointwise(self, h, r, t, y):
        with tf.GradientTape() as tape:
            preds = self.model.forward(h, r, t)

            loss = tf.reduce_mean(tf.nn.softplus(y*preds)) 

            if hasattr(self.model, 'get_reg'): # for complex & complex-N3 & DistMult & CP & ANALOGY
                loss += self.model.get_reg(h, r, t)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train_model(self, monitor=Monitor.FILTERED_MEAN_RANK):
        """Function to train the model."""
        self.generator = Generator(self.model)
        self.evaluator = Evaluator(self.model)

        if self.config.loadFromData:
            self.load_model()
        
        for cur_epoch_idx in range(self.config.epochs):
            self._logger.info("Epoch[%d/%d]" % (cur_epoch_idx, self.config.epochs))
            
            self.train_model_epoch(cur_epoch_idx)

            if cur_epoch_idx % self.config.test_step == 0:
                metrics = self.evaluator.mini_test(cur_epoch_idx)
                              
                if self.early_stopper.should_stop(metrics):
                    ### Early Stop Mechanism
                    ### start to check if the metric is still improving after each mini-test. 
                    ### Example, if test_step == 5, the trainer will check metrics every 5 epoch.
                    break

        self.evaluator.full_test(cur_epoch_idx)
        self.evaluator.metric_calculator.save_test_summary(self.model.model_name)

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
        current_loss = float("inf")

        self.generator = Generator(self.model)
        self.evaluator = Evaluator(self.model, tuning=True)
       
        for cur_epoch_idx in range(self.config.epochs):
            current_loss = self.train_model_epoch(cur_epoch_idx, tuning=True)

        self.evaluator.full_test(cur_epoch_idx)

        self.generator.stop()
        
        return current_loss

    def train_model_epoch(self, epoch_idx, tuning=False):
        """Function to train the model for one epoch."""
        acc_loss = 0

        num_batch = self.model.config.kg_meta.tot_train_triples // self.config.batch_size if not self.config.debug else 10
       
        metrics_names = ['acc_loss', 'loss'] 
        progress_bar = tf.keras.utils.Progbar(num_batch, stateful_metrics=metrics_names)
        
        self.generator.start_one_epoch(num_batch)

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
        self._logger.info("""The training/loading of the model has finished!
                                    Now enter interactive mode :)
                                    -----
                                    Example 1: trainer.infer_tails(1,10,topk=5)""")
        self.infer_tails(1,10,topk=5)

        self._logger.info("""-----
                                    Example 2: trainer.infer_heads(10,20,topk=5)""")
        self.infer_heads(10,20,topk=5)

        self._logger.info("""-----
                                    Example 3: trainer.infer_rels(1,20,topk=5)""")
        self.infer_rels(1,20,topk=5)

    def exit_interactive_mode(self):
        self._logger.info("Thank you for trying out inference interactive script :)")

    def infer_tails(self,h,r,topk=5):
        tails = self.evaluator.test_tail_rank(h,r,topk).numpy()
        logs = []
        logs.append("")
        logs.append("(head, relation)->({},{}) :: Inferred tails->({})".format(h,r,",".join([str(i) for i in tails])))
        logs.append("")
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')
        logs.append("head: %s" % idx2ent[h])
        logs.append("relation: %s" % idx2rel[r])

        for idx, tail in enumerate(tails):
            logs.append("%dth predicted tail: %s" % (idx, idx2ent[tail]))

        self._logger.info("\n".join(logs))
        return {tail: idx2ent[tail] for tail in tails}

    def infer_heads(self,r,t,topk=5):
        heads = self.evaluator.test_head_rank(r,t,topk).numpy()
        logs = []
        logs.append("")
        logs.append("(relation,tail)->({},{}) :: Inferred heads->({})".format(t,r,",".join([str(i) for i in heads])))
        logs.append("")
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')
        logs.append("tail: %s" % idx2ent[t])
        logs.append("relation: %s" % idx2rel[r])

        for idx, head in enumerate(heads):
            logs.append("%dth predicted head: %s" % (idx, idx2ent[head]))

        self._logger.info("\n".join(logs))
        return {head: idx2ent[head] for head in heads}

    def infer_rels(self, h, t, topk=5):
        if self.model.model_name.lower() in ["proje_pointwise", "conve", "tucker"]:
            self._logger.info("%s model doesn't support relation inference in nature.")
            return

        rels = self.evaluator.test_rel_rank(h,t,topk).numpy()
        logs = []
        logs.append("")
        logs.append("(head,tail)->({},{}) :: Inferred rels->({})".format(h, t, ",".join([str(i) for i in rels])))
        logs.append("")
        idx2ent = self.model.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.model.config.knowledge_graph.read_cache_data('idx2relation')
        logs.append("head: %s" % idx2ent[h])
        logs.append("tail: %s" % idx2ent[t])

        for idx, rel in enumerate(rels):
            logs.append("%dth predicted rel: %s" % (idx, idx2rel[rel]))

        self._logger.info("\n".join(logs))
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