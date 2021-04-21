#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings
import torch
import torch.optim as optim
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from pykg2vec.utils.evaluator import Evaluator
from pykg2vec.utils.visualization import Visualization
from pykg2vec.utils.riemannian_optimizer import RiemannianOptimizer
from pykg2vec.data.generator import Generator
from pykg2vec.utils.logger import Logger
from pykg2vec.common import Importer, Monitor, TrainingStrategy
warnings.filterwarnings('ignore')


class EarlyStopper:

    """ Class used by trainer for handling the early stopping mechanism during the training of KGE algorithms.

        Args:
            patience (int): Number of epochs to wait before early stopping the training on no improvement.
            No early stopping if it is a negative number (default: {-1}).
            monitor (Monitor): the type of metric that earlystopper will monitor.

    """

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


class Trainer:
    """ Class for handling the training of the algorithms.

        Args:
            model (object): KGE model object

        Examples:
            >>> from pykg2vec.utils.trainer import Trainer
            >>> from pykg2vec.models.pairwise import TransE
            >>> trainer = Trainer(TransE())
            >>> trainer.build_model()
            >>> trainer.train_model()
    """
    TRAINED_MODEL_FILE_NAME = "model.vec.pt"
    TRAINED_MODEL_CONFIG_NAME = "config.npy"
    _logger = Logger().get_logger(__name__)

    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.best_metric = None
        self.monitor = None

        self.training_results = []

        self.evaluator = None
        self.generator = None
        self.optimizer = None
        self.early_stopper = None

    def build_model(self, monitor=Monitor.FILTERED_MEAN_RANK):
        """function to build the model"""
        if self.config.load_from_data is not None:
            self.load_model(self.config.load_from_data)

        self.evaluator = Evaluator(self.model, self.config)

        self.model.to(self.config.device)

        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
        elif self.config.optimizer == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
        elif self.config.optimizer == "rms":
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
        elif self.config.optimizer == "riemannian":
            param_names = [name for name, param in self.model.named_parameters()]
            self.optimizer = RiemannianOptimizer(
                self.model.parameters(),
                lr=self.config.learning_rate,
                param_names=param_names
            )
        else:
            raise NotImplementedError("No support for %s optimizer" % self.config.optimizer)

        self.config.summary()

        self.early_stopper = EarlyStopper(self.config.patience, monitor)

    # Training related functions:
    def train_step_pairwise(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_preds = self.model(pos_h, pos_r, pos_t)
        neg_preds = self.model(neg_h, neg_r, neg_t)

        if self.model.model_name.lower() == "rotate":
            loss = self.model.loss(pos_preds, neg_preds, self.config.neg_rate, self.config.alpha)
        else:
            loss = self.model.loss(pos_preds, neg_preds, self.config.margin)
        loss += self.model.get_reg(None, None, None)

        return loss

    def train_step_projection(self, h, r, t, hr_t, tr_h):
        if self.model.model_name.lower() in ["conve", "tucker", "interacte", "hyper", "acre"]:
            pred_tails = self.model(h, r, direction="tail")  # (h, r) -> hr_t forward
            pred_heads = self.model(t, r, direction="head")  # (t, r) -> tr_h backward

            if hasattr(self.config, 'label_smoothing'):
                loss = self.model.loss(pred_heads, pred_tails, tr_h, hr_t, self.config.label_smoothing, self.config.tot_entity)
            else:
                loss = self.model.loss(pred_heads, pred_tails, tr_h, hr_t, None, None)
        else:
            pred_tails = self.model(h, r, hr_t, direction="tail")  # (h, r) -> hr_t forward
            pred_heads = self.model(t, r, tr_h, direction="head")  # (t, r) -> tr_h backward
            loss = self.model.loss(pred_heads, pred_tails)
        loss += self.model.get_reg(h, r, t)

        return loss

    def train_step_pointwise(self, h, r, t, target):
        preds = self.model(h, r, t)
        loss = self.model.loss(preds, target.type(preds.type()))
        loss += self.model.get_reg(h, r, t)
        return loss

    def train_model(self):

        # for key, value in self.config.__dict__.items():
        #     print(key," ",value)
        #print(self.config.__dict__[""])
        # pdb.set_trace()

        """Function to train the model."""
        self.generator = Generator(self.model, self.config)
        self.monitor = Monitor.FILTERED_MEAN_RANK
        for cur_epoch_idx in range(self.config.epochs):
            self._logger.info("Epoch[%d/%d]" % (cur_epoch_idx, self.config.epochs))

            self.train_model_epoch(cur_epoch_idx)

            if cur_epoch_idx % self.config.test_step == 0:
                self.model.eval()
                with torch.no_grad():
                    metrics = self.evaluator.mini_test(cur_epoch_idx)

                    if self.early_stopper.should_stop(metrics):
                        ### Early Stop Mechanism
                        ### start to check if the metric is still improving after each mini-test.
                        ### Example, if test_step == 5, the trainer will check metrics every 5 epoch.
                        break

                    # store the best model weights.
                    if self.config.save_model:
                        if self.best_metric is None:
                            self.best_metric = metrics
                            self.save_model()
                        else:
                            if self.monitor == Monitor.MEAN_RANK or self.monitor == Monitor.FILTERED_MEAN_RANK:
                                is_better = self.best_metric[self.monitor.value] > metrics[self.monitor.value]
                            else:
                                is_better = self.best_metric[self.monitor.value] < metrics[self.monitor.value]
                            if is_better:
                                self.save_model()
                                self.best_metric = metrics

        self.model.eval()
        with torch.no_grad():
            self.evaluator.full_test(cur_epoch_idx)

        self.evaluator.metric_calculator.save_test_summary(self.model.model_name)

        self.generator.stop()
        self.save_training_result()

        # if self.config.save_model:
        #     self.save_model()

        if self.config.disp_result:
            self.display()

        self.export_embeddings()

        return cur_epoch_idx # the runned epoches.

    def tune_model(self):
        """Function to tune the model."""
        current_loss = float("inf")

        self.generator = Generator(self.model, self.config)
        self.evaluator = Evaluator(self.model, self.config, tuning=True)

        for cur_epoch_idx in range(self.config.epochs):
            current_loss = self.train_model_epoch(cur_epoch_idx, tuning=True)

        self.model.eval()
        with torch.no_grad():
            self.evaluator.full_test(cur_epoch_idx)

        self.generator.stop()

        return current_loss

    def train_model_epoch(self, epoch_idx, tuning=False):
        """Function to train the model for one epoch."""
        acc_loss = 0

        num_batch = self.config.tot_train_triples // self.config.batch_size if not self.config.debug else 10

        self.generator.start_one_epoch(num_batch)

        progress_bar = tqdm(range(num_batch))

        for _ in progress_bar:
            data = list(next(self.generator))
            self.model.train()
            self.optimizer.zero_grad()

            if self.model.training_strategy == TrainingStrategy.PROJECTION_BASED:
                h = torch.LongTensor(data[0]).to(self.config.device)
                r = torch.LongTensor(data[1]).to(self.config.device)
                t = torch.LongTensor(data[2]).to(self.config.device)
                hr_t = data[3].to(self.config.device)
                tr_h = data[4].to(self.config.device)
                loss = self.train_step_projection(h, r, t, hr_t, tr_h)
            elif self.model.training_strategy == TrainingStrategy.POINTWISE_BASED:
                h = torch.LongTensor(data[0]).to(self.config.device)
                r = torch.LongTensor(data[1]).to(self.config.device)
                t = torch.LongTensor(data[2]).to(self.config.device)
                y = torch.LongTensor(data[3]).to(self.config.device)
                loss = self.train_step_pointwise(h, r, t, y)
            elif self.model.training_strategy == TrainingStrategy.PAIRWISE_BASED:
                pos_h = torch.LongTensor(data[0]).to(self.config.device)
                pos_r = torch.LongTensor(data[1]).to(self.config.device)
                pos_t = torch.LongTensor(data[2]).to(self.config.device)
                neg_h = torch.LongTensor(data[3]).to(self.config.device)
                neg_r = torch.LongTensor(data[4]).to(self.config.device)
                neg_t = torch.LongTensor(data[5]).to(self.config.device)
                loss = self.train_step_pairwise(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            else:
                raise NotImplementedError("Unknown training strategy: %s" % self.model.training_strategy)

            loss.backward()
            self.optimizer.step()
            acc_loss += loss.item()

            if not tuning:
                progress_bar.set_description('acc_loss: %f, cur_loss: %f'% (acc_loss, loss))

        self.training_results.append([epoch_idx, acc_loss])

        return acc_loss

    def enter_interactive_mode(self):
        self.build_model()
        self.load_model()

        self._logger.info("""The training/loading of the model has finished!
                                    Now enter interactive mode :)
                                    -----
                                    Example 1: trainer.infer_tails(1,10,topk=5)""")
        self.infer_tails(1, 10, topk=5)

        self._logger.info("""-----
                                    Example 2: trainer.infer_heads(10,20,topk=5)""")
        self.infer_heads(10, 20, topk=5)

        self._logger.info("""-----
                                    Example 3: trainer.infer_rels(1,20,topk=5)""")
        self.infer_rels(1, 20, topk=5)

    def exit_interactive_mode(self):
        self._logger.info("Thank you for trying out inference interactive script :)")

    def infer_tails(self, h, r, topk=5):
        tails = self.evaluator.test_tail_rank(h, r, topk).detach().cpu().numpy()
        idx2ent = self.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.config.knowledge_graph.read_cache_data('idx2relation')
        logs = [
            "",
            "(head, relation)->({},{}) :: Inferred tails->({})".format(h, r, ",".join([str(i) for i in tails])),
            "",
            "head: %s" % idx2ent[h],
            "relation: %s" % idx2rel[r],
        ]

        for idx, tail in enumerate(tails):
            logs.append("%dth predicted tail: %s" % (idx, idx2ent[tail]))

        self._logger.info("\n".join(logs))
        return {tail: idx2ent[tail] for tail in tails}

    def infer_heads(self, r, t, topk=5):
        heads = self.evaluator.test_head_rank(r, t, topk).detach().cpu().numpy()
        idx2ent = self.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.config.knowledge_graph.read_cache_data('idx2relation')
        logs = [
            "",
            "(relation,tail)->({},{}) :: Inferred heads->({})".format(t, r, ",".join([str(i) for i in heads])),
            "",
            "tail: %s" % idx2ent[t],
            "relation: %s" % idx2rel[r],
        ]

        for idx, head in enumerate(heads):
            logs.append("%dth predicted head: %s" % (idx, idx2ent[head]))

        self._logger.info("\n".join(logs))
        return {head: idx2ent[head] for head in heads}

    def infer_rels(self, h, t, topk=5):
        if self.model.model_name.lower() in ["proje_pointwise", "conve", "tucker"]:
            self._logger.info("%s model doesn't support relation inference in nature.")
            return {}

        rels = self.evaluator.test_rel_rank(h, t, topk).detach().cpu().numpy()
        idx2ent = self.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.config.knowledge_graph.read_cache_data('idx2relation')
        logs = [
            "",
            "(head,tail)->({},{}) :: Inferred rels->({})".format(h, t, ",".join([str(i) for i in rels])),
            "",
            "head: %s" % idx2ent[h],
            "tail: %s" % idx2ent[t],
        ]

        for idx, rel in enumerate(rels):
            logs.append("%dth predicted rel: %s" % (idx, idx2rel[rel]))

        self._logger.info("\n".join(logs))
        return {rel: idx2rel[rel] for rel in rels}

    # ''' Procedural functions:'''
    def save_model(self):
        """Function to save the model."""
        saved_path = self.config.path_tmp / self.model.model_name
        saved_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(saved_path / self.TRAINED_MODEL_FILE_NAME))

        # Save hyper-parameters into a yaml file with the model
        save_path_config = saved_path / self.TRAINED_MODEL_CONFIG_NAME
        np.save(save_path_config, self.config)

    def load_model(self, model_path=None):
        """Function to load the model."""
        if model_path is None:
            model_path_file = self.config.path_tmp / self.model.model_name / self.TRAINED_MODEL_FILE_NAME
            model_path_config = self.config.path_tmp / self.model.model_name / self.TRAINED_MODEL_CONFIG_NAME
        else:
            model_path = Path(model_path)
            model_path_file = model_path / self.TRAINED_MODEL_FILE_NAME
            model_path_config = model_path / self.TRAINED_MODEL_CONFIG_NAME

        if model_path_file.exists() and model_path_config.exists():
            config_temp = np.load(model_path_config, allow_pickle=True).item()
            config_temp.__dict__['load_from_data'] = self.config.__dict__['load_from_data']
            self.config = config_temp

            _, model_def = Importer().import_model_config(self.config.model_name.lower())
            self.model = model_def(**self.config.__dict__)
            self.model.load_state_dict(torch.load(str(model_path_file)))
            self.model.eval()
        else:
            raise ValueError("Cannot load model from %s" % model_path_file)

    def display(self):
        """Function to display embedding."""
        options = {"ent_only_plot": True,
                   "rel_only_plot": not self.config.plot_entity_only,
                   "ent_and_rel_plot": not self.config.plot_entity_only}

        if self.config.plot_embedding:
            viz = Visualization(self.model, self.config, vis_opts=options)
            viz.plot_embedding(resultpath=self.config.path_figures, algos=self.model.model_name, show_label=False)

        if self.config.plot_training_result:
            viz = Visualization(self.model, self.config)
            viz.plot_train_result()

        if self.config.plot_testing_result:
            viz = Visualization(self.model, self.config)
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

        idx2ent = self.config.knowledge_graph.read_cache_data('idx2entity')
        idx2rel = self.config.knowledge_graph.read_cache_data('idx2relation')

        with open(str(save_path / "ent_labels.tsv"), 'w') as l_export_file:
            for label in idx2ent.values():
                l_export_file.write(label + "\n")

        with open(str(save_path / "rel_labels.tsv"), 'w') as l_export_file:
            for label in idx2rel.values():
                l_export_file.write(label + "\n")

        for named_embedding in self.model.parameter_list:
            all_ids = list(range(0, int(named_embedding.weight.shape[0])))

            stored_name = named_embedding.name

            if len(named_embedding.weight.shape) == 2:
                all_embs = named_embedding.weight.detach().detach().cpu().numpy()
                with open(str(save_path / ("%s.tsv" % stored_name)), 'w') as v_export_file:
                    for idx in all_ids:
                        v_export_file.write("\t".join([str(x) for x in all_embs[idx]]) + "\n")

    def save_training_result(self):
        """Function that saves training result"""
        files = os.listdir(str(self.config.path_result))
        l = len([f for f in files if self.model.model_name in f if 'Training' in f])
        df = pd.DataFrame(self.training_results, columns=['Epochs', 'Loss'])
        with open(str(self.config.path_result / (self.model.model_name + '_Training_results_' + str(l) + '.csv')),
                  'w') as fh:
            df.to_csv(fh)
