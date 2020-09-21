#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for visualizing the results
"""
import os
import seaborn
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from sklearn.manifold import TSNE
from matplotlib import colors as mcolors
from pykg2vec.utils.logger import Logger

seaborn.set_style("darkgrid")

class Visualization:
    """Class to aid in visualizing the results and embddings.

        Args:
            model (object): Model object
            vis_opts (list): Options for visualization.
            sess (object): TensorFlow session object, initialized by the trainer.

        Examples:
            >>> from pykg2vec.utils.visualization import Visualization
            >>> from pykg2vec.utils.trainer import Trainer
            >>> from pykg2vec.models.TransE import TransE
            >>> model = TransE()
            >>> trainer = Trainer(model=model)
            >>> trainer.build_model()
            >>> trainer.train_model()
            >>> viz = Visualization(model=model)
            >>> viz.plot_train_result()
    """
    _logger = Logger().get_logger(__name__)

    def __init__(self, model, config, vis_opts=None):
        if vis_opts:
            self.ent_only_plot = vis_opts["ent_only_plot"]
            self.rel_only_plot = vis_opts["rel_only_plot"]
            self.ent_and_rel_plot = vis_opts["ent_and_rel_plot"]
        else:
            self.ent_only_plot = False
            self.rel_only_plot = False
            self.ent_and_rel_plot = False

        self.model = model
        self.config = config

        self.algo_list = ['ANALOGY', 'Complex', 'ComplexN3', 'ConvE', 'CP', 'DistMult', 'DistMult2', 'HoLE',
                          'KG2E', 'NTN', 'ProjE_pointwise', 'Rescal', 'RotatE', 'SimplE_avg',
                          'SimplE_ignr', 'SLM', 'SME_Bilinear', 'SME_Linear', 'TransD', 'TransE', 'TransH', 'TransM',
                          'TransR', 'TuckER']

        self.h_name = []
        self.r_name = []
        self.t_name = []

        self.h_emb = []
        self.r_emb = []
        self.t_emb = []

        self.h_proj_emb = []
        self.r_proj_emb = []
        self.t_proj_emb = []

        if self.model is not None:
            self.validation_triples_ids = self.config.knowledge_graph.read_cache_data('triplets_valid')
            self.idx2entity = self.config.knowledge_graph.read_cache_data('idx2entity')
            self.idx2relation = self.config.knowledge_graph.read_cache_data('idx2relation')

        self.get_idx_n_emb()

    def get_idx_n_emb(self):
        """Function to get the integer ids and the embedding."""

        idx = np.random.choice(len(self.validation_triples_ids), self.config.disp_triple_num)
        triples = []
        for i, _ in enumerate(idx):
            triples.append(self.validation_triples_ids[idx[i]])

        for t in triples:
            self.h_name.append(self.idx2entity[t.h])
            self.r_name.append(self.idx2relation[t.r])
            self.t_name.append(self.idx2entity[t.t])

            emb_h, emb_r, emb_t = self.model.embed(torch.LongTensor([t.h]).to(self.config.device), torch.LongTensor([t.r]).to(self.config.device), torch.LongTensor([t.t]).to(self.config.device))

            self.h_emb.append(emb_h)
            self.r_emb.append(emb_r)
            self.t_emb.append(emb_t)

            if self.ent_and_rel_plot:
                try:
                    emb_h, emb_r, emb_t = self.model.embed(torch.LongTensor([t.h]).to(self.config.device), torch.LongTensor([t.r]).to(self.config.device), torch.LongTensor([t.t]).to(self.config.device))
                    self.h_proj_emb.append(emb_h)
                    self.r_proj_emb.append(emb_r)
                    self.t_proj_emb.append(emb_t)
                except Exception as e:
                    self._logger.exception(e)

    def plot_embedding(self,
                       resultpath=None,
                       algos=None,
                       show_label=False,
                       disp_num_r_n_e=20):
        """Function to plot the embedding.

            Args:
                resultpath (str): Path where the result will be saved.
                show_label (bool): If True, will display the labels.
                algos (str): Name of the algorithms that generated the embedding.
                disp_num_r_n_e (int): Total number of entities to display for head, tail and relation.

        """
        assert self.model is not None, 'Please provide a model!'

        if self.ent_only_plot:
            x = torch.cat(self.h_emb + self.t_emb, dim=0)
            ent_names = np.concatenate((self.h_name, self.t_name), axis=0)
            self._logger.info("\t Reducing dimension using TSNE to 2!")
            x = TSNE(n_components=2).fit_transform(x.detach().cpu())
            x = np.asarray(x)
            ent_names = np.asarray(ent_names)

            self.draw_embedding(x, ent_names, resultpath, algos + '_entity_plot', show_label)

        if self.rel_only_plot:
            x = torch.cat(self.r_emb, dim=0)
            self._logger.info("\t Reducing dimension using TSNE to 2!")
            x = TSNE(n_components=2).fit_transform(x.detach().cpu())
            self.draw_embedding(x, self.r_name, resultpath, algos + '_rel_plot', show_label)

        if self.ent_and_rel_plot:
            length = len(self.h_proj_emb)
            x = torch.cat(self.h_proj_emb + self.r_proj_emb + self.t_proj_emb, dim=0)
            self._logger.info("\t Reducing dimension using TSNE to 2!")
            x = TSNE(n_components=2).fit_transform(x.detach().cpu())

            h_embs = x[:length, :]
            r_embs = x[length:2 * length, :]
            t_embs = x[2 * length:3 * length, :]

            self.draw_embedding_rel_space(h_embs[:disp_num_r_n_e],
                                          r_embs[:disp_num_r_n_e],
                                          t_embs[:disp_num_r_n_e],
                                          self.h_name[:disp_num_r_n_e],
                                          self.r_name[:disp_num_r_n_e],
                                          self.t_name[:disp_num_r_n_e],
                                          resultpath, algos + '_ent_n_rel_plot', show_label)

    def plot_train_result(self):
        """Function to plot the training result."""
        algo = self.algo_list
        path = self.config.path_result
        result = self.config.path_figures
        data = [self.config.dataset_name]

        files = os.listdir(str(path))
        files_lwcase = [f.lower() for f in files]
        for d in data:
            df = pd.DataFrame()
            for a in algo:
                file_no = len([c for c in files_lwcase if a.lower() in c if 'training' in c])
                if file_no < 1:
                    continue
                file_path = str(path / (a.lower() + '_Training_results_' + str(file_no - 1) + '.csv'))
                if os.path.exists(file_path):
                    with open(str(path / (a.lower() + '_Training_results_' + str(file_no - 1) + '.csv')), 'r') as fh:
                        df_2 = pd.read_csv(fh)
                    if df.empty:
                        df['Epochs'] = df_2['Epochs']
                        df['Loss'] = df_2['Loss']
                        df['Algorithm'] = [a] * len(df_2)
                    else:
                        df_3 = pd.DataFrame()
                        df_3['Epochs'] = df_2['Epochs']
                        df_3['Loss'] = df_2['Loss']
                        df_3['Algorithm'] = [a] * len(df_2)
                        frames = [df, df_3]
                        df = pd.concat(frames)
            plt.figure()
            seaborn.lineplot(x="Epochs", y="Loss", hue="Algorithm", markers=True, dashes=False, data=df)
            files = os.listdir(str(result))
            files_lwcase = [f.lower() for f in files]
            file_no = len([c for c in files_lwcase if d.lower() in c if 'training' in c])
            plt.savefig(str(result / (d + '_training_loss_plot_' + str(file_no) + '.pdf')), bbox_inches='tight', dpi=300)
            # plt.show()

    def plot_test_result(self):
        """Function to plot the testing result."""
        algo = self.algo_list
        path = self.config.path_result
        result = self.config.path_figures
        data = [self.config.dataset_name]
        hits = self.config.hits
        assert path is not None and algo is not None and data is not None, 'Please provide valid path, algorithm and dataset!'
        files = os.listdir(str(path))
        # files_lwcase = [f.lower() for f in files if 'Testing' in f]
        # self._logger.info(files_lwcase)
        for d in data:
            df = pd.DataFrame()
            for a in algo:
                file_algo = [c for c in files if a.lower() in c.lower() if 'testing' in c.lower()]
                if not file_algo:
                    continue
                with open(str(path / file_algo[-1]), 'r') as fh:
                    df_2 = pd.read_csv(fh)

                if df.empty:
                    df['Algorithm'] = [a] * len(df_2)
                    df['Epochs'] = df_2['Epoch']
                    df['Mean Rank'] = df_2['Mean Rank']
                    df['Filt Mean Rank'] = df_2['Filtered Mean Rank']

                    for hit in hits:
                        df['Hits' + str(hit)] = df_2['Hit-%d Ratio'%hit]
                        df['Filt Hits' + str(hit)] = df_2['Filtered Hit-%d Ratio'%hit]

                else:
                    df_3 = pd.DataFrame()
                    df_3['Algorithm'] = [a] * len(df_2)
                    df_3['Epochs'] = df_2['Epoch']
                    df_3['Mean Rank'] = df_2['Mean Rank']
                    df_3['Filt Mean Rank'] = df_2['Filtered Mean Rank']

                    for hit in hits:
                        df_3['Hits' + str(hit)] = df_2['Hit-%d Ratio'%hit]
                        df_3['Filt Hits' + str(hit)] = df_2['Filtered Hit-%d Ratio'%hit]

                    frames = [df, df_3]
                    df = pd.concat(frames)

            files = os.listdir(str(result))
            df_4 = df.loc[df['Epochs'] == max(df['Epochs'])]
            df_4 = df_4.loc[:, df_4.columns != 'Epochs']

            file_no = len(
                [c for c in files if d.lower() in c.lower() if 'testing' in c.lower() if 'latex' in c.lower()])
            with open(str(result / (d + '_testing_latex_table_' + str(file_no + 1) + '.txt')), 'w') as fh:
                fh.write(df_4.to_latex(index=False))

            file_no = len(
                [c for c in files if d.lower() in c.lower() if 'testing' in c.lower() if 'table' in c.lower() if
                 'csv' in c.lower()])
            with open(str(result / (d + '_testing_table_' + str(file_no + 1) + '.csv')), 'w') as fh:
                df_4.to_csv(fh, index=False)

            df_5 = pd.DataFrame(columns=['Metrics', 'Algorithm', 'Score'])
            metrics = [f for f in df_4.columns if f != 'Algorithm']
            for i in range(len(df_4)):
                if df_5.empty:
                    df_5['Algorithm'] = [df_4.iloc[i]['Algorithm']] * len(metrics)
                    df_5['Metrics'] = metrics
                    df_5['Score'] = df_4.iloc[i][metrics].values
                else:
                    df_t = pd.DataFrame()
                    df_t['Algorithm'] = [df_4.iloc[i]['Algorithm']] * len(metrics)
                    df_t['Metrics'] = metrics
                    df_t['Score'] = df_4.iloc[i][metrics].values
                    frame = [df_5, df_t]
                    df_5 = pd.concat(frame)

            df_6 = df_5[~df_5['Metrics'].str.contains('Hits')]
            plt.figure()
            flatui = ["#d46a7e", "#d5b60a", "#9b59b6", "#3498db", "#95a5a6", "#34495e", "#2ecc71", "#e74c3c"]
            g = seaborn.barplot(x="Metrics", y='Score', hue="Algorithm", palette=flatui, data=df_6)
            g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=6)
            g.tick_params(labelsize=6)
            # ax = seaborn.lineplot(x="Metrics", y='Score', hue="Algorithm",
            #                       markers=True, dashes=False, data=df_5)

            files_lwcase = [f.lower() for f in files]
            file_no = len([c for c in files_lwcase if d.lower() in c if 'testing' in c if 'rank_plot' in c])
            plt.savefig(str(result / (d + '_testing_rank_plot_' + str(file_no + 1) + '.pdf')), bbox_inches='tight',
                        dpi=300)
            # plt.show()

            df_6 = df_5[df_5['Metrics'].str.contains('Hits')]
            plt.figure()
            flatui = ["#3498db", "#95a5a6", "#34495e", "#2ecc71", "#e74c3c", "#d46a7e", "#d5b60a", "#9b59b6"]
            g = seaborn.barplot(x="Metrics", y='Score', hue="Algorithm", palette=flatui, data=df_6)
            g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=6)
            g.tick_params(labelsize=6)

            files_lwcase = [f.lower() for f in files]
            file_no = len([c for c in files_lwcase if d.lower() in c if 'testing' in c if 'hits_plot' in c])
            plt.savefig(str(result / (d + '_testing_hits_plot_' + str(file_no + 1) + '.pdf')), bbox_inches='tight',
                        dpi=300)
            # plt.show()

    @staticmethod
    def draw_embedding(embs, names, resultpath, algos, show_label):
        """Function to draw the embedding.

            Args:
                embs (matrix): Two dimesnional embeddings.
                names (list):List of string name.
                resultpath (str):Path where the result will be save.
                algos (str): Name of the algorithms which generated the algorithm.
                show_label (bool): If True, prints the string names of the entities and relations.

        """
        pos = {}
        node_color_mp = {}
        unique_ent = set(names)
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

        tot_col = len(colors)
        j = 0
        for i, e in enumerate(unique_ent):
            node_color_mp[e] = colors[j]
            j += 1
            if j >= tot_col:
                j = 0

        G = nx.Graph()
        hm_ent = {}
        for i, ent in enumerate(names):
            hm_ent[i] = ent
            G.add_node(i)
            pos[i] = embs[i]

        colors = []
        for n in list(G.nodes):
            colors.append(node_color_mp[hm_ent[n]])

        plt.figure()
        nodes_draw = nx.draw_networkx_nodes(G,
                                            pos,
                                            node_color=colors,
                                            node_size=50)
        nodes_draw.set_edgecolor('k')
        if show_label:
            nx.draw_networkx_labels(G, pos, font_size=8)

        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        files = os.listdir(resultpath)
        file_no = len(
            [c for c in files if algos + '_embedding_plot' in c])
        filename = algos + '_embedding_plot_' + str(file_no) + '.png'
        plt.savefig(str(resultpath / filename), bbox_inches='tight', dpi=300)
        # plt.show()

    @staticmethod
    def draw_embedding_rel_space(h_emb,
                                 r_emb,
                                 t_emb,
                                 h_name,
                                 r_name,
                                 t_name,
                                 resultpath,
                                 algos,
                                 show_label):
        """Function to draw the embedding in relation space.

            Args:
                h_emb (matrix): Two dimesnional embeddings of head.
                r_emb (matrix): Two dimesnional embeddings of relation.
                t_emb (matrix): Two dimesnional embeddings of tail.
                h_name (list):List of string name of the head.
                r_name (list):List of string name of the relation.
                t_name (list):List of string name of the tail.
                resultpath (str):Path where the result will be save.
                algos (str): Name of the algorithms which generated the algorithm.
                show_label (bool): If True, prints the string names of the entities and relations.

        """
        pos = {}
        node_color_mp_ent = {}
        node_color_mp_rel = {}
        unique_ent = set(h_name) | set(t_name)
        unique_rel = set(r_name)
        colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

        tot_col = len(colors)
        j = 0
        for i, e in enumerate(unique_ent):
            node_color_mp_ent[e] = colors[j]
            j += 1
            if j >= tot_col:
                j = 0

        tot_col = len(colors)
        j = 0
        for i, r in enumerate(unique_rel):
            node_color_mp_rel[r] = colors[j]
            j += 1
            if j >= tot_col:
                j = 0

        G = nx.DiGraph()
        idx = 0
        head_colors = []
        rel_colors = []
        tail_colors = []
        head_nodes = []
        tail_nodes = []
        rel_nodes = []

        for i, _ in enumerate(h_name):
            G.add_edge(idx, idx + 1)
            G.add_edge(idx + 1, idx + 2)

            head_nodes.append(idx)
            rel_nodes.append(idx + 1)
            tail_nodes.append(idx + 2)

            head_colors.append(node_color_mp_ent[h_name[i]])
            rel_colors.append(node_color_mp_rel[r_name[i]])
            tail_colors.append(node_color_mp_ent[t_name[i]])

            pos[idx] = h_emb[i]
            pos[idx + 1] = r_emb[i]
            pos[idx + 2] = t_emb[i]
            idx += 3

        plt.figure()
        nodes_draw = nx.draw_networkx_nodes(G,
                                            pos,
                                            nodelist=head_nodes,
                                            node_color=head_colors,
                                            node_shape='o',
                                            node_size=50)
        nodes_draw.set_edgecolor('k')

        nodes_draw = nx.draw_networkx_nodes(G,
                                            pos,
                                            nodelist=rel_nodes,
                                            node_color=rel_colors,
                                            node_size=50,
                                            node_shape='D')
        nodes_draw.set_edgecolor('k')

        nodes_draw = nx.draw_networkx_nodes(G,
                                            pos,
                                            nodelist=tail_nodes,
                                            node_color=tail_colors,
                                            node_shape='*',
                                            node_size=50)
        nodes_draw.set_edgecolor('k')

        if show_label:
            nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, arrows=True, width=0.5, alpha=0.5)

        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        files = os.listdir(resultpath)
        file_no = len(
            [c for c in files if algos + '_embedding_plot' in c])
        plt.savefig(str(resultpath / (algos + '_embedding_plot_' + str(file_no) + '.png')), bbox_inches='tight',
                    dpi=300)
        # plt.show()
