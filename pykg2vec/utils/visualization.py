#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is for visualizing the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

disp_avlbl = True
from os import environ

if 'DISPLAY' not in environ:
    disp_avlbl = False
    import matplotlib

    matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import seaborn
import pandas as pd

seaborn.set_style("darkgrid")


class Visualization(object):
    def __init__(self, triples=None, idx2entity=None, idx2relation=None):
        self.triples = triples
        if triples:
            self.tot_triples = len(triples)
        else:
            self.tot_triples = None
        self.h_emb = []
        self.t_emb = []
        self.r_emb = []
        self.h_name = []
        self.r_name = []
        self.t_name = []
        self.idx2entity = idx2entity
        self.idx2relation = idx2relation
        self.G = nx.DiGraph()
        self.h_embs = None
        self.r_embs = None
        self.t_embs = None

    def get_idx_n_emb(self, model=None, sess=None):

        for t in self.triples:
            self.h_name.append(self.idx2entity[t.h])
            self.r_name.append(self.idx2relation[t.r])
            self.t_name.append(self.idx2entity[t.t])
            emb_h, emb_r, emb_t = model.predict_embed(t.h, t.r, t.t, sess)

            # print("\nembs", emb_h)
            self.h_emb.append(emb_h)
            self.r_emb.append(emb_r)
            self.t_emb.append(emb_t)

    def reduce_dim(self):
        print("\t reducing dimension to 2 using TSNE!")
        self.h_emb = np.array(self.h_emb)
        self.r_emb = np.array(self.r_emb)
        self.t_emb = np.array(self.t_emb)

        print("dimension self.h_emb", np.shape(self.h_emb))
        print("dimension self.r_emb", np.shape(self.r_emb))
        print("dimension self.t_emb", np.shape(self.t_emb))

        length = len(self.h_emb)
        x = np.concatenate((self.h_emb, self.r_emb, self.t_emb), axis=0)
        x_reduced = TSNE(n_components=2).fit_transform(x)

        self.h_embs = x_reduced[:length, :]
        self.r_embs = x_reduced[length:2 * length, :]
        self.t_embs = x_reduced[2 * length:3 * length, :]

        print("dimension self.h_emb", np.shape(self.h_embs))
        print("dimension self.r_emb", np.shape(self.r_embs))
        print("dimension self.t_emb", np.shape(self.t_embs))

        # print(self.h_embs)
        # print(self.r_embs)
        # print(self.t_embs)

    def plot_train_result(self, path=None, result=None, algo=None, data=None):
        if not os.path.exists(result):
            os.mkdir(result)
        if path is None or algo is None or data is None:
            raise NotImplementedError('Please provide valid path, algorithm and dataset!')
        files = os.listdir(path)
        files_lwcase = [f.lower() for f in files]
        for d in data:
            df = pd.DataFrame()
            # print(algo)
            for a in algo:
                file_no = len([c for c in files_lwcase if a.lower() in c if 'training' in c])
                # print(a,file_no)
                if file_no < 1:
                    continue
                with open(path + '/' + a + '_Training_results_' + str(file_no - 1) + '.csv', 'r') as fh:
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
                # print(df)
            plt.figure()
            ax = seaborn.lineplot(x="Epochs", y="Loss", hue="Algorithm",
                                  markers=True, dashes=False, data=df)
            files = os.listdir(result)
            files_lwcase = [f.lower() for f in files]
            file_no = len([c for c in files_lwcase if d.lower() in c if 'training' in c])
            plt.savefig(result + '/' + d + '_training_loss_plot_' + str(file_no) + '.pdf', bbox_inches='tight', dpi=300)
            plt.show()

    def plot_test_result(self, path=None, result=None, algo=None, data=None, paramlist=None, hits=None):
        if not os.path.exists(result):
            os.mkdir(result)
        if path is None or algo is None or data is None:
            raise NotImplementedError('Please provide valid path, algorithm and dataset!')
        files = os.listdir(path)
        # files_lwcase = [f.lower() for f in files if 'Testing' in f]
        # print(files_lwcase)
        for d in data:
            df = pd.DataFrame()
            for a in algo:
                file_algo = [c for c in files if a.lower() in c.lower() if 'testing' in c.lower()]
                if not file_algo:
                    continue
                with open(path + '/' + file_algo[-1], 'r') as fh:
                    df_2 = pd.read_csv(fh)

                if df.empty:
                    df['Algorithm'] = [a] * len(df_2)
                    df['Epochs'] = df_2['Epoch']
                    df['Mean Rank'] = df_2['mean_rank']
                    df['Filt Mean Rank'] = df_2['filter_mean_rank']
                    df['Norm Mean Rank'] = df_2['norm_mean_rank']
                    df['Norm Filt Mean Rank'] = df_2['norm_filter_mean_rank']
                    for hit in hits:
                        df['Hits' + str(hit)] = df_2['hits' + str(hit)]
                        df['Filt Hits' + str(hit)] = df_2['filter_hits' + str(hit)]
                        df['Norm Hits' + str(hit)] = df_2['norm_hit' + str(hit)]
                        df['Norm Filt Hits' + str(hit)] = df_2['norm_filter_hit' + str(hit)]
                else:
                    df_3 = pd.DataFrame()
                    df_3['Algorithm'] = [a] * len(df_2)
                    df_3['Epochs'] = df_2['Epoch']
                    df_3['Mean Rank'] = df_2['mean_rank']
                    df_3['Filt Mean Rank'] = df_2['filter_mean_rank']
                    df_3['Norm Mean Rank'] = df_2['norm_mean_rank']
                    df_3['Norm Filt Mean Rank'] = df_2['norm_filter_mean_rank']
                    for hit in hits:
                        df_3['Hits' + str(hit)] = df_2['hits' + str(hit)]
                        df_3['Filt Hits' + str(hit)] = df_2['filter_hits' + str(hit)]
                        df_3['Norm Hits' + str(hit)] = df_2['norm_hit' + str(hit)]
                        df_3['Norm Filt Hits' + str(hit)] = df_2['norm_filter_hit' + str(hit)]
                    frames = [df, df_3]
                    df = pd.concat(frames)

            files = os.listdir(result)
            df_4 = df.loc[df['Epochs'] == max(df['Epochs'])]
            df_4 = df_4.loc[:, df_4.columns != 'Epochs']

            file_no = len(
                [c for c in files if d.lower() in c.lower() if 'testing' in c.lower() if 'latex' in c.lower()])
            with open(result + '/' + d + '_testing_latex_table_' + str(file_no+1) + '.txt', 'w') as fh:
                fh.write(df_4.to_latex(index=False))

            file_no = len(
                [c for c in files if d.lower() in c.lower() if 'testing' in c.lower() if 'table' in c.lower() if
                 'csv' in c.lower()])
            with open(result + '/' + d + '_testing_table_' + str(file_no+1) + '.csv', 'w') as fh:
                df_4.to_csv(fh, index=False)

            df_5 = pd.DataFrame(columns=['Metrics','Algorithm','Score'])
            metrics = [f for f in df_4.columns if f != 'Algorithm']
            for i in range(len(df_4)):
                # import pdb
                # pdb.set_trace()
                if df_5.empty:
                    df_5['Algorithm']= [df_4.iloc[i]['Algorithm']]*len(metrics)
                    df_5['Metrics'] = metrics
                    df_5['Score'] = df_4.iloc[i][metrics].values
                else:
                    df_t=pd.DataFrame()
                    df_t['Algorithm'] = [df_4.iloc[i]['Algorithm']] * len(metrics)
                    df_t['Metrics'] = metrics
                    df_t['Score'] = df_4.iloc[i][metrics].values
                    frame= [df_5,df_t]
                    df_5=pd.concat(frame)

            df_6= df_5[df_5['Metrics'].str.contains('Hits') == False]
            plt.figure()
            flatui = ["#d46a7e", "#d5b60a", "#9b59b6", "#3498db", "#95a5a6", "#34495e", "#2ecc71", "#e74c3c"]
            g=seaborn.barplot(x="Metrics", y='Score', hue="Algorithm", palette=flatui, data=df_6)
            g.legend(loc='upper center',bbox_to_anchor=(0.5, 1.14),ncol=6)
            g.tick_params(labelsize=6)
            # ax = seaborn.lineplot(x="Metrics", y='Score', hue="Algorithm",
            #                       markers=True, dashes=False, data=df_5)

            files_lwcase = [f.lower() for f in files]
            file_no = len([c for c in files_lwcase if d.lower() in c if 'testing' in c if 'rank_plot' in c])
            plt.savefig(result + '/' + d + '_testing_rank_plot_' + str(file_no+1) + '.pdf', bbox_inches='tight', dpi=300)
            plt.show()

            df_6 = df_5[df_5['Metrics'].str.contains('Hits') == True]
            plt.figure()
            flatui = [ "#3498db", "#95a5a6", "#34495e", "#2ecc71", "#e74c3c","#d46a7e", "#d5b60a", "#9b59b6"]
            g = seaborn.barplot(x="Metrics", y='Score', hue="Algorithm", palette=flatui, data=df_6)
            g.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=6)
            g.tick_params(labelsize=6)

            files_lwcase = [f.lower() for f in files]
            file_no = len([c for c in files_lwcase if d.lower() in c if 'testing' in c if 'hits_plot' in c])
            plt.savefig(result + '/' + d + '_testing_hits_plot_' + str(file_no + 1) + '.pdf', bbox_inches='tight', dpi=300)
            plt.show()

    def plot_embedding(self, resultpath=None, algos=None):
        print("\t drawing figure!")
        pos = {}
        node_color_mp = {}
        for i in range(self.tot_triples):
            self.G.add_edge(self.h_name[i], self.r_name[i])

            self.G.add_edge(self.r_name[i], self.t_name[i])

            node_color_mp[self.h_name[i]] = 'r'
            node_color_mp[self.r_name[i]] = 'g'
            node_color_mp[self.t_name[i]] = 'b'

            pos[self.h_name[i]] = self.h_embs[i]
            pos[self.r_name[i]] = self.r_embs[i]
            pos[self.t_name[i]] = self.t_embs[i]
            # print(self.h_name[i], self.h_embs[i])
            # print(self.r_name[i], self.r_embs[i])
            # print(self.t_name[i], self.t_embs[i])
        colors = []
        for n in list(self.G.nodes):
            colors.append(node_color_mp[n])

        plt.figure()
        nodes_draw = nx.draw_networkx_nodes(self.G,
                                            pos,
                                            node_color=colors,
                                            node_size=50,
                                            with_labels=True)
        nodes_draw.set_edgecolor('w')
        nx.draw_networkx_labels(self.G, pos, font_size=8)
        nx.draw_networkx_edges(self.G, pos, arrows=True, width=0.5, alpha=0.5)
        # print(list(self.G.nodes))
        # print(pos)

        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        plt.savefig(resultpath + '/' + algos + '_embedding_plot.png', bbox_inches='tight', dpi=300)
        plt.show()


if __name__ == '__main__':
    v = Visualization()
    viz = Visualization()
    # viz.plot_train_result(path='../results',
    #                       result='../figures',
    #                       algo=['TransE', 'TransR', 'TransH'],
    #                       data=['Freebase15k'])
    viz.plot_test_result(path='../results',
                         result='../figures',
                         algo=['TransE', 'TransR', 'TransH'],
                         data=['Freebase15k'], paramlist=None, hits=[10, 5])

    # h_name = ['/m/07pd_j', '/m/06wxw', '/m/0d4fqn', '/m/07kcvl', '/m/012201']
    # h_embs = [[462.293, 296.02106],
    #           [476.82443, 3.0669365],
    #           [-376.1712, 3.5659008],
    #           [204.21953, 421.02557],
    #           [-229.96628, -253.05414]]
    #
    # r_name = ['/film/film/genre',
    #           '/location/location/time_zones',
    #           '/award/award_winner/awards_won.',
    #           '/american_football/football_team/historical_roster./american_football/football_historical_roster_position/position_s',
    #           '/film/music_contributor/film']
    #
    # r_embs = [[78.29823, 39.317097],
    #           [-73.834854, 472.82117],
    #           [388.3856, -286.81555],
    #           [108.599106, -383.84006],
    #           [108.599106, -383.84006]]
    #
    # t_name = ['/m/02l7c8', '/m/02fqwt', '/m/03wh8kl', '/m/0bgv8y', '/m/0ckrnn']

    # h_name = [s.replace('/','_') for s in h_name]
    # r_name = [s.replace('/', '_') for s in r_name]
    # r_name = [s.replace('.', '_') for s in r_name]
    # t_name = [s.replace('/', '_') for s in t_name]

    # t_embs = [[-248.9943, 258.7389],
    #           [-1.2189212, -176.04027],
    #           [262.3874, 161.24255],
    #           [-141.36205, 28.307116],
    #           [12.954701, 247.43892]]
    #
    # pos = {}
    # G = nx.DiGraph()
    # for i in range(5):
    #     G.add_edge(h_name[i], r_name[i])
    #     G.add_edge(r_name[i], t_name[i])
    #     pos[h_name[i]] = h_embs[i]
    #     pos[r_name[i]] = r_embs[i]
    #     pos[t_name[i]] = t_embs[i]
    #
    # plt.figure()
    # nodes_draw = nx.draw_networkx_nodes(G,
    #                                     pos,
    #                                     node_size=40,
    #                                     with_labels=True)
    # nodes_draw.set_edgecolor('w')
    # nx.draw_networkx_labels(G, pos, font_size=8)
    # nx.draw_networkx_edges(G, pos, arrows=True, width=0.5, alpha=0.5)
    #
    # plt.show()
    # if not os.path.exists('../figures'):
    #     os.mkdir('../figures')
    #
    # plt.savefig('../figures/transe_test.pdf', bbox_inches='tight', dpi=300)
    # plt.show()
