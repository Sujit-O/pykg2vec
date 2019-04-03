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


seaborn.set_style("darkgrid")


class Visualization(object):
    def __init__(self, triples= None, idx2entity = None, idx2relation= None):
        self.triples = triples
        self.tot_triples = len(triples)
        self.h_emb = []
        self.t_emb = []
        self.r_emb = []
        self.h_name = []
        self.r_name = []
        self.t_name = []
        self.idx2entity = idx2entity
        self.idx2relation = idx2relation
        self.G=nx.DiGraph()
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
        x = np.concatenate((self.h_emb, self.r_emb,self.t_emb), axis=0)
        x_reduced = TSNE(n_components=2).fit_transform(x)

        self.h_embs = x_reduced[:length,:]
        self.r_embs = x_reduced[length:2*length, :]
        self.t_embs = x_reduced[2*length:3*length, :]

        print("dimension self.h_emb", np.shape(self.h_embs))
        print("dimension self.r_emb", np.shape(self.r_embs))
        print("dimension self.t_emb", np.shape(self.t_embs))

        # print(self.h_embs)
        # print(self.r_embs)
        # print(self.t_embs)


    def draw_figure_v2(self,
                       triples,
                       h_name,
                       r_name,
                       t_name,
                       h_embs,
                       r_embs,
                       t_embs,
                       fig_name='test_figure'):
        print("\t drawing figure!")
        pos = {}
        node_color_mp = {}

        for i in range(len(triples)):
            self.G.add_edge(h_name[i], r_name[i])

            self.G.add_edge(r_name[i], t_name[i])

            node_color_mp[h_name[i]] = 'r'
            node_color_mp[r_name[i]] = 'g'
            node_color_mp[t_name[i]] = 'b'

            pos[h_name[i]] = h_embs[i]
            pos[r_name[i]] = r_embs[i]
            pos[t_name[i]] = t_embs[i]

        colors=[]
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

        if not os.path.exists('../figures'):
            os.mkdir('../figures')

        plt.savefig('../figures/'+fig_name+'.png', bbox_inches='tight', dpi=300)
        plt.show()

    def draw_figure(self):
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
        colors=[]
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

        if not os.path.exists('../figures'):
            os.mkdir('../figures')

        plt.savefig('../figures/transe_test.png', bbox_inches='tight', dpi=300)
        plt.show()


if __name__=='__main__':
    # v = Visualization()

    h_name = ['/m/07pd_j','/m/06wxw','/m/0d4fqn','/m/07kcvl','/m/012201']
    h_embs  = [[462.293, 296.02106],
              [476.82443,3.0669365],
              [-376.1712,3.5659008],
              [204.21953, 421.02557],
              [-229.96628, -253.05414]]

    r_name = ['/film/film/genre',
              '/location/location/time_zones',
              '/award/award_winner/awards_won.',
              '/american_football/football_team/historical_roster./american_football/football_historical_roster_position/position_s',
              '/film/music_contributor/film']

    r_embs = [[78.29823,39.317097],
             [-73.834854, 472.82117 ],
             [ 388.3856 ,-286.81555],
             [ 108.599106 ,-383.84006 ],
              [ 108.599106 ,-383.84006 ]]

    t_name = [ '/m/02l7c8', '/m/02fqwt','/m/03wh8kl','/m/0bgv8y','/m/0ckrnn']

    # h_name = [s.replace('/','_') for s in h_name]
    # r_name = [s.replace('/', '_') for s in r_name]
    # r_name = [s.replace('.', '_') for s in r_name]
    # t_name = [s.replace('/', '_') for s in t_name]

    t_embs = [[-248.9943,258.7389],
             [  -1.2189212,-176.04027],
             [262.3874,161.24255],
             [-141.36205,28.307116],
             [ 12.954701,247.43892 ]]

    pos = {}
    G=nx.DiGraph()
    for i in range(5):
        G.add_edge(h_name[i], r_name[i])
        G.add_edge(r_name[i], t_name[i])
        pos[h_name[i]] = h_embs[i]
        pos[r_name[i]] = r_embs[i]
        pos[t_name[i]] = t_embs[i]

    plt.figure()
    nodes_draw = nx.draw_networkx_nodes(G,
                                        pos,
                                        node_size=40,
                                        with_labels=True)
    nodes_draw.set_edgecolor('w')
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, arrows=True, width=0.5, alpha=0.5)

    plt.show()
    if not os.path.exists('../figures'):
        os.mkdir('../figures')

    plt.savefig('../figures/transe_test.pdf', bbox_inches='tight', dpi=300)
    plt.show()



