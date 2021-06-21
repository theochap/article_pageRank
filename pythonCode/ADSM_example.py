# -*- coding: utf-8 -*-
"""
Created on Sat May 29 11:55:45 2021

@author: theod
"""
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_nodes_from(range(5))
edges = [(0, 1, 0.25), (2, 1, 0.25), (3,1, 0.25), (5,1, 0.25), (3,2, 1), (1,4, 0.5), (5,4, 0.5), (4, 5, 0.5), (2, 5, 0.5), (1, 0, 0.25), (2,0, 0.25), (3, 0, 0.25), (5, 0, 0.25)]

G.add_weighted_edges_from(edges)

nx.draw_spectral(G, with_labels=True, font_weight='bold')