# -*- coding: utf-8 -*-
"""
Created on Mon May 10 22:59:13 2021

@author: theod
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

G1 = nx.barabasi_albert_graph(100, 5)

plt.subplot(121)

nx.draw(G1, with_labels=True, font_weight='bold')

plt.subplot(122)

G2 = nx.barabasi_albert_graph(100, 5)
nx.draw(G2, with_labels=True, font_weight='bold',node_color=np.random.random(100),
cmap=plt.get_cmap("Reds"))