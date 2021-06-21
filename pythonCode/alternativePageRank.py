# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:03:13 2021

@author: theod
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import copy
import pageRank_geek4geeks as g4g

class WebGraph():
    
    def __init__(self, *args, connexComponents = 4, directed = True, random = True, method="barabasi_albert"):
        
        GList = [] 
        
        for i in range(connexComponents): 
            if method=="personalized": GList.append(args)
            
            elif method=="barabasi_albert":
                (nNodes, param) = args if args else (20, 8)
                GList.append(nx.barabasi_albert_graph(nNodes, param))
                
            elif method=="complete":
                nNodes = args[0] if args else 10
                GList.append(nx.complete_graph(nNodes))
        
        self.inputGraph = nx.DiGraph()
        
        for G in GList : self.inputGraph = nx.disjoint_union(self.inputGraph, G)
        
        self.G = nx.DiGraph()
        self.nNodes = self.inputGraph.number_of_nodes()
        self.adjM = np.zeros((self.nNodes, self.nNodes))
        
        self.computeAdj(directed, random)
            
        self.stochM = copy.deepcopy(self.adjM)
        
        self.computeStochM()
        
        self.ADSM = None
        
        self.computeADSM()
        
            
    def draw(self):
        nx.draw(self.G)
        
    def computeAdj(self, directed, random):
        for (startN, endN) in list(self.inputGraph.edges):
            
            self.adjM[startN][startN] = 1
            self.adjM[endN][endN] = 1
            
            if random:
                rand = np.random.rand()
                
                
                if rand <= 0.6:    #prob i->j = 2/3
                    self.adjM[startN][endN] = 1
                    self.G.add_edge(startN, endN)
                if rand >= 0.3:    #prob j->i = 2/3
                    self.adjM[endN][startN] = 1
                    self.G.add_edge(endN, startN)
            
            else:
                self.adjM[startN][endN] = 1
                self.G.add_edge(startN, endN)
                
                if not directed:
                    self.adjM[endN][startN] = 1
                    self.G.add_edge(endN, startN)
                    
                
    def computeStochM(self):
         for lineN in range(self.nNodes): #normalize stochM
            
            line = self.stochM[lineN] 
            
            if sum(line) == 0: print(lineN) 
            
            self.stochM[lineN]  = line/sum(line)
    
        
    def computeADSM(self, type = 2, graphExt = None, alpha = 0.05):
        
        self.ADSM = copy.deepcopy(self.stochM)
        
        if type == 2:    
            if not graphExt:
                AdjT = self.adjM.T
            else:
                AdjT = graphExt.adjM.T
            
            for i in range(self.nNodes):
                if sum(AdjT[i]) != 0:    
                    
                    self.ADSM[i] *= (1-alpha)
                    self.ADSM[i] += alpha*AdjT[i]/sum(AdjT[i])
                
                elif sum(self.ADSM[i]) == 0:
                    self.ADSM[i] += AdjT[i]/sum(AdjT[i])
            
        if type == 1:
            if not graphExt:
                stochMT = self.stochM.T
            else:
                stochMT = graphExt.stochM.T
            
            for i in range(self.nNodes):
                
                self.ADSM[i] *= (1-alpha)
                self.ADSM[i] += alpha*stochMT[i]/sum(stochMT[i]) if sum(stochMT[i]) != 0 else 0
        
                
        
class PageRank():
    
    def __init__(self, inputGraph = WebGraph()):
        self.Gclass = inputGraph
        self.G = inputGraph.G
        self.I = np.zeros(inputGraph.G.number_of_nodes())
        
    def __call__(self, method = "classical", **kwargs):
        
        if method == "classical":
            if kwargs:    
                self.I = g4g.pagerank(self.G, kwargs)
            else:
                self.I = g4g.pagerank(self.G)
            
        elif method == "alternative":
            if kwargs : self.I = altPageRank(self.Gclass.ADSM, kwargs) 
            else : self.I = altPageRank(self.Gclass.ADSM)
            
    def drawResults(self, inputType = "dic"):
        
        if inputType == "dic":
            
            colors = np.zeros(self.Gclass.nNodes)
            
            for key in self.I.keys():
                
                colors[key] = self.I[key]
            
            nx.draw(self.G, pos = nx.circular_layout(self.G), with_labels=True, font_weight='bold', node_color = colors, cmap=plt.get_cmap("Reds"))

        if inputType == "array":
            nx.draw(self.G, pos = nx.circular_layout(self.G), with_labels=True, font_weight='bold', node_color = self.I, cmap=plt.get_cmap("Reds"))

    def compareResults(self, **kwargs):
        #plt.subplot(221)
        
        results = []
        classements = []
        
        self(method =  "classical")
        resultsClass = np.zeros(len(self.I))
        
        for i in self.I.keys() :
            resultsClass[i] = self.I[i]
        
        classement = sorted(self.I.items(), key= lambda x : x[1], reverse = True)
        print(classement)
        classements.append(classement)
        results.append(resultsClass)
        
        #self.drawResults()
        
        print()
        
        
        plt.subplot(121)
        self(method = "alternative")
        results.append(self.I)
        
        classement = sorted(enumerate(self.I), key = lambda x : x[1], reverse = True)
        print(classement)
        classements.append(classement)
        
        self.drawResults(inputType = "array")
        
        plt.subplot(122)
        plt.title("Vecteurs d'importance obtenus par les deux méthodes de pagerank")
        plt.xlabel("Numéro de noeud")
        plt.ylabel("Score d'importance")
        plt.plot(results[-1],'b')
        plt.plot(results[-2],'r')
        
        
        
        #plt.subplot(224)
        #nx.draw(self.G,  with_labels=True, font_weight='bold', node_color = self.I, cmap=plt.get_cmap("Reds"))
        
        return results
        
        
def altPageRank(ADSM):
    
    (eigenvalues, eigenvectors) = np.linalg.eig(ADSM.T)
    indexes = np.where((1. - np.abs(eigenvalues)) < 2)
    eigenvec = eigenvectors[:, indexes].T
    
    eigenvec = [[ (np.abs(eigenvec[j][0][i]) * np.exp(abs(eigenvalues[j])))/ np.linalg.norm(np.abs(eigenvec[j][0]), ord = 1) if eigenvec[j][0][i] != 0 else 0 \
                 for i in range(len(eigenvec[j][0]))] for j in range(len(eigenvec))]
    return np.sum(eigenvec, axis = 0)/(len(eigenvec))
    

        