import networkx as nx
import numpy as np

class Graph:
    """ Create the network topology """
    def __init__(self,graphType='erdos_renyi',number_of_nodes=10,probability=0.5):
        self.graphType=graphType
        self.n=number_of_nodes
        self.p=probability
        self.W=np.eye(number_of_nodes)
    
    
    def add_edge_weights(self,G, n):
        W = np.zeros(shape=(n,n))
        for i in range(0, n):
            for j in range(0, n):
                if i == j:
                    W[i][j] = 0
                else:
                    if(G.has_edge(i,j)):
                        deg1 = G.degree(i)
                        deg2 = G.degree(j)
                        if deg1 >= deg2:
                            W[i][j] = 1. * 1/deg1
                        else:
                            W[i][j] = 1. * 1/deg2
                    else:
                        continue
        for i in range(0, n):
            sum = 0
            for j in range(0, n):
                if j == i:
                    continue
                sum = sum + W[i][j]
            W[i][i] = 1 - sum

        self.W=W
        
        
    def create_graph(self):
        if(self.graphType == 'erdos_renyi'):
            G = nx.erdos_renyi_graph(self.n, self.p)
        if(self.graphType == 'star'):
            G = nx.star_graph(self.n)
        if(self.graphType == 'ring'):
            G = nx.cycle_graph(self.n)

        while(nx.is_connected(G) == False):
            if(self.graphType == 'erdos_renyi'):
                G = nx.erdos_renyi_graph(self.n, self.p)
            if(self.graphType == 'star'):
                G = nx.star_graph(self.n)
            if(self.graphType == 'ring'):
                G = nx.cycle_graph(self.n)
        self.G=G        
        self.add_edge_weights(G, self.n)
        
        
    def weight_matrix(self):
        return self.W
    
    def ShowPlot(self):
        nx.draw(self.G)
    
   