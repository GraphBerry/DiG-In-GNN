from dgl.data.utils import load_graphs
import dgl
import numpy as np
import torch
import os, tqdm
from collections import defaultdict

graph = None
graph, label_dict = load_graphs('./data/tfinance/tfinance')

graph = graph[0]
graph.ndata['label'] = graph.ndata['label'].argmax(1)

# Initialize an empty dictionary to store the neighbors of each node
homo_list = defaultdict(set)

'''
# pseudo
for node in range(len(graph.ndata['label'])):
    graph.edges[1][graph.edges[0].whereis(node)].tolist() + [node]
    toset()
    store in homo_list[node]
'''
e1 = graph.edges()[0]
e2 = graph.edges()[1]
e1.to(0)
e2.to(0)
for node in range(len(graph.ndata['label'])):
    homo_list[node] = set(np.array(e2[(graph.edges()[0] == node).nonzero(as_tuple=True)[0]].cpu()).tolist())
    homo_list[node].add(node)

# Save the dictionary to a pickle file
import pickle
with open('./data/tfinance/homo_tfinance.pickle', 'wb+') as handle:
    pickle.dump(homo_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

