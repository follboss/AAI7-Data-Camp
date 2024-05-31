#%%

import csv
import numpy as np
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss


import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec

#%%

import networkx as nx


#%%

# Create a directed graph
G = nx.read_edgelist("edgelist.txt", nodetype=int, create_using=nx.DiGraph)
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

#%%


# Read training data
train_papers = list()
y_train = list()
with open("y_train.txt", "r") as f:
    for line in f:
        t = line.split(',')
        train_papers.append(int(t[0]))
        y_train.append(int(t[1][:-1]))

# Read test data
test_papers = list()
with open("test.txt", "r") as f:
    for line in f:
        t = line.split(',')
        test_papers.append(int(t[0]))

#%%

# Create the training matrix. Each row corresponds to a research paper.
# Use the following 3 features for each paper:
# (1) out-degree of node
# (2) in-degree of node
# (3) average degree of neighborhood of node
core_nums = nx.core_number(G.to_undirected())
X_train = np.zeros((len(train_papers), 3))
#avg_neig_deg = nx.average_neighbor_degree(G, nodes=train_papers)
for i in range(len(train_papers)):
    X_train[i,0] = G.in_degree(train_papers[i])
    X_train[i,1] = G.out_degree(train_papers[i])
    X_train[i,2] = core_nums[train_papers[i]]
    


#%%

# extraire un sous graphe le plus cohérent possible de G  de 1000 noeuds

nG = nx.subgraph(G, train_papers)

print('Number of nodes:', nG.number_of_nodes())
print('Number of edges:', nG.number_of_edges())


#%%

import matplotlib.pyplot as plt


#%%



#%%

train_papers
#%%

def extract_features(G, nodes):
    X = np.zeros((len(nodes), 4))
    core_nums = nx.core_number(G.to_undirected())
    for i in range(len(nodes)):
        X[i,0] = G.in_degree(nodes[i])
        X[i,1] = G.out_degree(nodes[i])
        X[i,2] = core_nums[nodes[i]]
        X[i,3] = nx.clustering(G, nodes[i])

    
    return X

X_train = extract_features(G, train_papers)
#%%

X_test = extract_features(G, test_papers)

#%%
#%%

from nodevectors import Node2Vec

#%%

import nodevectors



#%%



def random_walk(G, node, walk_length):
    
    ##################
    # your code here #
    ##################
    walk = [node]

    walki = node

    for i in range(walk_length-1):
        # on cherche les voisins de node dans le graphe
        neighbors = list(G.neighbors(walki))
        # si node n'a pas de voisin, on arrête le random walk
        if len(neighbors) == 0:
            break
        # on choisit un voisin aléatoirement
        walki = neighbors[randint(0, len(neighbors)-1)]
        # on ajoute le voisin à la marche
        walk.append(walki)

    # on retourne la marche
    walkL = [str(n) for n in walk]
    return walkL


#%%

#%% 

Gun = nx.to_undirected(G)

#%%
u = random_walk(Gun, 1, 10)
print(u)

#%%


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    
    nodes = list(G.nodes())

    nodes = np.random.permutation(nodes)

    for i in range(num_walks):
        for node in nodes:
            walk = random_walk(G, node, walk_length)
            walks.append(walk)
    
    return walks

#%%

w = generate_walks(Gun, 2, 5)


#%%

# nombre d eneouds
print (len(G.nodes()))

#%%

print (len(w))


#%%


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=3, min_count=0, sg=0, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model


#%%

model = deepwalk(Gun, 4, 10, 40)

#%%

# montre moi le vocabulaire

model.wv[6]

#%%

# taille de l'embedding

# taille du vocabulaire

print (len(model.wv))


model.wv[str(train_papers[6])]
#%%%


#%%


def extract_features(G, nodes):
    X = np.zeros((len(nodes), 24))
    core_nums = nx.core_number(G.to_undirected())
    for i in range(len(nodes)):
        X[i,0] = G.in_degree(nodes[i])
        X[i,1] = G.out_degree(nodes[i])
        X[i,2] = core_nums[nodes[i]]
        X[i,3] = nx.clustering(G, nodes[i])
        X[i,4:] = model.wv[str(nodes[i])]
    return X

X_train = extract_features(G, train_papers)
X_test = extract_features(G, test_papers)



#%%

print("Train matrix dimension: ", X_train.shape)
print("Test matrix dimension: ", X_test.shape)

#%%

# Use logistic regression to classify the research papers of the test set
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# Compute accuracy and log loss
y_test = np.loadtxt('y_test_proba.txt', dtype=int, delimiter=',') 
y_test = np.argmax(y_test[:,1:], axis=1)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Log loss:', log_loss(y_test, y_pred_proba))


#%%





#%%



import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

# Charger les données
dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Entraînement du modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Évaluation du modèle
model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')