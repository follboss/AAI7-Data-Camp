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

import networkx as nx


#%%

# Create a directed graph
G = nx.read_edgelist("edgelist.txt", nodetype=int, create_using=nx.DiGraph)
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

#%%

# montre moi les 10 premiers noeuds
# *************  - controle  
print (list(G.nodes())[0:10])
print (list(G.edges())[0:10])

#%%


# Read training data
# lire les données d'entrainement et de test

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

# le graphe en se limitant aux neoeds de train_papers
# *************  - controle  

nG = G.subgraph(train_papers)
# #%%
print('Number of nodes:', nG.number_of_nodes())
print('Number of edges:', nG.number_of_edges())


#%%

def random_walk(G, node, walk_length):
    
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
# *************  - controle  
Gun = nx.to_undirected(G)
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
# *************  - controle  
w = generate_walks(Gun, 2, 5)
# nombre d eneouds
print (len(G.nodes()))
print (len(w))
print (len(train_papers))

#%%


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    modelV = Word2Vec(vector_size=n_dim, window=3, min_count=0, sg=0, workers=8, hs=1) ####### PARAMETRES !
    modelV.build_vocab(walks)
    modelV.train(walks, total_examples=modelV.corpus_count, epochs=5)

    return modelV


#%%

modelV = deepwalk(Gun, 4, 10, 64) ####### PARAMETRES !

#%%

# montre moi le vocabulaire
# *************  - controle  
modelV.wv[6]
print (len(modelV.wv))
modelV.wv[str(train_papers[6])]




#%%
# charge les embeddings

# les poids qui viennent du traitement des abstracts

empb_train = np.load('empb_train.npy')
empb_test = np.load('empb_test.npy')
#%%


####### PARAMETRES => paramétriser le 68, le 200, la couche d'entrée du NN
def extract_features(G, nodes, embd):
    X = np.zeros((len(nodes), 268))
    print (X.shape)
    core_nums = nx.core_number(G.to_undirected())
    for i in range(len(nodes)):
        X[i,0] = G.in_degree(nodes[i])
        X[i,1] = G.out_degree(nodes[i])
        X[i,2] = core_nums[nodes[i]]
        X[i,3] = nx.clustering(G, nodes[i])
        X[i,4:68] = modelV.wv[str(nodes[i])]
        X[i,68:268] = embd[i]
    return X

X_train = extract_features(G, train_papers, empb_train)
X_test = extract_features(G, test_papers, empb_test)

#%%
# *************  - controle  
print (X_train.shape)
print (len(y_train))
#%%

from sklearn.model_selection import train_test_split
X_train , X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)



#%%
# *************  - controle  
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

##################################



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

from torch.utils.data import DataLoader, TensorDataset


#%%

#%%
class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = torch.nn.Linear(268, 128)
        self.fc2 = torch.nn.Linear(128,32)
        self.fc3 = torch.nn.Linear(32, 5)
        self.dropout = torch.nn.Dropout(0.5) #268/256/128/5 ; 0.5 ####### PARAMETRES !
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
    

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

#%%


train_dataset = TensorDataset(torch.tensor(X_train,dtype=torch.float), torch.tensor(y_train,dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(torch.tensor(X_test,dtype=torch.float), torch.tensor(y_test,dtype=torch.long))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

val_dataset = TensorDataset(torch.tensor(X_val,dtype=torch.float), torch.tensor(y_val,dtype=torch.long))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

#%%



#%%
num_epochs = 20  ####### PARAMETRES !

criterion = nn.CrossEntropyLoss()


best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    correct = 0
    total = 0
    running_loss_val = 0.0

    for i, data in enumerate(val_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        running_loss_val += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_acc = correct / total
    avg_eval_loss = running_loss_val / len(test_loader.dataset)
    
    epoch_val_loss = running_loss_val / len(val_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, epoch_loss))
    print('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch, epoch_val_loss))
    print('Validation accuracy: {:.4f}'.format(val_acc))

    ## ici on sauve les parametres pour un usage ultérieur
    is_best = (val_acc >= best_acc)
    best_acc = max(val_acc, best_acc)
    if is_best:
        torch.save({
                'state_dict': model.state_dict()
         }, 'model_best.pth.tar')
    
    
    
    # model.eval()
    # correct = 0
    # total = 0
    # eval_loss = 0.0
    
    # with torch.no_grad():
    #     for data in test_loader:
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         outputs = model(inputs.float())
    #         loss = criterion(outputs, labels)
    #         eval_loss += loss.item() * inputs.size(0)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # accuracy = correct / total
    # avg_eval_loss = eval_loss / len(test_loader.dataset)

    # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")
    # let le log_loss

     
#%%

### 

# alller rechercher le meilleur modele 
# dans le file 

checkpoint = torch.load('model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Final Test Accuracy: {accuracy:.4f}')

