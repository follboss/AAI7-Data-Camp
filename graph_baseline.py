import csv
import numpy as np
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss

# Create a directed graph
G = nx.read_edgelist("edgelist.txt", nodetype=int, create_using=nx.DiGraph)
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

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

# Create the test matrix. Use the same 3 features as above
X_test = np.zeros((len(test_papers), 3))
#avg_neig_deg = nx.average_neighbor_degree(G, nodes=test_papers)
for i in range(len(test_papers)):
    X_test[i,0] = G.in_degree(test_papers[i])
    X_test[i,1] = G.out_degree(test_papers[i])
    X_test[i,2] = core_nums[test_papers[i]]

print("Train matrix dimension: ", X_train.shape)
print("Test matrix dimension: ", X_test.shape)

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