import csv
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss

# Read abstracts of research papers
abstracts = dict()
with open("abstracts.txt", "r") as f:
    for line in f:
        #print(line)
        t = line.split('||')
        abstracts[int(t[0])] = t[1][:-1]
    
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

# Retrieve abstracts of papers of the training set
train_abstracts = [abstracts[node] for node in train_papers]

# Retrieve abstracts of papers of the test set
test_abstracts = [abstracts[node] for node in test_papers]

# Create the training matrix. Each row corresponds to a research paper and each column to a word present in one or more of the
# papers' abstracts. The value of each entry in a row is equal to the frequency of that word in the corresponding abstract     
vec = CountVectorizer()
X_train = vec.fit_transform(train_abstracts)

# Create the test matrix following the same approach as in the case of the training matrix
X_test = vec.transform(test_abstracts)

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