#%% 
import csv
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss


#%%
# ici on lit les abstracts des papiers
# on crée les jeux de données  de base 

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


#%%
# Compute accuracy and log loss
y_test = np.loadtxt('y_test_proba.txt', dtype=int, delimiter=',') 
y_test = np.argmax(y_test[:,1:], axis=1)

#%%

# y_train doit etres encodé en array (onehotencoder)
# pour la classification 

y_train = np.array(y_train)
y_train_oh = np.zeros((len(y_train), 5))
y_train_oh[np.arange(len(y_train)), y_train] = 1

y_test = np.array(y_test)
y_test_oh = np.zeros((len(y_test), 5))
y_test_oh[np.arange(len(y_test)), y_test] = 1

y_train = y_train_oh
y_test = y_test_oh


#%%

# le truc du datacamp mais à retravailler (avec les fonctions de GESIM ? )

import re

def clean_str(string):
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().split()


def preprocessing(docs):
  preprocessed_docs = []
  for doc in docs:
    preprocessed_docs.append(clean_str(doc))
  return preprocessed_docs

processed_train = preprocessing(train_abstracts)
processed_test = preprocessing(test_abstracts)
#%%

### ***** controle 
print('example of  processed review: ', processed_train[13])

#%%

# on crée le vocabulaire 
#########################

def get_vocab(processed_docs, vocab=None):
  
  if vocab is None:
    vocab = dict() # {} : empty dict

  for doc in processed_docs:
    for word in doc:
      if word not in vocab:
        vocab[word] = len(vocab) + 1 # step1: {'the': 1}; step2: {'the': 1, 'rock':2}

  return vocab

vocab = get_vocab(processed_train)
vocab = get_vocab(processed_test, vocab)
#%%

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

############
#ici on va précalculer les vecteurs des mots avec les vecteurs de Google
# honnetement c'est tres décevant à ce stade, il y a pres d'un mot sur deux non reconnu ... à traiter ...

def load_embeddings(fname, vocab):
  # fill the gaps here
  embeddings = np.zeros((len(vocab)+1, 300)) # put the dimensions of the embedding matrix # +1: padding special token

  o = 0
  u = 0
  model = KeyedVectors.load_word2vec_format(fname, binary=True)
  for word in vocab:
    if word in model:
      embeddings[vocab[word]] = model[word]
      o += 1
    else:
      embeddings[vocab[word]] = np.random.uniform(-0.25, 0.25, 300)
      u += 1
    
  print (o, u)
  return embeddings

path_to_embeddings = 'GoogleNews-vectors-negative300.bin'
embeddings = load_embeddings(path_to_embeddings, vocab)

#%%


# montre moi les premieres lignes de l'embedding matrix
# ****** controle pas à pas
print(embeddings[:5])
embeddings.shape



#%%
# on regarde le nombre de mots maximum dans les textes pour faire la taille du vecteur d'entrée
# 
max_len = 0
for doc in processed_train:
  max_len = np.max((len(doc), max_len))
print(max_len)
for doc in processed_test:
    max_len = np.max((len(doc), max_len))
print(max_len)

#%%

######
# ici on construit X_train et X_test

X_train = np.zeros((len(processed_train), max_len))
for i, doc in enumerate(processed_train):
  for j, word in enumerate(doc):
     X_train[i][j] = vocab[word]

X_test = np.zeros((len(processed_test), max_len))
for i, doc in enumerate(processed_test):
  for j, word in enumerate(doc):
     X_test[i][j] = vocab[word]

#%%

# **** controle pas à pas
embeddings.shape
len(vocab)

#%%
################################################
################################################
################################################
# Go classification 
# en CGG

# en keras (plus facile à lire et à comprendre)

from tensorflow.keras.layers import Input, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.layers import Dropout, Dense, Concatenate
from tensorflow.keras.models import Model

#%%
my_input = Input(shape=(max_len,))

embedding = Embedding(input_dim=embeddings.shape[0], 
                      output_dim=embeddings.shape[1],
                      weights=[embeddings], 
                      input_length=max_len
                      )(my_input)

####### PARAMETRES !

conv4 = Conv1D(filters = 100,
              kernel_size = 4,
              activation = 'relu',
              )(embedding)

conv3 = Conv1D(filters = 100,
              kernel_size = 3,
              activation = 'relu',
              )(embedding)

maxpool4 = GlobalMaxPooling1D()(conv4)

maxpool3 = GlobalMaxPooling1D()(conv3)

conc = Concatenate()([maxpool4, maxpool3])
res = Dropout(0.5)(conc)
res = Dense(5, activation='softmax')(res)

model = Model(my_input, res)


#%%

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#%%
#################
###########
## OBSERVATION

## ca c'est pour regarder la structure de la couche 6 
embeddingmodel = Model(model.input, model.get_layer('concatenate_1').output)

# ****  controle pas à pas 

n_plot = 1000
print('plotting embeddings of first',n_plot,'documents')

u = [np.array(X_test[:n_plot])]
empb = embeddingmodel.predict(u)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

my_pca = PCA(n_components=10)
my_tsne = TSNE(n_components=2,perplexity=10) #https://lvdmaaten.github.io/tsne/
doc_emb_pca = my_pca.fit_transform(empb)
doc_emb_tsne = my_tsne.fit_transform(doc_emb_pca)

y_lab = np.argmax(y_test, axis=1)

labels_plt = y_lab[:n_plot].astype(np.int32)
my_colors = ['blue','red', 'yellow', 'green', 'orange']

fig, ax = plt.subplots()

for label in list(set(labels_plt)):
  idxs = [idx for idx,elt in enumerate(labels_plt) if elt==label]
  ax.scatter(doc_emb_tsne[idxs,0],
              doc_emb_tsne[idxs,1],
              c = my_colors[label],
              label=str(label),
              alpha=0.7,
              s=10)

ax.legend(scatterpoints=1)
fig.suptitle('t-SNE visualization of CNN-based doc embeddings \n (first 1000 docs from test set)', fontsize=10)
fig.set_size_inches(6,4)


#%%

## LANCEMENT DU GNN - ENTRAINEMENT 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


early_stopping = EarlyStopping(monitor='val_accuracy', # go through epochs as long as accuracy on validation set increases
                               patience=2,
                               mode='max')

# make sure that the model corresponding to the best epoch is saved
checkpointer = ModelCheckpoint(filepath='cnn_text_categorization.keras',
                               monitor='val_accuracy',
                               save_best_only=True,
                               verbose=0)



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(np.array(X_train),
          y_train,
          batch_size=64,
          epochs=10, ######### PARAMETRES !! 
          callbacks=[early_stopping, checkpointer],
          validation_data=(np.array(X_test), y_test))


#%%
####  Reprise du meilleur modèle apres l'entrainement 

from tensorflow.keras.models import load_model
model = load_model('cnn_text_categorization.keras')

#%% 
## ici on peut réexecuter le code de visuasation précédent pour visualiser :-) 
##je ne le copie pas

###
# calculer la prédiction sur X_test et calculer : l'accuracy et la loss_log

y_pred = model.predict(np.array(X_test))

y_pred_c = np.argmax(y_pred, axis=1)
y_test_c = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred_c == y_test_c)
print (accuracy)


print('Accuracy:', accuracy_score(y_test_c, y_pred_c))
cross_entropy_loss = log_loss(y_test, y_pred)
print('Log loss:', cross_entropy_loss)


#%%%

# on fait un predict sur le modele

##################
# ICI ON ENREGISTRE LES EMBBEDINGS POUR LES UTILISER DANS UN AUTRE MODELE
##################

#%%
empb_train = embeddingmodel.predict(np.array(X_train))
empb_test = embeddingmodel.predict(np.array(X_test))

# on veut sauvegarder les embeddings pour les utiliser dans un autre modèle
np.save('empb_train.npy', empb_train)
np.save('empb_test.npy', empb_test)
