#import packages
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import cluster
from sklearn.svm import SVC
  

# %% Dataset Moons (dataset synthétique)
acc_adl = [] #liste pour les scores pour chaque taille n du dataset
elapsed_full = [] #liste pour les temps de calcul pour chaque taille n du dataset
for i in (1000,5000,10000,50000,80000,100000):
    start = time.time()
    X,Y = datasets.make_moons(n_samples=i)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.7, random_state=3)
    
    svc = svm.SVC(kernel ='linear', C = 1).fit(Xtrain, ytrain)
    ypred = svc.predict(Xtest)
    
    acc_adl.append(accuracy_score(ytest,ypred))
    end = time.time()
    elapsed_full.append(end - start)
print(acc_adl)
round_to_whole = [round(num,3) for num in elapsed_full]
print(round_to_whole)

#Graphique score à la prédiction par n données   
xpoints = ([1000,5000,10000,50000,80000,100000])
ypoints = (acc_adl)

plt.plot(xpoints, ypoints)
plt.ylabel('accuracy (en %)')
plt.xlabel('nombre de données')
plt.show()

#Graphique temps de calcul par n données 
xpoints = ([1000,5000,10000,50000,80000,100000])
ypoints = (elapsed_full)

plt.plot(xpoints, ypoints)
plt.ylabel('temps de calcul (en secondes)')
plt.xlabel('nombre de données')
plt.show()

#Test du Kmeans sur Moons mais pas utile finalement_ pas adapté pour séparer les classes
X,Y = datasets.make_moons(n_samples=1000)
acc_adl = []
elapsed=[]
for i in (3,5,6,10,15,20):
    start = time.time()
    kmeans = cluster.KMeans(n_clusters=i)
    X = kmeans.fit_transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.7, random_state=123)
    svc = SVC(kernel ='linear', C = 1).fit(Xtrain, ytrain)
    ypred = svc.predict(Xtest)
    acc_adl.append(accuracy_score(ytest,ypred))
    end = time.time()
    elapsed.append(end - start)
print(acc_adl)
print(elapsed)


# %% Dataset Swiss roll (dataset synthétique)
acc_adl = []
elapsed=[]
for i in (100,500,1000,5000,10000,50000):
    start = time.time()
    X,Y = datasets.make_swiss_roll(n_samples=i)
    y_class = np.where(Y<9.5, 0, 1) #création de la variable y car non binaire à la base
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_class, train_size=0.7, random_state=123)
    
    svc = svm.SVC(kernel ='linear', C = 1).fit(Xtrain, ytrain)
    ypred = svc.predict(Xtest)
    
    acc_adl.append(accuracy_score(ytest,ypred))
    end = time.time()
    elapsed.append(end - start)
print(acc_adl)
print(elapsed) 
   
xpoints = ([100,500,1000,5000,10000,50000])
ypoints = (acc_adl)

#Graphique score à la prédiction par n données
plt.plot(xpoints, ypoints)
plt.ylabel('accuracy (en %)')
plt.xlabel('nombre de données')
plt.show()

#Graphique temps de calcul par n données 
xpoints = ([100,500,1000,5000,10000,50000])
ypoints = (elapsed)

plt.plot(xpoints, ypoints)
plt.ylabel('temps de calcul (en secondes)')
plt.xlabel('nombre de données')
plt.show()


# %% Test sur datasets pima-indians-diabetes

import pandas as pd
from sklearn import cluster
from sklearn.svm import SVC
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

#test sur tout le dataset
acc_full = []
elapsed_full = []
for i in (0.30,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1):
    df = pd.read_csv(r'.\pima-indians-diabetes.data')
    start = time.time()
    X = df.iloc[:,:-1]
    Y = df.iloc[:, -1]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.7, random_state=3)
    svc = svm.SVC(kernel ='rbf', C = 1).fit(Xtrain, ytrain)
    ypred = svc.predict(Xtest)
    acc_full.append(accuracy_score(ytest,ypred))
    end = time.time()
    elapsed_full.append(end - start)
print(acc_full)
print(elapsed_full)
#0.70129

# Tirage aléatoire s'
acc_rand = []
elapsed_rand = []
for i in (0.30,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1):
    df = pd.read_csv(r'.\pima-indians-diabetes.data')
    start = time.time()
    df = df.sample(frac=i)
    X = df.iloc[:,:-1]
    Y = df.iloc[:, -1]
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.7, random_state=3)
    svc = svm.SVC(kernel ='rbf', C = 1).fit(Xtrain, ytrain)

    ypred = svc.predict(Xtest)
    acc_rand.append(accuracy_score(ytest,ypred))
    end = time.time()
    elapsed_rand.append(end - start)
print(acc_rand)
print(elapsed_rand)

#Détermination de la valeur optimale de K
df = pd.read_csv(r'.\pima-indians-diabetes.data')
tab=[]
for i in range(1,10):
    kmeans=cluster.KMeans(n_clusters=i)
    kmeans.fit(df)
    tab.append(kmeans.inertia_)
plt.plot(range(1,10),tab)
plt.title("La méthode Eblow")
plt.xlabel("nombre de cluster")
plt.ylabel("Inertie intra-classe")
plt.show()
#Nous remarquons sur ce graphique une courbe ayant la forme d’un bras. 
#Selon la méthode d’Elbow, la valeur optimale de K est 4.  
#Ce qui concorde pas vraiment avec l’ensemble de données utilisé divisé en 2 classes.

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = cluster.KMeans(n_clusters = k).fit(X)
  labels = kmeans.labels_
  sil.append(silhouette_score(X, labels, metric = 'euclidean'))
plt.plot(range(2,11),sil)
plt.title("Coefficient de silhouette")
plt.xlabel("nombre de cluster")
plt.ylabel("Score de silhouette")
plt.show()
  
#Kmeans avec noyau linéaire
df = pd.read_csv(r'.\pima-indians-diabetes.data')
print(df)
acc_kmeans = []
elapsed_k = []

for i in (0.30,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1):
    df = pd.read_csv(r'.\pima-indians-diabetes.data')
    start = time.time()
    df = df.sample(frac=i)
    X = df.iloc[:,:-1]
    Y = df.iloc[:, -1]
    kmeans = cluster.KMeans(n_clusters=2)
    X = kmeans.fit_transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.7, random_state=3)
    svc = SVC(kernel ='linear', C = 1).fit(Xtrain, ytrain)
    ypred = svc.predict(Xtest)
    acc_kmeans.append(accuracy_score(ytest,ypred))
    end = time.time()
    elapsed_k.append(end - start)
print(acc_kmeans)
print(elapsed_k)

#Kmeans avec noyau gaussien
df = pd.read_csv(r'.\pima-indians-diabetes.data')
print(df)
acc_kmeans = []
elapsed_k = []

for i in (0.30,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1):
    df = pd.read_csv(r'.\pima-indians-diabetes.data')
    start = time.time()
    df = df.sample(frac=i)
    X = df.iloc[:,:-1]
    Y = df.iloc[:, -1]
    kmeans = cluster.KMeans(n_clusters=2)
    X = kmeans.fit_transform(X)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.7, random_state=3)
    svc = SVC(kernel='rbf', gamma=0.01).fit(Xtrain, ytrain)
    ypred = svc.predict(Xtest)
    acc_kmeans.append(accuracy_score(ytest,ypred))
    end = time.time()
    elapsed_k.append(end - start)
print(acc_kmeans)
print(elapsed_k)
#à priori on va partir sur le noyau gaussien donc l'autre pas très utile

#Graphique temps de calcul par n données
# à améliorer pour mettre chaque courbe (tirage aléatoire, kmeans et sur tout le dataset)
#et l'autre graphique pour le score par le % du dataset
xpoints = ([0.30,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1])
ypoints = (elapsed_full)
ypoints2 = (elapsed_k)
ypoints3 = (elapsed_rand)


plt.plot(xpoints, ypoints, label='All dataset')
plt.plot(xpoints,ypoints2, label='Kmeans')
plt.plot(xpoints,ypoints3, label='Rand landmarks')
plt.legend(loc="upper left")
plt.ylabel('temps de calcul (en secondes)')
plt.xlabel('prct de données')
plt.title('')
plt.show()

xpoints = ([0.30,0.40,0.45,0.50,0.60,0.70,0.80,0.90,1])
ypoints = (acc_full)
ypoints2 = (acc_kmeans)
ypoints3 = (acc_rand)


plt.plot(xpoints, ypoints, label='All dataset')
plt.plot(xpoints,ypoints2, label='Kmeans')
plt.plot(xpoints,ypoints3, label='Rand Landmarks')
plt.legend(loc="upper left")
plt.ylabel('accuracy (en %)')
plt.xlabel('prct de données')
plt.title('')
plt.show()

