import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from GBRFF2 import GBRFF

# N = [1000, 5000, 10000, 15000, 25000, 50000, 100000, 250000, 500000,
#      1000000, 2500000, 5000000, 10000000]

N = [1000, 5000, 10000, 15000, 25000, 50000, 100000, 250000, 500000, 1000000]

#N = [1000, 5000, 10000, 15000, 25000, 50000]

vec_kmeans = []

moons = make_moons(n_samples=1000)

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(moons[0])
    vec_kmeans.append(kmeans.inertia_)

plt.plot(range(1, 10), vec_kmeans)
plt.title("La méthode Eblow")
plt.xlabel("nombre de cluster")
plt.ylabel("Inertie intra-classe")
plt.show()
# --> Best Kmeans for moons : 4 -- #


gb = GBRFF()
svm = SVC(kernel='linear', C=1)
kmeans = Pipeline([('kmeans', KMeans(n_clusters=4)), ('svc', svm)])
random_landmarks = SVC(kernel='rbf')

methods = {
    'gb': gb,
    'svm': svm,
    'kmeans': kmeans,
    'random_landmarks': random_landmarks,
}


results = {
    'time': [],
    'score': []
}


for i, j in methods.items():
    for n in N:
        X, Y = make_moons(n_samples=n)
        if i == 'random_landmarks':
            df = pd.DataFrame({0: X[:, 0], 1: X[:, 1], 2: Y})
            df = df.sample(frac=0.4)
            X, Y = df.iloc[:, :-1], df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, train_size=0.7, random_state=123)
        j.fit(X_train, y_train)

        st = time()
        y_pred = j.predict(X_test)
        end = time()

        acc = accuracy_score(y_test, y_pred)

        score = round(acc*100, 2)
        duration = round(end - st, 4)

        results['time'].append(duration)
        results['score'].append(score)

    plt.figure(1)
    plt.plot(N, results['time'], label=i)
    plt.figure(2)
    plt.plot(N, results['score'], label=i)

    results['time'] = []
    results['score'] = []

plt.figure(1)
plt.ylabel('Temps en secondes')
plt.xlabel('Nombre de données')
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=3)
plt.savefig('time_execution.png')

plt.figure(2)
plt.ylabel('Accuracy score')
plt.xlabel('Nombre de données')
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
           mode="expand", borderaxespad=0, ncol=3)
plt.savefig('accuracy_score.png')
plt.show()
