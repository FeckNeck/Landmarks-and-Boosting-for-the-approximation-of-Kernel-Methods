import pandas as pd
from GBRFF2 import GBRFF
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from time import time

le = LabelEncoder()

cancer = pd.read_csv("data/breast-cancer.csv", delimiter=";")
diabete = pd.read_csv("data/pima-indians-diabetes.data", header=None)
heart = pd.read_csv("data/heart.data", header=None)
sonar = pd.read_csv("data/sonar.dat", header=None)
spambase = pd.read_csv("data/spambase.dat", header=None)

sonar[60] = le.fit_transform(sonar[60])
cancer['diagnosis'] = le.fit_transform(cancer['diagnosis'])

datasets = {
    'cancer': cancer,
    'diabete': diabete,
    'heart': heart,
    'sonar': sonar,
    'spambase': spambase,
}

params = [0.1, 0.2, 0.5, 0.7, 1]

results = {
    'lambda': [],
    'gamma': [],
    'score': [],
    'time': [],
    'dataset': []
}

for i, j in datasets.items():
    X = j.iloc[:, :-1]
    y = j.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)
    for l in params:
        for g in params:
            gb = GBRFF(Lambda=l, gamma=g)
            gb.fit(X_train, y_train)

            st = time()
            y_pred = gb.predict(X_test)
            end = time()

            acc = accuracy_score(y_test, y_pred)

            score = round(acc*100, 2)
            duration = round(end - st, 4)

            results['score'].append(score)
            results['time'].append(duration)
            results['lambda'].append(l)
            results['gamma'].append(g)
            results['dataset'].append(i)

df = pd.DataFrame.from_dict(results)
maxScores = df.sort_values(
    'score', ascending=False).drop_duplicates(['dataset'])
maxScores.to_csv('Hyperparameter optimization.csv', sep=';', encoding='utf-8')
