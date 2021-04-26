import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

headers = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
dataset = pd.paises_csv("./prova", encoding = "ISO-8859-1", decimal=",", header=None, names=headers)

for col in dataset.columns[0:1]:
  dataset[col] = dataset[col].astype(float)

X = dataset.iloc[:, 0:1] #seleciona todos os dados da coluna de 0 a 4
#y = dataset.iloc[:, 4] #seleciona todos os dados da coluna 4
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
centroids = kmeans.cluster_centers_
X_clustered = kmeans.fit_predict(X)
results = dataset[['class']].copy()
results['clusterNumber'] = X_clustered

#results

print('Centroids')
print(centroids)
print('Dados classificados')
print(X_clustered)
print('Dados de entrada')
print(X)

LABEL_COLOR_MAP = {0: 'red', 1: 'blue', 2: 'green'}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

c1 = 0 # valor do índice da coluna, pode ser 0, 1 ou 
c2 = 1
labels = ['sepal length', 'sepal width', 'petal length']
c1label = labels[c1]
c2label = labels[c2]
title = c1label + ' x ' + c2label

plt.figure(figsize=(12, 12))
plt.scatter(X.iloc[:, c1], X.iloc[:, c2], c=label_color, alpha=0.3)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel(c1label, fontsize=18)
plt.ylabel(c2label, fontsize=18)
plt.suptitle(title, fontsize=20)
plt.savefig(title + '.jpg')
plt.show()