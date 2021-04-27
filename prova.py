import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []
iris = pd.read_csv("C:/Users/fabio/Desktop/prova/paises.csv")
iris.head()

X = iris.iloc[:, 0:2].values
X

kmeans = KMeans(n_clusters = 5, init = 'random')
kmeans.fit(X)
kmeans.cluster_centers_

distance = kmeans.fit_transform(X)
distance

labels = kmeans.labels_
labels
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans.fit(X)
    print (i,kmeans.inertia_)
    wcss.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()
