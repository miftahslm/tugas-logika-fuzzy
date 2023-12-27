import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("AMZN.csv")
features = data[['High', 'Low']]
k = int(input("Masukan Jumlah Cluster : "))
kmeans = KMeans(n_clusters=k)
data['Cluster_KMeans'] = kmeans.fit_predict(features)

plt.scatter(data['High'], data['Low'], c=data['Cluster_KMeans'], cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='black', label='Centroids')
plt.xlabel('High')
plt.ylabel('Low')
plt.title(f'Cluster_KMeans Clustering')
plt.legend()
plt.show()



