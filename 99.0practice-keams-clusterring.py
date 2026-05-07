import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x_data, _ = make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=42)
df = pd.DataFrame(x_data,columns=['F1','F2'])

kmeans = KMeans(n_clusters = 4, random_state=42,n_init=10)

kmeans.fit(df)

preductions = kmeans.predict(df)

centrioids = kmeans.cluster_centers_

plt.scatter(df['F1'],df['F2'],c=preductions,cmap='viridis',alpha=0.5)

plt.scatter(centrioids[:, 0], centrioids[:, 1], color='red', s=200, marker='X', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()