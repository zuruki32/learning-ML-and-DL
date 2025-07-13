import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x,y = make_blobs(n_samples=100, centers=5, random_state=0, cluster_std=5)

plt.scatter(x[:,0],x[:,1],s=50)
plt.show()
