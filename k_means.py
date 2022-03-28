import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# setting seed
np.random.seed(0)
random.seed(0)
RANDOM_STATE = 0

def execute_kmeans(dataset, n_clusters, max_iter):
    dataset = dataset.sample(frac=0.005)
    clusters=  KMeans(n_clusters=n_clusters, max_iter=max_iter) #Creacion del modelo
    clusters.fit(dataset) #Aplicacion del modelo de cluster

    dataset['cluster'] = clusters.labels_ #Asignacion de los clusters

    pca = PCA(3)
    pca_dataset = pca.fit_transform(dataset)
    pca_df = pd.DataFrame(data = pca_dataset, columns = ['PC1', 'PC2', 'PC3'])
    pca_clusters = pd.concat([pca_df, dataset[['cluster']]], axis = 1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('Clusters', fontsize = 20)

    pca_clusters.plot.scatter(x='PC1', y='PC2', c='cluster', colormap='viridis', s=50, ax=ax)
    plt.show()

    fig = plt.figure(1, figsize=(16, 12))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(pca_clusters['PC1'], pca_clusters['PC2'], pca_clusters['PC3'], c=clusters.labels_.astype(float), edgecolor='k')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Clusters")
    ax.dist = 12
    plt.show()

