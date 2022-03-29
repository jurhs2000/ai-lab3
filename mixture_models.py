import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score

# setting seed
np.random.seed(0)
random.seed(0)
RANDOM_STATE = 0


def execute_mixture_models(dataset, n_clusters):
    dataset = dataset.sample(frac=0.05) # Cambiar para manipular graficas o utilizar el dataset completo
    clusters = GaussianMixture(n_components=n_clusters, covariance_type='full')
    clusters.fit(dataset)

    dataset['cluster'] = clusters.predict(dataset)

    # Obteniendo 3 componentes principales
    pca = PCA(3)
    pca_dataset = pca.fit_transform(dataset)
    pca_df = pd.DataFrame(data = pca_dataset, columns = ['PC1', 'PC2', 'PC3'])
    pca_clusters = pd.concat([pca_df, dataset[['cluster']]], axis = 1)

    dataset.drop(['cluster'], axis=1, inplace=True)
    # Graficando clusters 3D
    fig = plt.figure(1, figsize=(16, 12))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(pca_clusters['PC1'], pca_clusters['PC2'], pca_clusters['PC3'], c=clusters.predict(dataset).astype(float), edgecolor='k')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Clusters")
    ax.dist = 12
    plt.show()

    # silhouette
    cluster_labels = clusters.fit_predict(dataset)
    silhouette_avg = silhouette_score(dataset, cluster_labels)
    print("For clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(dataset, cluster_labels)

    fig, (ax) = plt.subplots(1)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(pca_clusters) + (5 + 1) * 10])
    ax.scatter(x = pca_clusters.PC1, y = pca_clusters.PC2, marker="$%d$" % 5, alpha=1, s=50, edgecolor="k")

    y_lower = 10
    for i in range(5):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / 5)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for 5 clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()
