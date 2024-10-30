import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from _02_clean import clean

df, _, non_bm = clean()
df_scaled = StandardScaler().fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

pca = PCA()
pcs_data = pca.fit_transform(df_scaled)
pcs_df = pd.DataFrame(pcs_data, columns=[f'PC{i+1}' for i in range(pcs_data.shape[1])])

def kmeans_clustering(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df)
    # Colors for the clusters and centers
    color_mapping = {
        'blue': 'purple',
        'yellow': 'orange',
        'cyan': 'blue',
        'pink': 'red'
    }
    cluster_colors = list(color_mapping.keys())[:n_clusters]
    center_colors = [color_mapping[color] for color in cluster_colors]
    
    # Scatter plot
    scatter = plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=kmeans.labels_, cmap=plt.cm.get_cmap('viridis', n_clusters))
    centers = plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c=center_colors, marker='x')
    
    # Plot customization
    for i, (cluster_color, center_color) in enumerate(zip(cluster_colors, center_colors)):
        plt.scatter([], [], c=cluster_color, label=f'Cluster {i + 1}', marker='o')
        plt.scatter([], [], c=center_color, label=f'Center {i + 1}', marker='x')
    plt.legend(title="Clusters and Centers")
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1 (PC1)')
    plt.ylabel('Feature 2 (PC2)')
    
    plt.show()
    return kmeans

kmeans_clustering(pcs_df, 2)