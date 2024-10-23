#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data set
iris = datasets.load_iris()

# Convert to pandas 
df = pd.DataFrame(iris.data)
df['class'] = iris.target
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True)

# Extract only the data and scale
x = df.loc[:, ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].values
x = StandardScaler().fit_transform(x)

# Write scaled data to a new pandas dataframe
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_iris = pd.DataFrame(x, columns=feat_cols)

# Apply PCA
pca_iris = PCA()
pcs_data = pca_iris.fit_transform(x)

# 1. Plot cumulative explained variance
explained_variance_ratio = pca_iris.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance by Principal Components', fontsize=16)
plt.xlabel('Number of Principal Components', fontsize=14)
plt.ylabel('Cumulative Explained Variance', fontsize=14)
plt.xticks([1, 2, 3, 4])
plt.ylim(0, 1)
plt.grid(True)
plt.savefig('results/cumulative_explained_variance.png')
plt.close()

# 2. Plot PC1 vs PC2 with explained variance in axis titles
plt.figure(figsize=(8, 6))
plt.scatter(pcs_data[:, 0], pcs_data[:, 1], c=df['class'], cmap='viridis', s=50)

# Set axis titles with explained variance
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
plt.title('PCA of Iris Dataset (PC1 vs PC2)', fontsize=16)

# Save the PC1 vs PC2 plot
plt.savefig('results/pc1_vs_pc2.png')
plt.close()

# 3. Plot PC1 vs PC2 with most important loadings (longest vectors)

# Get the PCA loadings (components) and calculate their vector lengths
loadings = pca_iris.components_.T * np.sqrt(pca_iris.explained_variance_)
loading_lengths = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)

# Define the number of most important loadings to display (e.g., top 2)
top_n = 2
top_indices = np.argsort(loading_lengths)[-top_n:]  # Indices of the top N longest vectors

plt.figure(figsize=(8, 6))

# PC1 vs PC2 scatter plot
plt.scatter(pcs_data[:, 0], pcs_data[:, 1], c=df['class'], cmap='viridis', s=50)

# Set axis titles with explained variance
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
plt.title(f'PCA of Iris Dataset (PC1 vs PC2) with Top {top_n} Loadings', fontsize=16)

# Add the most important loadings as arrows (vectors)
features = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
for i in top_indices:
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5, head_width=0.05, head_length=0.05)
    plt.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, features[i], color='r', ha='center', va='center', fontsize=12)

# Save the plot with the top loadings
plt.grid(True)
plt.savefig('results/pc1_vs_pc2_w_loadings.png')
plt.close()
