# %%
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from _02_clean import clean

# %%
# Load data
df, _ = clean()

# Scale data
df_scaled = StandardScaler().fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled

# Load targets and set index
targets = pd.read_csv('data/raw.csv')
targets = targets["TYPE"]

# Join to df
df_scaled = df_scaled.join(targets)

# %%
# Apply PCA
pca = PCA()
pcs_data = pca.fit_transform(df_scaled)

# %%
# 1. Plot cumulative explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', color='b')
plt.title('Cumulative Explained Variance by Principal Components', fontsize=16)
plt.xlabel('Number of Principal Components', fontsize=14)
plt.ylabel('Cumulative Explained Variance', fontsize=14)
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.ylim(0, 1.1)
plt.grid(True)
#plt.show()
plt.savefig('results/cumulative_explained_variance.png')
plt.close()

# %%
# 2. Plot PC1 vs PC2 with explained variance in axis titles
plt.figure(figsize=(8, 6))
plt.scatter(pcs_data[:, 0], pcs_data[:, 1], c=df_scaled['TYPE'], cmap='viridis', s=50)

# Set axis titles with explained variance
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
plt.title('PCA of Iris Dataset (PC1 vs PC2)', fontsize=16)

# Save the PC1 vs PC2 plot
#plt.show()
plt.savefig('results/pc1_vs_pc2.png')
plt.close()

# %%
# 3. Plot PC1 vs PC2 with most important loadings (longest vectors)

# Get the PCA loadings (components) and calculate their vector lengths
loadings = pca.components_.T * np.sqrt(pca.explained_variance_) * 5
loading_lengths = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)

# Define the number of most important loadings to display (e.g., top 2)
top_n = 8
top_indices = np.argsort(loading_lengths)[-top_n:]  # Indices of the top N longest vectors

plt.figure(figsize=(8, 6))

# PC1 vs PC2 scatter plot
scatter = plt.scatter(pcs_data[:, 0], pcs_data[:, 1], c=df_scaled['TYPE'], cmap='viridis', s=50)

# Add color legend for the scatter points
plt.legend(*scatter.legend_elements(), title="Type")

# Set axis titles with explained variance
plt.xlabel(f'Principal Component 1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=14)
plt.ylabel(f'Principal Component 2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=14)
plt.title(f'PCA of Iris Dataset (PC1 vs PC2) with Top {top_n} Loadings', fontsize=16)

# Add the most important loadings as arrows (vectors)
features = df.columns
for i in top_indices:
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5, head_width=0.05, head_length=0.05)
    plt.text(loadings[i, 0], loadings[i, 1], features[i], color='r', ha='center', va='center', fontsize=12)

# Save the plot with the top loadings
plt.grid(True)
#plt.show()
plt.savefig('results/pc1_vs_pc2_w_loadings.png')
plt.close()


