import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_bm = pd.read_csv('data/preprocessed.csv')[['HE4','CEA']]
df_meta = pd.read_csv('data/non_bm.csv')[['TYPE']]

df_bm_custom = pd.read_csv('data/preprocessed_custom_split.csv')[['HE4','CEA']]
df_meta_custom = pd.read_csv('data/non_bm_custom_split.csv')[['TYPE']]

# Split data
train_df = df_bm.iloc[:235, :]
train_labels_df = df_meta.iloc[:235]
test_df = df_bm.iloc[235:, :]
test_labels_df = df_meta.iloc[235:]

train_df_custom = df_bm_custom.iloc[:235, :]
train_labels_df_custom = df_meta_custom.iloc[:235]
test_df_custom = df_bm_custom.iloc[235:, :]
test_labels_df_custom = df_meta_custom.iloc[235:]

# Define colors for TYPE. Color is created for All, Train, Test, Train_custom, Test_custom (as the order and length of TYPE is different)
colors_all = df_meta
colors_all = colors_all.replace(1, 0.4)
colors_all = colors_all.replace(0, 1)

colors_train = train_labels_df
colors_train = colors_train.replace(1, 0.4)
colors_train = colors_train.replace(0, 1)

colors_test = test_labels_df
colors_test = colors_test.replace(1, 0.4)
colors_test = colors_test.replace(0, 1)

colors_train_custom = train_labels_df_custom
colors_train_custom = colors_train_custom.replace(1, 0.4)
colors_train_custom = colors_train_custom.replace(0, 1)

colors_test_custom = test_labels_df_custom
colors_test_custom = colors_test_custom.replace(1, 0.4)
colors_test_custom = colors_test_custom.replace(0, 1)

# Plot
sns.set(style="whitegrid")
fig, axs = plt.subplots(2,3, figsize=(18, 10), gridspec_kw={'hspace': 0.2})
axs = axs.flatten()
for ax in axs:
    ax.set_xlim(-1.8, 4)
    ax.set_ylim(-2.4, 5.25)

custom_palette = {0: 'red', 1: 'lime'}    
sns.scatterplot(x='HE4', y='CEA', hue='TYPE', palette=custom_palette, ax=axs[0], alpha=colors_all, data=pd.concat([df_bm, df_meta], axis=1), legend=False)
sns.scatterplot(x='HE4', y='CEA', hue='TYPE', palette=custom_palette, ax=axs[1], alpha=colors_train, data=pd.concat([train_df, train_labels_df], axis=1), legend=False)
sns.scatterplot(x='HE4', y='CEA', hue='TYPE', palette=custom_palette, ax=axs[2], alpha=colors_test, data=pd.concat([test_df, test_labels_df], axis=1), legend=False)
sns.scatterplot(x='HE4', y='CEA', hue='TYPE', palette=custom_palette, ax=axs[4], alpha=colors_train_custom, data=pd.concat([train_df_custom, train_labels_df_custom], axis=1), legend=False)
sns.scatterplot(x='HE4', y='CEA', hue='TYPE', palette=custom_palette, ax=axs[5], alpha=colors_test_custom, data=pd.concat([test_df_custom, test_labels_df_custom], axis=1), legend=False)

# Remove axes for the empty subplot
axs[3].axis('off')

# Set titles for each subplot
axs[0].set_title('All')
axs[1].set_title('Train, predefined split')
axs[2].set_title('Test, predefined split')
axs[4].set_title('Train, custom split')
axs[5].set_title('Test, custom split')

# Remove x-axis labels for consistency
axs[1].set_xlabel('')
axs[2].set_xlabel('')

# Remove y-axis labels for consistency
axs[1].set_ylabel('')
axs[2].set_ylabel('')
axs[5].set_ylabel('')

# Custom legends
legend_labels = ['0', '1']  # Replace with your custom labels
legend_colors = ['red', 'lime']  # Replace with your custom colors

cn1 = df_meta['TYPE'].value_counts()
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8) for label, color in zip([f'Cancer (n={cn1[0]})',f'Benign (n={cn1[1]})'], legend_colors)]
axs[0].legend(handles=custom_legend, title='TYPE')

cn2 = train_labels_df['TYPE'].value_counts()
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8) for label, color in zip([f'Cancer (n={cn2[0]})',f'Benign (n={cn2[1]})'], legend_colors)]
axs[1].legend(handles=custom_legend, title='TYPE')

cn3 = test_labels_df['TYPE'].value_counts()
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8) for label, color in zip([f'Cancer (n={cn3[0]})',f'Benign (n={cn3[1]})'], legend_colors)]
axs[2].legend(handles=custom_legend, title='TYPE')

cn4 = train_labels_df_custom['TYPE'].value_counts()
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8) for label, color in zip([f'Cancer (n={cn4[0]})',f'Benign (n={cn4[1]})'], legend_colors)]
axs[4].legend(handles=custom_legend, title='TYPE')

cn5 = test_labels_df_custom['TYPE'].value_counts()
custom_legend = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8) for label, color in zip([f'Cancer (n={cn5[0]})',f'Benign (n={cn5[1]})'], legend_colors)]
axs[5].legend(handles=custom_legend, title='TYPE')

# plt.show()
plt.savefig('results/red_handed.png')