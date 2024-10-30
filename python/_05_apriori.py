#%%
import pandas as pd
from mlxtend import frequent_patterns as fp
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

#%%
# Load and process data
df_clean = pd.read_csv('../data/cleaned.csv')
df_clean = df_clean.apply(lambda x: np.where(x > x.median(), 1, 0)).astype(bool)
df_meta  = pd.read_csv('../data/non_bm.csv')[['TYPE']].astype(bool)

# concatenate dataframes
df = pd.concat([df_clean, df_meta], axis=1)
df
#%%
# Apply apriori algorithm (get itemsets)
df_fp = fp.apriori(df, min_support=0.1, use_colnames=True)
df_fp

#%%
# Find itemsets of fixed length
df_fp['length'] = df_fp['itemsets'].apply(len)
df_fp_len2 = df_fp[df_fp['length'] == 2]
df_fp_len2.sort_values('support', ascending=False)

#%% 
# Find association rules (compute association metrics, given itemsets)
df_conf = fp.association_rules(df_fp, metric='confidence', min_threshold=0.001)
df_conf['interest'] = df_conf['confidence'] - df_conf['support']
df_conf.sort_values('interest', ascending=False)

#%% 
# only show consequents and antecedents that contain 'TYPE'
top_rules_cons = df_conf[df_conf['consequents'].apply(lambda x: 'TYPE' in x)].sort_values('interest', ascending=False)[:30]
top_rules_ante = df_conf[df_conf['antecedents'].apply(lambda x: 'TYPE' in x)].sort_values('interest', ascending=False)[:30]

top_rules = pd.concat([top_rules_cons, top_rules_ante]).drop_duplicates()
top_rules

#%%
G = nx.Graph()

# Add nodes and edges
unique_nodes = set()
for _, row in top_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            if antecedent == 'TYPE':
                color = 'red'
                unique_nodes.add(consequent)
            elif consequent == 'TYPE':
                color = 'lime'
                unique_nodes.add(antecedent)
            else:
                color = 'black'
            G.add_edge(antecedent, consequent, color=color)
            
          

# Define the layout with "TYPE" at the center
n_nodes = len(G.nodes) - 1  # Exclude "TYPE" from the circle count
angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
pos = {"TYPE": (0, 0)}  # Position "TYPE" at the center
pos.update({node: (np.cos(angle), np.sin(angle)) for node, angle in zip(G.nodes - {"TYPE"}, angles)})

# Draw the graph
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')

# add edge and increase size of arrow head
edge_colors = nx.get_edge_attributes(G, 'color').values()
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
plt.show()

#%%
list(unique_nodes)