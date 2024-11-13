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

# df_meta = ~df_meta  # Invert the values

# concatenate dataframes
df = pd.concat([df_clean, df_meta], axis=1)
df

#%%
# Apply apriori algorithm (get itemsets)
K = 5
df_fp = fp.apriori(df, min_support=0.1, use_colnames=True, max_len=K)
df_fp


#%% 

import pandas as pd
from itertools import combinations

def apriori(df, min_support=0.1, max_len=5):
    N = len(df)
    K = max_len

    # Precompute support for single items and filter based on min_support
    itemsets_k = {1: []}
    supports_k = {1: []}
    frequent_items = []
    for col in df.columns:
        support = df[col].sum() / N
        if support >= min_support:
            itemsets_k[1].append(frozenset([col]))
            supports_k[1].append(support)
            frequent_items.append(col)

    # Generate itemsets for lengths k > 1
    for k in range(2, K + 1):
        itemsets_k[k] = []
        supports_k[k] = []
        candidates = [i.union(j) for i in itemsets_k[k - 1] for j in itemsets_k[1] if len(i.union(j)) == k]
        candidates = set(candidates)  # Remove duplicate candidates

        for candidate in candidates:
            candidate_support = (df[list(candidate)].sum(axis=1) == k).sum() / N
            if candidate_support >= min_support:
                itemsets_k[k].append(candidate)
                supports_k[k].append(candidate_support)

        # Break if no further itemsets meet min_support
        if not itemsets_k[k]:
            break

    # Flatten the results into a DataFrame
    itemsets = [itemset for k in itemsets_k for itemset in itemsets_k[k]]
    supports = [support for k in supports_k for support in supports_k[k]]
    df_fp = pd.DataFrame({'itemsets': itemsets, 'support': supports})
    
    return df_fp

#%%

df_fp = apriori(df, min_support=0.1, max_len=5)
df_fp


# %%
