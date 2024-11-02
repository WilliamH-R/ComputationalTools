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

def apriori(df, min_support=0.1, max_len=5):
    
    N = len(df)
    K = max_len

    # calculate support for singleton sets
    itemsets_k = {1:[]}
    supports_k = {1:[]}
    single_items = []
    for col in df.columns:
        support = df[col].sum() / N
        if support >= min_support:
            itemsets_k[1].append(set([col]))
            supports_k[1].append(support)
            single_items.append(col)

    # generate candidate pairs
    M = len(itemsets_k[1])
    itemsets_k[2] = []
    supports_k[2] = []
    for i in range(M):
        for j in range(i+1, M):
            support = (df[single_items[i]] & df[single_items[j]]).sum() / N
            if support >= min_support:
                itemsets_k[2].append(set([single_items[i], single_items[j]]))
                supports_k[2].append(support)
    
    for k in range(2,K):
        itemsets_k[k+1] = []
        supports_k[k+1] = []
        for itemset_k in itemsets_k[k]:
            for itemset_1 in itemsets_k[1]:
                new_set = itemset_k.union(itemset_1)
                if len(new_set) > k:                 
                    support = (df[[item for item in new_set]].sum(axis=1) == k+1).sum() / N
                    if support >= min_support:
                        itemsets_k[k+1].append(new_set)
                        supports_k[k+1].append(support)
        break




    itemsets, supports = [],[]
    for k in itemsets_k:
        itemsets.extend(itemsets_k[k])
        supports.extend(supports_k[k])
    
    # Create the dataframe
    df_fp = pd.DataFrame({'itemsets': itemsets, 'support': supports})
    return df_fp

df_fp = apriori(df, min_support=0.1, max_len=5)
df_fp


# %%
