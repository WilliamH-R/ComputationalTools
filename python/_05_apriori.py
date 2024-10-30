#%%
import pandas as pd
from mlxtend import frequent_patterns as fp
from mlxtend.preprocessing import TransactionEncoder
import numpy as np


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
# only show consequents that contain 'TYPE'
df_conf[df_conf['consequents'].apply(lambda x: 'TYPE' in x)].sort_values('conviction', ascending=False).head(30)
