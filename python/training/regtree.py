
#%%
import numpy as np
import pandas as pd

df = pd.read_csv('../../data/cleaned.csv')

df_train = df.iloc[:235]
df_test  = df.iloc[235:]

# %%
