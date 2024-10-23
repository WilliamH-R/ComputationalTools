from IPython.display import display
from _01_load import load
from _02_clean import clean


### LOADING
print('\n'+"-"*100+'\n\n'+'LOADING'+'\n\n'+"-"*100+'\n')
df, df_meta = load()
display(df)
display(df_meta)


### CLEANING
print('\n'+"-"*100+'\n\n'+'CLEANING'+'\n\n'+"-"*100+'\n')
df_clean, bm_dict = clean()
display(df_clean)
display(bm_dict)

