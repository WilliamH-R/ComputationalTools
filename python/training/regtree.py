
#%%
import numpy as np
import pandas as pd

df = pd.read_csv('../../data/cleaned.csv')
df_meta = pd.read_csv('../../data/non_bm.csv')[['TYPE']]

df_train = df.iloc[:235]
df_test  = df.iloc[235:]

df_meta_train = df_meta.iloc[:235]
df_meta_test  = df_meta.iloc[235:]

# attributes = ['HE4', 'CEA']
attributes = ['LYM#','LYM%','HGB','RBC','DBIL']

# %%
# classification tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=2)

# Train the classifier
clf.fit(df_train[attributes], df_meta_train['TYPE'])

#&&
# Performance
from sklearn.metrics import accuracy_score

# Predict on test data
pred = clf.predict(df_test[attributes])
accuracy_score(df_meta_test['TYPE'], pred)

# # Confusion matrix
# from sklearn.metrics import confusion_matrix

# confusion_matrix(df_meta_test['TYPE'], pred)

# %%
# Plot the decision tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=df_train.columns, class_names=['Non-Benign', 'Benign'])
plt.show()
# %%
