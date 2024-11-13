
#%%
import numpy as np
import pandas as pd

df = pd.read_csv('../../data/cleaned.csv')
df_meta = pd.read_csv('../../data/non_bm.csv')[['TYPE']]

df_train = df.iloc[:235]
df_test  = df.iloc[235:]

df_meta_train = df_meta.iloc[:235]
df_meta_test  = df_meta.iloc[235:]

# Perform SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
df_train, df_meta_train = sm.fit_resample(df_train, df_meta_train)
df_test, df_meta_test = sm.fit_resample(df_test, df_meta_test)

# attributes = ['HE4', 'CEA']
attributes = ['UA','CEA','AST','GLO','BUN','HE4']
attributes = ['HE4','LYM%','Ca','PDW','GLO']

# %%
# classification tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=5) 
# clf = RandomForestClassifier(n_estimators=1000, max_depth=10, class_weight='balanced')

# Train the classifier
clf.fit(df_train[attributes], df_meta_train['TYPE'])

# Performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on test data
pred = clf.predict(df_test[attributes])
print("Accuracy:",accuracy_score(df_meta_test['TYPE'], pred))

# confusion matrix
cm = confusion_matrix(df_meta_test['TYPE'], pred)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()




# %%
