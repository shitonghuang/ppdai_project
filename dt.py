# encoding: utf-8
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

df = pd.read_csv('df_lc_state1.csv')
label = np.array(df['state'].values.tolist())
fea = np.array(df[['初始评级']].values.tolist())


# 3-Fold Validation
kf = KFold(n_splits=10, shuffle=True)
acc = np.zeros([10, 1])
idx = 0
for train_idx, test_idx in kf.split(fea, label):
    fea_train, label_train = fea[train_idx], label[train_idx]
    fea_test, label_test = fea[test_idx], label[test_idx]
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(fea_train, label_train)
    pred = clf.predict(fea_test)

    acc[idx] = accuracy_score(label_test, pred)
    print('Fold %d Accuracy: %.4f' % (idx+1, acc[idx]))
    idx += 1
    
print('Mean Accuracy: %.4f' % np.mean(acc))
