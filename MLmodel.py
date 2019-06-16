import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataWithLable = {}
with open('dataWithLable.pkl', 'rb') as f:
    dataWithLable = pickle.load(f)

df = pd.DataFrame.from_dict(dataWithLable)

X = df.drop(['UUID', 'is_Churn'], axis=1)
y = df.loc[:,['is_Churn']]
X = X.to_numpy()
y = y.to_numpy()

# 切訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 標準化
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

# 訓練
from sklearn import tree 
clf_tree = tree.DecisionTreeClassifier() 
clf_tree = clf_tree.fit(X_train, y_train.ravel())

from sklearn.svm import SVC
clf_svc = SVC()
clf_svc = clf_svc.fit(X_train, y_train.ravel())

# 預測
y_pred_tree = clf_tree.predict(X_test)
y_pred_svc = clf_svc.predict(X_test)


# 結果
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
 
print('DecisionTree:', accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
# print(pd.DataFrame(confusion_matrix(y_test, y_pred_tree)))

print('')
print('SVM:', accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))
# print(pd.DataFrame(confusion_matrix(y_test, y_pred_svc)))
