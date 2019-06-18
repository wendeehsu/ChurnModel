from sklearn.datasets import load_iris
from sklearn import cluster, datasets, metrics
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest
import matplotlib.pyplot as plt
import pandas as pd
import os
import struct
import pickle
import numpy as np

def search(row):
	if(row['MemberCardLevel_Medium'] == 1):
		return 1
	elif(row['MemberCardLevel_High'] == 1):
		return 2
	else:
		return 0

#read data
data=[]
with open('dataWithLable.pkl', 'rb') as f:
    data = pickle.load(f)

#merge 3 type of carde level
X=data
X["cardlevel"]=X.apply(search, axis=1)
X=data.drop(['is_Churn', 'UUID', 'OnlineMemberId', 'MemberCardLevel_Low', 'MemberCardLevel_Medium', 'MemberCardLevel_High'], axis=1)
y=data['is_Churn']

#最高的k個features
selector= SelectKBest(k=5)
selector.fit(X, y)
GetSupport= selector.get_support(True)
TransX= selector.transform(X)
Scores= selector.scores_
print(Scores)#看全部features分數,順序和column順序相同
print(GetSupport)#有被選到的features的index

#有被選到的features
for i in range(0, 27):
	if selector.get_support()[i]:
		print(X.columns.values.tolist()[i])