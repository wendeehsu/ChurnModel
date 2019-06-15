#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xlsxwriter
import math
import pickle
from datetime import datetime, timedelta
from openpyxl import load_workbook


# In[2]:


with open("sortedOrderData.pk1", 'rb') as f:
    OrderData = pickle.load(f)


# In[3]:


OrderData.head()


# In[4]:


personalPeriods = {}
with open("personalPeriods.pkl", 'rb') as f:
    personalPeriods = pickle.load(f)


# In[5]:


personalPeriods


# In[6]:


targetMemberOrderData = OrderData[(-pd.isna(OrderData.sinceLastOrder)) & 
                                  (OrderData.TotalSalesAmount > 0)]


# In[7]:


targetMemberOrderData.head(10)


# In[27]:


def FormateDate(d):
    return datetime.strptime(d,"%Y/%m/%d")


# In[28]:


targetMemberOrderData["TradesDate"] = targetMemberOrderData["TradesDate"].apply(FormateDate)


# In[31]:


targetMemberOrderData = targetMemberOrderData.sort_values(by='TradesDate')
targetMemberOrderData.head(10)


# In[32]:


data_matrix = targetMemberOrderData.pivot_table(columns=['TradesDate'],index=['UUID'],aggfunc='size').fillna(0)
data_matrix.head()


# In[43]:


dates = list(data_matrix.columns.values.astype('datetime64[D]'))
dates


# In[50]:


def Distance(d):
    return (np.datetime64('2019-05-01') - d )/ np.timedelta64(1, 'D')


# In[51]:


distances = list(map(Distance, dates))
distances # sorted


# In[53]:


personalPeriods = {}
with open("personalPeriods.pkl", 'rb') as f:
    personalPeriods = pickle.load(f)


# In[65]:


a = [0,1,2,3,4,5]
k = 2
[0] * (7 - (k+1)) + a[:k+1]


# In[66]:


def GetSequence(rawSequence, personalPeriod):
    personalSequece = []
    predictPeriod = personalPeriod *1.5
    if predictPeriod >= 1217:
         personalSequece = [0] *1225
    else:
        index = 0
        for (i, distance) in enumerate(distances):
            if distance > predictPeriod:
                index = i
            else:
                break
        personalSequece = [0] * (1225 - (index+1)) + rawSequence[:index+1]
        
    return personalSequece


# In[69]:


sequenceDic = {}
for i in personalPeriods:
    rawSequence = data_matrix.loc[i].tolist()
    sequenceDic[i] = GetSequence(rawSequence, personalPeriods[i])


# In[70]:


sequenceDic


# In[71]:


with open("sequences.pkl", 'wb') as f:
    pickle.dump(sequenceDic, f)


# In[72]:


df = pd.DataFrame(sequenceDic)
df


# In[73]:


df = df.transpose()


# In[74]:


df.head()


# In[75]:


labelDict = {}
with open("labels.pkl", 'rb') as f:
    labelDict = pickle.load(f)


# In[77]:


df["is_churn"] = df.index.to_series().map(labelDict)
df.head(10)


# In[93]:


df.is_churn.value_counts()


# In[81]:


OrderDate2Result = df
interval = 35*35
target = OrderDate2Result.values[:,-1]
data_matrix_x = OrderDate2Result.values[:,-(1+interval):-1].reshape(-1,35,35)


# In[82]:


print(data_matrix_x.shape)
print(target.shape)


# In[83]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data_matrix_x,target ,test_size=0.33, stratify=target)


# In[84]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data


# In[85]:


class LSTM(nn.Module):
    def __init__(self,input_size,hidden_dim,dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_dim,num_layers=2,dropout=dropout,batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 2)
    def forward(self, seq_in):
        lstm_out, (h_n,h_c) = self.lstm(seq_in,None)
        ht = lstm_out[:, -1, :]
        #convert to fully connected
        out = self.fc1(ht)
        return out


# In[86]:


lstm = LSTM(35,32).double()
print(lstm) 


# In[87]:


optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

torch.manual_seed(42)    # reproducible
BATCH_SIZE = 200 
torch_dataset = Data.TensorDataset(Variable(torch.DoubleTensor(X_train)),Variable(torch.LongTensor(y_train)))
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打亂數據
    num_workers=2,              # 多線程
)


# In[88]:


for epoch in range(6):   
    for step, (batch_x,batch_y) in enumerate(loader):
        output = lstm.forward(batch_x.view(-1,35,35))
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch: ', epoch, '| Loss: ', loss.data.item())


# In[89]:


from sklearn.metrics import accuracy_score, classification_report
test_output = lstm(Variable(torch.DoubleTensor(X_test)))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print("accuracy_score = ", accuracy_score(y_test,pred_y))
print(classification_report(y_test, pred_y))


# In[90]:


#------------------------------------train--------------------------------
train_output = lstm(Variable(torch.DoubleTensor(X_train)))
pred_y_train = torch.max(train_output, 1)[1].data.numpy().squeeze()
print("accuracy_score = ",accuracy_score(y_train, pred_y_train))
print(classification_report(y_train, pred_y_train))


# In[91]:


import matplotlib.pyplot as plt
import itertools
print(__doc__)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[92]:


from sklearn.metrics import confusion_matrix
class_names = np.array(['no_churn','is_churn'])
cnf_matrix = confusion_matrix(y_test, pred_y)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

