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


with open("pickle/sortedOrderData.pkl", 'rb') as f:
    OrderData = pickle.load(f)


# In[3]:


OrderData.head()


# In[4]:


targetMemberOrderData = OrderData[(-pd.isna(OrderData.sinceLastOrder)) & 
                                  (OrderData.TotalSalesAmount > 0)]


# In[5]:


targetMemberOrderData.head()


# In[7]:


targetMemberOrderData.shape


# In[8]:


with open("pickle/Labels.pkl", 'rb') as f:
    labels = pickle.load(f)


# In[10]:


targetMemberOrderData["is_churn"] = targetMemberOrderData["UUID"].map(labels)


# In[11]:


def FormateDate(d):
    return datetime.strptime(d,"%Y/%m/%d")

targetMemberOrderData["TradesDate"] = targetMemberOrderData["TradesDate"].apply(FormateDate)


# In[12]:


targetMemberOrderData = targetMemberOrderData.sort_values(by='TradesDate')
targetMemberOrderData.head(10)


# In[17]:


sinceLastOrder = targetMemberOrderData["sinceLastOrder"].tolist()
UUID = targetMemberOrderData["UUID"].tolist()
uuid2LastOrder = {}
for i,uuid in enumerate(UUID):
    uuid2LastOrder[uuid] = sinceLastOrder[i]


# In[19]:


len(uuid2LastOrder.keys())


# In[13]:


data_matrix = targetMemberOrderData.pivot_table(columns=['TradesDate'],index=['UUID'],aggfunc='size').fillna(0)
data_matrix.head()


# In[51]:


def GetSequence(rawSequence, lastDay):
    lastDay = int(lastDay)
    if lastDay <= 0:
        return [0] * 1225
    return [0] * (lastDay + (1225 - len(rawSequence))) + rawSequence[:-lastDay]


# In[52]:


sequenceDic = {}
for i in uuid2LastOrder:
    rawSequence = data_matrix.loc[i].tolist()
    sequenceDic[i] = GetSequence(rawSequence, uuid2LastOrder[i])


# In[54]:


with open("pickle/sequences.pkl", 'wb') as f:
    pickle.dump(sequenceDic, f)


# In[56]:


df = pd.DataFrame(sequenceDic)
df.head()


# In[57]:


df = df.transpose()
df.head()


# In[58]:


df["is_churn"] = df.index.to_series().map(labels)
df.head(10)


# In[59]:


df.is_churn.value_counts()


# In[60]:


OrderDate2Result = df
interval = 35*35
target = OrderDate2Result.values[:,-1]
data_matrix_x = OrderDate2Result.values[:,-(1+interval):-1].reshape(-1,35,35)


# In[61]:


print(data_matrix_x.shape)
print(target.shape)


# In[62]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data_matrix_x,target ,test_size=0.33, stratify=target)


# In[63]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_dim,dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_dim,
                            num_layers=2,
                            dropout=dropout,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 2)
    def forward(self, seq_in):
        lstm_out, (h_n,h_c) = self.lstm(seq_in,None)
        ht = lstm_out[:, -1, :]  # just want last time step hidden states
        out = self.fc1(ht)       # convert to fully connected
        return out


# In[64]:


lstm = LSTM(35,32).double()
print(lstm) 


# In[65]:


optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

torch.manual_seed(42)    # reproducible
BATCH_SIZE = 200 
torch_dataset = Data.TensorDataset(
    Variable(torch.DoubleTensor(X_train)),Variable(torch.LongTensor(y_train)))
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打亂數據
    num_workers=2,              # 多線程
)


# In[66]:


for epoch in range(6):   
    for step, (batch_x,batch_y) in enumerate(loader):
        output = lstm.forward(batch_x.view(-1,35,35))
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch: ', epoch, '| Loss: ', loss.data.item())


# In[67]:


from sklearn.metrics import accuracy_score, classification_report
test_output = lstm(Variable(torch.DoubleTensor(X_test)))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print("accuracy_score = ", accuracy_score(y_test,pred_y))
print(classification_report(y_test, pred_y))


# In[68]:


#------------------------------------train--------------------------------
train_output = lstm(Variable(torch.DoubleTensor(X_train)))
pred_y_train = torch.max(train_output, 1)[1].data.numpy().squeeze()
print("accuracy_score = ",accuracy_score(y_train, pred_y_train))
print(classification_report(y_train, pred_y_train))


# In[71]:


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


# In[72]:


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

