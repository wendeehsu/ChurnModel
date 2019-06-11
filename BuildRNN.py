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

targetUserStorageFile = 'targetUUIDs.pkl'
targetUsers = []
with open(targetUserStorageFile, 'rb') as f:
    targetUsers = pickle.load(f)


# In[2]:


User2OrderStorageFile = 'User2Order.pkl'
User2Order = {}
with open(User2OrderStorageFile, 'rb') as f:
    User2Order = pickle.load(f)


# In[3]:


OrderData = pd.read_csv("OrderDataForNTU.txt", sep = ',', encoding = 'utf8')
OrderData.info()


# In[4]:


def FormateTime(date):
    return datetime.strptime(date, '%Y/%m/%d')


# In[5]:


sinceLastOrder = {}
for user in targetUsers:
    tradesDate = User2Order[user]["TradesDate"].tolist()
    tradesDate = list(map(FormateTime,tradesDate))
    sinceLastOrder[user] = datetime.now() - max(tradesDate)
    sinceLastOrder[user] = sinceLastOrder[user].days


# In[6]:


sinceLastOrder


# In[7]:


OrderData = pd.read_csv("OrderDataForNTU.txt", sep = ',', encoding = 'utf8')


# In[8]:


OrderData["sinceLastOrder"] = OrderData["UUID"].map(sinceLastOrder)
OrderData.head(30)


# In[9]:


meanOrderPeriod = 59
OrderData['is_churn'] = OrderData.sinceLastOrder.apply(
    lambda x: 1 if x > meanOrderPeriod or pd.isna(x) else 0)


# In[10]:


targetMemberOrderData = OrderData[(-pd.isna(OrderData.sinceLastOrder)) & 
                                  (OrderData.TotalSalesAmount > 0)]


# In[11]:


data_matrix = targetMemberOrderData.pivot_table(columns=['TradesDate'],index=['UUID'],aggfunc='size').fillna(0)
data_matrix.head()


# In[12]:


target = targetMemberOrderData.groupby(by=['UUID']).mean()
target.head()


# In[13]:


target.is_churn.value_counts()


# In[14]:


OrderDate2Result = pd.merge(data_matrix,target[['is_churn']],
                            left_index=True, right_index=True)


# In[15]:


OrderDate2Result.head()


# In[48]:


import numpy as np
interval = 16*16
target = OrderDate2Result.values[:,-1]
data_matrix_x = OrderDate2Result.values[:,-(60+interval):-60].reshape(-1,16,16)


# In[49]:


print(data_matrix_x.shape)
print(target.shape)


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data_matrix_x,target ,test_size=0.33, stratify=target)


# In[51]:


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data


# In[52]:


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


# In[53]:


lstm = LSTM(16,32).double()
print(lstm) 


# In[54]:


optimizer = torch.optim.Adam(lstm.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()


# In[55]:


torch.manual_seed(42)    # reproducible
BATCH_SIZE = 200 
torch_dataset = Data.TensorDataset(Variable(torch.DoubleTensor(X_train)),Variable(torch.LongTensor(y_train)))
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打亂數據
    num_workers=2,              # 多線程
)


# In[56]:


for epoch in range(6):   
    for step, (batch_x,batch_y) in enumerate(loader):
        output = lstm.forward(batch_x.view(-1,16,16))
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch: ', epoch, '| Loss: ', loss.data.item())


# In[57]:


test_output = lstm(Variable(torch.DoubleTensor(X_test)))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()


# In[58]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred_y)


# In[59]:


from sklearn.metrics import classification_report
print(classification_report(y_test, pred_y))


# In[60]:


#------------------------------------train--------------------------------
train_output = lstm(Variable(torch.DoubleTensor(X_train)))
pred_y_train = torch.max(train_output, 1)[1].data.numpy().squeeze()


# In[61]:


print(classification_report(y_train, pred_y_train))


# In[62]:


accuracy_score(y_train, pred_y_train)

