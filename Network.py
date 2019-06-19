#!/usr/bin/env python
# coding: utf-8

# In[29]:


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
                            num_layers=3,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, seq_in):
        lstm_out, (h_n,h_c) = self.lstm(seq_in,None)
        ht = lstm_out[:, -1, :]  # just want last time step hidden states
        out = self.fc(ht)       # convert to fully connected
        return out


# In[30]:


lstm = LSTM(35,32).double()
print(lstm) 


# In[31]:


print(len(list(lstm.parameters())))
for i in range(len(list(lstm.parameters()))):
    print(list(lstm.parameters())[i].size())


# In[11]:


#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import pickle


# In[14]:


sequenceDic = {}
with open("pickle/sequences.pkl", 'rb') as f:
    sequenceDic = pickle.load(f)


# In[15]:


df = pd.DataFrame(sequenceDic)
df.head()


# In[17]:


with open("pickle/Label.pkl", 'rb') as f:
    labels = pickle.load(f)
df = df.transpose()
df["is_churn"] = df.index.to_series().map(labels)
df.head(10)


# In[18]:


df.is_churn.value_counts()


# In[35]:


with open("pickle/rnnInput.pkl", 'wb') as f:
    pickle.dump(df,f)


# In[19]:


OrderDate2Result = df
interval = 35*35
target = OrderDate2Result.values[:,-1]
data_matrix_x = OrderDate2Result.values[:,-(1+interval):-1].reshape(-1,35,35)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    data_matrix_x,target ,test_size=0.33, stratify=target)


# In[32]:


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


# In[33]:


for epoch in range(6):   
    for step, (batch_x,batch_y) in enumerate(loader):
        output = lstm.forward(batch_x.view(-1,35,35))
        loss = loss_func(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch: ', epoch, '| Loss: ', loss.data.item())


# In[34]:


from sklearn.metrics import accuracy_score, classification_report
test_output = lstm(Variable(torch.DoubleTensor(X_test)))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print("accuracy_score = ", accuracy_score(y_test,pred_y))
print(classification_report(y_test, pred_y))


# In[36]:


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


# In[37]:


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

