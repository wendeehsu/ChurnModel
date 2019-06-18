# ChurnModel
Final project for BigData

## Spec
1. 先刪掉買低於兩次的人，只取 orderData裡購買次數高於兩次的UUID們 => 存進 `targetUUIDs.pkl`
2. 計算這些人的個別平均購買週期和標準差
3. 定義流失：假設購物週期為常態分佈，則顧客最後一筆下單的時間距離 2019/5/1 若大於這個人自己的 `平均購物週期+兩倍的標準差` 則為流失
4. 用兩種方式（RNN、FeatureMatrix）來做流失預測模型

### Files
請參照[pickle/README.md](pickle/README.md)

## Method 1 : RNN
related files: (python) `RNN.py`, (result) `RNN.html`

* 一個會員所有的資料 (2016/1/1-2019/5/1)，最後一筆到2019/5/1用來做流失標記
, 最後一筆之前全部拿來做rnn訓練

1. 將目標客戶在 OrderData 裡的記錄轉成以時間為標記的數字矩陣（數字表示當日購物次數）
```
data_matrix = targetMemberOrderData.pivot_table(columns=['TradesDate'],index=['UUID'],aggfunc='size').fillna(0)
```
2. 將最後一筆交易距離2019/5/1的時間記錄下來，若此 `時間間隔 > （這個人的平均購物週期+兩倍的標準差）` 則為流失
3. 將最後一筆交易前的資料補齊成長為1225的sequence （1225是為了之後方便製造35x35的方陣）當成訓練資料
4. 建立RNN模型

* Input size = 35
* Hidden units = 32
* Hidden layers = 2
* Dropout = 0.3 (避免 overfitting)
* Output feature = 2 (因為 is_churn 是 binary 標籤)

```
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_dim,dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_dim,
                            num_layers=2,dropout=dropout,batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 2)
    def forward(self, seq_in):
        lstm_out, (h_n,h_c) = self.lstm(seq_in,None)
        ht = lstm_out[:, -1, :]
        #convert to fully connected
        out = self.fc1(ht)
        return out
"""
LSTM(
  (lstm): LSTM(35, 32, num_layers=2, batch_first=True, dropout=0.3)
  (fc1): Linear(in_features=32, out_features=2, bias=True)
)
"""
```
並選用`CrossEntropy` loss function 和 `Adam` Optimizer
 
5. 結果 <br/>
LSTM 若為一層：accuracy score = 0.76865 <br/>
LSTM 若為兩層：accuracy score = 0.80945 <br/>
LSTM 若為三層：accuracy score = 0.81150 <br/>

# Method 2 : Feature Engineering

