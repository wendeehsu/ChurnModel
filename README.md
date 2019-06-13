# ChurnModel
Final project for BigData

## Spec
1. 先刪掉買低於兩次的人，只取 orderData裡購買次數高於兩次的UUID們 => 存進 `targetUUIDs.pkl`

2. 計算這些人的平均購買週期（先算個別的購買週期，再求出所有人的平均購買週期）=>求出的數字為”多久未回購為流失“的值
```
平均購物週期為109天，中位數是59天
```
3. 丟RNN => 結果請看 `BuildRNN.html`

### How to start?
```
import pickle
```
1. Get a target user list (下單次數大於等於2的人)
```
users = []
with open("targetUUIDs.pkl", 'rb') as f:
    users = pickle.load(f)
```

2. Get (id to order) dictionary
```
user2Order = {}
with open('User2Order.pkl', 'rb') as f:
    user2Order = pickle.load(f)
```
## Files
```
personalPeriods.pkl   # a dictionary (key, value) = (uuid,個人平均購物週期)
targetMembers.pk1   # a pandas dataframe, derived from MemberData but store only data of target members (index is their uuid)
sortedOrderData.pk1  # a pandas dataframe, derived from OrderData but store only data of target members (with a column "is_churn")
```
