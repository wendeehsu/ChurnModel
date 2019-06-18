# ChurnModel
Final project for BigData

## Spec
1. 先刪掉買低於兩次的人，只取 orderData裡購買次數高於兩次的UUID們 => 存進 `targetUUIDs.pkl`

2. 計算這些人的平均購買週期（先算個別的購買週期，再求出所有人的平均購買週期）=>求出的數字為”多久未回購為流失“的值
```
平均購物週期為109天，中位數是59天
```
3. 丟RNN => 結果請看 `BuildRNN.html`

### Files
請參照[pickle/README.md](pickle/README.md)
