# How to start?
```
import pickle

#  Get a target user list (下單次數大於等於2的人)
users = []
with open("targetUUIDs.pkl", 'rb') as f:
    users = pickle.load(f)

# Get (id to order) dictionary
user2Order = {}
with open('User2Order.pkl', 'rb') as f:
    user2Order = pickle.load(f)
```

# Files

`targetUUIDs.pkl` a list storing uuids of each target member <br/>
`personalPeriods.pkl`  a dictionary (key, value) = (uuid,個人平均購物週期) <br/>
`PersonalSd.pkl` a dictionary (key, value) = (uuid, 個人購物週期之標準差)<br/>
`Label.pkl` a dictionary (key, value) = (uuid, 是否流失的binary標籤)<br/>
`User2Order.pkl` a dictionary (key, value) = (uuid, pandas dataFrame storing OrderData)<br/>
`uuid2online.pkl` a dictionary (key, value) = (uuid, onlineId)<br/>
`targetMembers.pkl`   a pandas dataframe, derived from MemberData but store only data of target members (index is their uuid)<br/>
`sortedOrderData.pkl`   a pandas dataframe, derived from OrderData but store only data of target members (with a column "is_churn")<br/>
`dataWithLabel.pkl` a pandas dataframe, stores the featureMatrix (including behavior data) with the label `is_churn` as the last column<br/>
`FeatureMatrix.pkl` a pandas dataframe, stores the featureMatrix (including behavior data) with uuid as their row index (but no `is_churn` label)<br/>
`Part_uuid2Behavior.pkl` a test version dictionary (key, value) = (uuid, pandas dataFrame storing Behavior data)<br/>
`uuid2Behavior.pkl` a full version dictionary (key, value) = (uuid, pandas dataFrame storing Behavior data)<br/>

