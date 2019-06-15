import pandas as pd
import numpy as np
import datetime as dt
import pickle
from sklearn.preprocessing import StandardScaler

def UntilNow(date):
	d1 = dt.datetime.strptime(date, '%Y/%m/%d')
	d2 = dt.datetime(2019, 4, 30)
	delta = d2 - d1
	return int(delta.days)
	
def TotalTrade(UUID):
	return len(user2Order[UUID])

def TradeFrequency(UUID):
	timelist = user2Order[UUID]['TradesDate'].tolist()
	for i,j in enumerate(timelist):
		timelist[i] = dt.datetime.strptime(j, '%Y/%m/%d')
	deltalist = []
	timelist = sorted(timelist, reverse = True)
	for i in range(len(timelist) - 1):
		deltalist.append(int((timelist[i] - timelist[i + 1]).days))
	return int(sum(deltalist) / len(deltalist))

def ChannelTypeFrequency_OfficialECom(UUID):
	ChannelTypelist = user2Order[UUID]['ChannelType'].tolist()
	return ChannelTypelist.count('OfficialECom')

def ChannelTypeFrequency_Pos(UUID):
	ChannelTypelist = user2Order[UUID]['ChannelType'].tolist()
	return ChannelTypelist.count('Pos')

def ChannelDetail_DesktopOfficialWeb(UUID):
	ChannelDetaillist = user2Order[UUID]['ChannelDetail'].tolist()
	return ChannelDetaillist.count('DesktopOfficialWeb')

def ChannelDetail_MoblieWeb(UUID):
	ChannelDetaillist = user2Order[UUID]['ChannelDetail'].tolist()
	return ChannelDetaillist.count('MobileWeb')

def ChannelDetail_iOSApp(UUID):
	ChannelDetaillist = user2Order[UUID]['ChannelDetail'].tolist()
	return ChannelDetaillist.count('iOSApp')

def ChannelDetail_AndroidApp(UUID):
	ChannelDetaillist = user2Order[UUID]['ChannelDetail'].tolist()
	return ChannelDetaillist.count('AndroidApp')

def PaymentType_Cash(UUID):
	PaymentTypelist = user2Order[UUID]['PaymentType'].tolist()
	return PaymentTypelist.count('Cash')

def PaymentType_Family(UUID):
	PaymentTypelist = user2Order[UUID]['PaymentType'].tolist()
	return PaymentTypelist.count('Family')

def PaymentType_SevenEleven(UUID):
	PaymentTypelist = user2Order[UUID]['PaymentType'].tolist()
	return PaymentTypelist.count('SevenEleven')

def PaymentType_CreditCardOnce(UUID):
	PaymentTypelist = user2Order[UUID]['PaymentType'].tolist()
	return PaymentTypelist.count('CreditCardOnce')

def PaymentType_LinePay(UUID):
	PaymentTypelist = user2Order[UUID]['PaymentType'].tolist()
	return PaymentTypelist.count('LinePay')

def PaymentType_ATM(UUID):
	PaymentTypelist = user2Order[UUID]['PaymentType'].tolist()
	return PaymentTypelist.count('ATM')

def ShippingType_Store(UUID):
	PaymentTypelist = user2Order[UUID]['ShippingType'].tolist()
	return PaymentTypelist.count('Store')

def ShippingType_Family(UUID):
	PaymentTypelist = user2Order[UUID]['ShippingType'].tolist()
	return PaymentTypelist.count('Family')

def ShippingType_SevenEleven(UUID):
	PaymentTypelist = user2Order[UUID]['ShippingType'].tolist()
	return PaymentTypelist.count('SevenEleven')

def ShippingType_Home(UUID):
	PaymentTypelist = user2Order[UUID]['ShippingType'].tolist()
	return PaymentTypelist.count('Home')

def ShippingType_LocationPickup(UUID):
	PaymentTypelist = user2Order[UUID]['ShippingType'].tolist()
	return PaymentTypelist.count('LocationPickup')

def ShippingType_SevenElevenPickup(UUID):
	PaymentTypelist = user2Order[UUID]['ShippingType'].tolist()
	return PaymentTypelist.count('SevenElevenPickup')

def ShippingType_FamilyPickup(UUID):
	PaymentTypelist = user2Order[UUID]['ShippingType'].tolist()
	return PaymentTypelist.count('FamilyPickup')

def TsCountAverage(UUID):
	TsCount = user2Order[UUID]['TsCount'].tolist()
	return sum(TsCount) / len(TsCount)

def QtyAverage(UUID):
	Qty = user2Order[UUID]['Qty'].tolist()
	return sum(Qty) / len(Qty)

def TotalSalesAmountAverage(UUID):
	TotalSalesAmount = user2Order[UUID]['TotalSalesAmount'].tolist()
	return sum(TotalSalesAmount) / len(TotalSalesAmount)

def TotalDiscountAverage(UUID):
	TotalDiscount = user2Order[UUID]['TotalDiscount'].tolist()
	return sum(TotalDiscount) / len(TotalDiscount)

# 讀入所有下單>=2的人的memberdata
targetMembers = {}
with open('targetMembers.pk1', 'rb') as f:
    targetMembers = pickle.load(f)
# 讀入所有下單>=2的人的orderdata
user2Order = {}
with open('User2Order.pkl', 'rb') as f:
    user2Order = pickle.load(f)
# 讀入target user list
users = []
with open('targetUUIDs.pkl', 'rb') as f:
    users = pickle.load(f)

df = pd.DataFrame.from_dict(targetMembers)
# 取出要處理的欄位
X = df.loc[:,['UUID', 'OnlineMemberId', 'RegisterDate', 'MemberCardLevel']]
X['UUID'] = X.index
# 把遺失值統一填入0
X = X.fillna(0)
# 處理MemberData
X['OnlineMemberId'] = X['OnlineMemberId'].apply(lambda x : 0 if x == 0 else 1)
X['RegisterDate'] = X['RegisterDate'].apply(UntilNow)
# MemberCardLevel有四個level必須要去掉1個避免dummy variable trap
# X['MemberCardLevel_NaN'] = X['MemberCardLevel'].apply(lambda x : 1 if x == 0 else 0)
X['MemberCardLevel_Low'] = X['MemberCardLevel'].apply(lambda x : 1 if x == 10 else 0)
X['MemberCardLevel_Medium'] = X['MemberCardLevel'].apply(lambda x : 1 if x == 20 else 0)
X['MemberCardLevel_High'] = X['MemberCardLevel'].apply(lambda x : 1 if x == 30 else 0)
X['TotalTrade'] = X['UUID'].apply(TotalTrade)
X['TradeFrequency'] = X['UUID'].apply(TradeFrequency)

X['ChannelTypeFrequency_OfficialECom'] = X['UUID'].apply(ChannelTypeFrequency_OfficialECom)
X['ChannelTypeFrequency_Pos'] = X['UUID'].apply(ChannelTypeFrequency_Pos)

X['ChannelDetail_DesktopOfficialWeb'] = X['UUID'].apply(ChannelDetail_DesktopOfficialWeb)
X['ChannelDetail_MoblieWeb'] = X['UUID'].apply(ChannelDetail_MoblieWeb)
X['ChannelDetail_iOSApp'] = X['UUID'].apply(ChannelDetail_iOSApp)
X['ChannelDetail_AndroidApp'] = X['UUID'].apply(ChannelDetail_AndroidApp)

X['PaymentType_Cash'] = X['UUID'].apply(PaymentType_Cash)
X['PaymentType_Family'] = X['UUID'].apply(PaymentType_Family)
X['PaymentType_SevenEleven'] = X['UUID'].apply(PaymentType_SevenEleven)
X['PaymentType_CreditCardOnce'] = X['UUID'].apply(PaymentType_CreditCardOnce)
X['PaymentType_LinePay'] = X['UUID'].apply(PaymentType_LinePay)
X['PaymentType_ATM'] = X['UUID'].apply(PaymentType_ATM)

X['ShippingType_Store'] = X['UUID'].apply(ShippingType_Store)
X['ShippingType_Family'] = X['UUID'].apply(ShippingType_Family)
X['ShippingType_SevenEleven'] = X['UUID'].apply(ShippingType_SevenEleven)
X['ShippingType_Home'] = X['UUID'].apply(ShippingType_Home)
X['ShippingType_LocationPickup'] = X['UUID'].apply(ShippingType_LocationPickup)
X['ShippingType_SevenElevenPickup'] = X['UUID'].apply(ShippingType_SevenElevenPickup)
X['ShippingType_FamilyPickup'] = X['UUID'].apply(ShippingType_FamilyPickup)

X['TsCountAverage'] = X['UUID'].apply(TsCountAverage)
X['QtyAverage'] = X['UUID'].apply(QtyAverage)
X['TotalSalesAmountAverage'] = X['UUID'].apply(TotalSalesAmountAverage)
X['TotalDiscountAverage'] = X['UUID'].apply(TotalDiscountAverage)

# pd.set_option('display.max_columns', 500) #最大列數
pd.set_option('display.width', 4000) #頁面宽度
X = X.drop(['UUID', 'MemberCardLevel'], axis=1)
print(X.head())

X.to_excel("FeatureMatrix.xlsx")
X.to_pickle("FeatureMatrix.pkl")