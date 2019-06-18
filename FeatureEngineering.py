import pandas as pd
import numpy as np
import datetime as dt
import pickle
from sklearn.preprocessing import StandardScaler
import math
import statsmodels.api as sm

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

def OperationSystem_iOS(UUID):
	if UUID in keys:
		UUID_behavior = uuid2Behavior[UUID]		
		UUID_behavior['HitDateTime'] = UUID_behavior['HitDateTime'].apply(lambda x : dt.datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S'))
		UUID_behavior = UUID_behavior.sort_values(by = ['HitDateTime'])
		
		HitDateTime = UUID_behavior['HitDateTime'].tolist()
		TrafficSourceCategory = UUID_behavior['TrafficSourceCategory'].tolist()
		SessionNumber = UUID_behavior['SessionNumber'].tolist()
		OperationSystem = UUID_behavior['OperationSystem'].tolist()
		
		diff1 = dt.timedelta(days = 1)
		diff2 = dt.timedelta(minutes = 30)

		os = []
		
		for i in range(len(HitDateTime) - 1):
			same_session = False
			combine = False
			if HitDateTime[i].date() == HitDateTime[i + 1].date():
				if TrafficSourceCategory[i] == TrafficSourceCategory[i + 1]:
					if SessionNumber[i] == SessionNumber[i + 1]:
						same_session = True
			elif (HitDateTime[i] + diff1).date() == HitDateTime[i + 1].date():
				if TrafficSourceCategory[i] == TrafficSourceCategory[i + 1]:
					if (HitDateTime[i] + diff2) >= HitDateTime[i + 1]:
						same_session = True
						combine = True
			if same_session == False:
				if combine == False:
					os.append(OperationSystem[i])
		
		iOS = 0.0
		Android = 0.0
		
		for i, j in enumerate(os):
			if j == 'Android':
				Android += math.log(i + 1)
			elif j == 'iOS':
				iOS += math.log(i + 1)
		return iOS
	else:
		return 0.0

def OperationSystem_Android(UUID):
	if UUID in keys:
		UUID_behavior = uuid2Behavior[UUID]
		
		# UUID_behavior['HitDateTime'] = UUID_behavior['HitDateTime'].apply(lambda x : dt.datetime.strptime(x[:19], '%Y-%m-%d %H:%M:%S'))
		UUID_behavior = UUID_behavior.sort_values(by = ['HitDateTime'])
		
		HitDateTime = UUID_behavior['HitDateTime'].tolist()
		TrafficSourceCategory = UUID_behavior['TrafficSourceCategory'].tolist()
		SessionNumber = UUID_behavior['SessionNumber'].tolist()
		OperationSystem = UUID_behavior['OperationSystem'].tolist()
		
		diff1 = dt.timedelta(days = 1)
		diff2 = dt.timedelta(minutes = 30)

		os = []
		
		for i in range(len(HitDateTime) - 1):
			same_session = False
			combine = False
			if HitDateTime[i].date() == HitDateTime[i + 1].date():
				if TrafficSourceCategory[i] == TrafficSourceCategory[i + 1]:
					if SessionNumber[i] == SessionNumber[i + 1]:
						same_session = True
			elif (HitDateTime[i] + diff1).date() == HitDateTime[i + 1].date():
				if TrafficSourceCategory[i] == TrafficSourceCategory[i + 1]:
					if (HitDateTime[i] + diff2) >= HitDateTime[i + 1]:
						same_session = True
						combine = True
			if same_session == False:
				if combine == False:
					os.append(OperationSystem[i])
		
		iOS = 0.0
		Android = 0.0
		
		for i, j in enumerate(os):
			if j == 'Android':
				Android += math.log(i + 1)
			elif j == 'iOS':
				iOS += math.log(i + 1)
		return Android
	else:
		return 0.0

def SessionTrend(UUID):
	if UUID in keys:
		UUID_behavior = uuid2Behavior[UUID]		
		UUID_behavior = UUID_behavior.sort_values(by = ['HitDateTime'])
		
		HitDateTime = UUID_behavior['HitDateTime'].tolist()
		TrafficSourceCategory = UUID_behavior['TrafficSourceCategory'].tolist()
		SessionNumber = UUID_behavior['SessionNumber'].tolist()
		OperationSystem = UUID_behavior['OperationSystem'].tolist()
		
		diff1 = dt.timedelta(days = 1)
		diff2 = dt.timedelta(minutes = 30)

		sessions_start = [HitDateTime[0]]
		sessions_end = []
		if len(HitDateTime) > 1:
			for i in range(len(HitDateTime) - 1):
				same_session = False
				combine = False
				if HitDateTime[i].date() == HitDateTime[i + 1].date():
					if TrafficSourceCategory[i] == TrafficSourceCategory[i + 1]:
						if SessionNumber[i] == SessionNumber[i + 1]:
							same_session = True
				elif (HitDateTime[i] + diff1).date() == HitDateTime[i + 1].date():
					if TrafficSourceCategory[i] == TrafficSourceCategory[i + 1]:
						if (HitDateTime[i] + diff2) >= HitDateTime[i + 1]:
							same_session = True
							combine = True
				
				if same_session == False:
					if combine == False:
						sessions_start.append(HitDateTime[i + 1])
						sessions_end.append(HitDateTime[i])
			sessions_start = sessions_start[ : len(sessions_start) - 1]
		else:
			return 0
			
		sessions_start = np.array(sessions_start)
		sessions_end = np.array(sessions_end)
		sessions_time = sessions_end - sessions_start
		sessions_time = sessions_time.tolist()
		
		X = []
		
		for i in range(len(sessions_time)):
			sessions_time[i] = sessions_time[i].seconds
			X.append(i + 1)
		
		print(UUID)
		
		if X == [] or sessions_time == []:
			return 0
		
		X = sm.add_constant(X)
		model = sm.OLS(sessions_time, X)
		results = model.fit()
		
		if len(results.params) == 1:
			return results.params[0]
		return results.params[1]
	else:
		return 0


# 讀入所有下單>=2的人的memberdata
targetMembers = {}
with open('pickle/targetMembers.pkl', 'rb') as f:
    targetMembers = pickle.load(f)
# 讀入所有下單>=2的人的orderdata
user2Order = {}
with open('pickle/User2Order.pkl', 'rb') as f:
    user2Order = pickle.load(f)
# 讀入target user list
users = []
with open('pickle/targetUUIDs.pkl', 'rb') as f:
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

print('接下來進入behavior data ...')

# 讀入behavior data
uuid2Behavior = {}
with open('pickle/uuid2Behavior.pkl', 'rb') as f:
    uuid2Behavior = pickle.load(f)
keys = list(uuid2Behavior.keys())

# 處理behavior data
# Operating System
X['OperationSystem_iOS'] = X['UUID'].apply(OperationSystem_iOS)
X['OperationSystem_Android'] = X['UUID'].apply(OperationSystem_Android)
# 瀏覽時間趨勢
X['SessionTrend'] = X['UUID'].apply(SessionTrend)
# 各項行為趨勢
def BehaviorTypeTrend(keys):
	
	behaviordict = dict()
	session_num_dict = dict()
	
	for j in keys: # loop每一個有行為的人
		
		UUID_behavior = uuid2Behavior[j]
		UUID_behavior = UUID_behavior.sort_values(by = ['HitDateTime'])

		HitDateTime = UUID_behavior['HitDateTime'].tolist()
		TrafficSourceCategory = UUID_behavior['TrafficSourceCategory'].tolist()
		SessionNumber = UUID_behavior['SessionNumber'].tolist()
		OperationSystem = UUID_behavior['OperationSystem'].tolist()
		BehaviorType = UUID_behavior['BehaviorType'].tolist()
		
		diff1 = dt.timedelta(days = 1)
		diff2 = dt.timedelta(minutes = 30)		
		
		behaviorlist = []
		session_num_list = []
		
		if len(HitDateTime) > 1:
			for i in range(len(HitDateTime) - 1):
				same_session = False
				combine = False
				if HitDateTime[i].date() == HitDateTime[i + 1].date():
					if TrafficSourceCategory[i] == TrafficSourceCategory[i + 1]:
						if SessionNumber[i] == SessionNumber[i + 1]:
							same_session = True
				elif (HitDateTime[i] + diff1).date() == HitDateTime[i + 1].date():
					if TrafficSourceCategory[i] == TrafficSourceCategory[i + 1]:
						if (HitDateTime[i] + diff2) >= HitDateTime[i + 1]:
							same_session = True
							combine = True
				if same_session == False:
					if combine == False:
						behaviorlist.append(BehaviorType[i])
						session_num_list.append(SessionNumber[i])
			behaviordict[j] = [behaviorlist, session_num_list]
		else:
			session_num_list.append(SessionNumber[0])
			behaviorlist.append(BehaviorType[0])
			behaviordict[j] = [behaviorlist,session_num_list]
	return behaviordict

def BehaviorTypeTrend_Purchase(UUID):
	if UUID in keys:		
		UUID_behavior_typelist, UUID_behavior_session_num_list = behavedict[UUID]
		X = []
		y = []
		
		for i, j in enumerate(UUID_behavior_typelist):
			if j == 'Purchase':
				X.append(i + 1)
				y.append(UUID_behavior_session_num_list[i])
		
		if X == [] or y == []:
			return 0
		
		X = sm.add_constant(X)
		model = sm.OLS(y, X)
		results = model.fit()
		print('Purchase', UUID)
		if len(results.params) == 1:
			return results.params[0]
		return results.params[1]
	else:
		return 0

def BehaviorTypeTrend_Fav(UUID):
	if UUID in keys:		
		UUID_behavior_typelist, UUID_behavior_session_num_list = behavedict[UUID]
		X = []
		y = []
		
		for i, j in enumerate(UUID_behavior_typelist):
			if j == 'Fav':
				X.append(i + 1)
				y.append(UUID_behavior_session_num_list[i])
		
		if X == [] or y == []:
			return 0
		
		X = sm.add_constant(X)
		model = sm.OLS(y, X)
		results = model.fit()
		print('Fav', UUID)
		if len(results.params) == 1:
			return results.params[0]
		return results.params[1]
	else:
		return 0

def BehaviorTypeTrend_Search(UUID):
	if UUID in keys:		
		UUID_behavior_typelist, UUID_behavior_session_num_list = behavedict[UUID]
		X = []
		y = []
		
		for i, j in enumerate(UUID_behavior_typelist):
			if j == 'Search':
				X.append(i + 1)
				y.append(UUID_behavior_session_num_list[i])
		
		if X == [] or y == []:
			return 0
		
		X = sm.add_constant(X)
		model = sm.OLS(y, X)
		results = model.fit()
		print('Search', UUID)
		if len(results.params) == 1:
			return results.params[0]
		return results.params[1]
	else:
		return 0
		
def BehaviorTypeTrend_ViewSalePageCategory(UUID):
	if UUID in keys:		
		UUID_behavior_typelist, UUID_behavior_session_num_list = behavedict[UUID]
		X = []
		y = []
		
		for i, j in enumerate(UUID_behavior_typelist):
			if j == 'ViewSalePageCategory':
				X.append(i + 1)
				y.append(UUID_behavior_session_num_list[i])
		
		if X == [] or y == []:
			return 0
		
		X = sm.add_constant(X)
		model = sm.OLS(y, X)
		results = model.fit()
		print('ViewSalePageCategory', UUID)
		if len(results.params) == 1:
			return results.params[0]
		return results.params[1]
	else:
		return 0

def BehaviorTypeTrend_Cart(UUID):
	if UUID in keys:		
		UUID_behavior_typelist, UUID_behavior_session_num_list = behavedict[UUID]
		X = []
		y = []
		
		for i, j in enumerate(UUID_behavior_typelist):
			if j == 'Cart':
				X.append(i + 1)
				y.append(UUID_behavior_session_num_list[i])
		
		if X == [] or y == []:
			return 0
		
		X = sm.add_constant(X)
		model = sm.OLS(y, X)
		results = model.fit()
		print('Cart', UUID)
		if len(results.params) == 1:
			return results.params[0]
		return results.params[1]
	else:
		return 0

def BehaviorTypeTrend_ViewSalePage(UUID):
	if UUID in keys:		
		UUID_behavior_typelist, UUID_behavior_session_num_list = behavedict[UUID]
		X = []
		y = []
		
		for i, j in enumerate(UUID_behavior_typelist):
			if j == 'ViewSalePage':
				X.append(i + 1)
				y.append(UUID_behavior_session_num_list[i])
		
		if X == [] or y == []:
			return 0
		
		X = sm.add_constant(X)
		model = sm.OLS(y, X)
		results = model.fit()
		print('ViewSalePage', UUID)
		if len(results.params) == 1:
			return results.params[0]
		return results.params[1]
	else:
		return 0

behavedict = BehaviorTypeTrend(keys)
X['BehaviorTypeTrend_Purchase'] = X['UUID'].apply(BehaviorTypeTrend_Purchase)
X['BehaviorTypeTrend_Fav'] = X['UUID'].apply(BehaviorTypeTrend_Fav)
X['BehaviorTypeTrend_Search'] = X['UUID'].apply(BehaviorTypeTrend_Search)
X['BehaviorTypeTrend_ViewSalePageCategory'] = X['UUID'].apply(BehaviorTypeTrend_ViewSalePageCategory)
X['BehaviorTypeTrend_Cart'] = X['UUID'].apply(BehaviorTypeTrend_Cart)
X['BehaviorTypeTrend_ViewSalePage'] = X['UUID'].apply(BehaviorTypeTrend_ViewSalePage)

#pd.set_option('display.max_columns', 500) #最大列數
pd.set_option('display.width', 4000) #頁面宽度
#print(X.loc[X['SessionTrend'] != 0])

X = X.drop(['UUID', 'MemberCardLevel'], axis=1)
print(X.head())

X.to_excel("FeatureMatrix.xlsx")
X.to_pickle("FeatureMatrix.pkl")
