#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xlsxwriter
import math
import re
from datetime import datetime, timedelta
from openpyxl import load_workbook

OrderData = pd.read_csv("OrderDataForNTU.txt", sep = ',', encoding = 'utf8')

def FormateTime(date):
    return datetime.strptime(date, '%Y/%m/%d')

def GetPeriod(dates):
    dates.sort()
    avg = (dates[-1] - dates[0])/ (len(dates)-1)
    return avg

tradeDates = OrderData['TradesDate'].tolist()
UUIDs = OrderData['UUID'].tolist()

targetUUIDS = [] # 購買次數高於兩次的UUID們
averageDate = [] # 購買次數高於兩次的人的購買週期

for i,uuid in enumerate(list(set(UUIDs))):
    buyNums = UUIDs.count(uuid)
    print(i,uuid)
    if buyNums >= 2:
        targetUUIDS.append(uuid)
        rawDates = OrderData.loc[OrderData['UUID'] == uuid]['TradesDate'].tolist()
        dates = list(map(FormateTime, rawDates))
        del rawDates
        
        averageDate.append(GetPeriod(dates))
        del dates

avgPeriod = sum(averageDate, timedelta()) / len(averageDate)
print("targetUUIDS = ", targetUUIDS)
print("averageDate = ", avgPeriod)

# 3. 丟RNN (參考 “寶哥RNN顧客流失預測”)

