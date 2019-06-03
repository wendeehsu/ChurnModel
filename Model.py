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

# output:
# averageDate = datetime.timedelta(days=85, seconds=10904, microseconds=556717)
# 中位數 = datetime.timedelta(days=44, seconds=57600)

def filterZeros(delta):
    if delta == timedelta(0):
        return False
    else:
        return True

modifyAverageDate = list(filter(filterZeros, averageDate))
modifyAverageDate.sort()
nonZeroAvgPeriod = sum(modifyAverageDate, timedelta()) / len(modifyAverageDate)
print("nonZeroAvgPeriod = ", nonZeroAvgPeriod)

# output:
# nonZeroAvgPeriod = datetime.timedelta(days=96, seconds=78825, microseconds=751115)
# 中位數 = datetime.timedelta(days=55)



