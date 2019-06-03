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


# 1.先刪掉買低於兩次的人，只取 orderData裡購買次數高於兩次的UUID們 => 存成一個List ）
# 2. 計算這些人的平均購買週期（先算個別的購買週期，再求出所有人的平均購買週期）=>求出的數字為”多久未回購為流失“的值

def FormateTime(date):
    return datetime.strptime(date, '%Y/%m/%d')

tradeDates = OrderData['TradesDate'].tolist()
UUIDs = OrderData['UUID'].tolist()

targetUUIDS = [] # 購買次數高於兩次的UUID們
averageDate = [] # 購買次數高於兩次的人的購買週期

for i,uuid in enumerate(list(set(UUIDs))):
    buyNums = UUIDs.count(uuid)
    if buyNums >= 2:
        targetUUIDS.append(uuid)
        
        dates = []
        for index, globalId in enumerate(UUIDs):
            if globalId == uuid:
                dates.append(FormateTime(tradeDates[index]))
        dates.sort()
        avg = (dates[-1] - dates[0])/ (buyNums-1)
        averageDate.append(avg)
        del dates

print("targetUUIDS = ", targetUUIDS)
print("averageDate = ", mean(averageDate))

# 3. 丟RNN (參考 “寶哥RNN顧客流失預測”)

