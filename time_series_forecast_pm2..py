#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#讀取資料集

df = pd.read_csv("D:/新竹_2019.csv", engine="python")


# In[3]:


df.columns = df.columns.str.strip() 
df


# In[4]:


#將空白值的那一行drop掉，並選取10~12月的資料，並將測站跟日期去除
df = df.drop([0])
df = df[df["日期"].str.contains("2019/10|2019/11|2019/12")]
df = df.drop(columns=['測站', '日期'])
df


# In[5]:


#因為要做時序的預測，因此將直向的日期資料改為橫向
new_df = df.iloc[:18,:]
plot_size = int(len(df) / 18)
for i in range(plot_size-1):
    new_df = pd.merge(new_df, df.iloc[i * 18 + 18:i * 18 + 2 * 18,:],on='測項')
    #new_df = pd.concat(new_df, df.iloc[i * 18 + 18:i * 18 + 2 * 18,:],axis=1)
df = new_df
#df.iloc[0:18,20:30]
df


# In[6]:


#將columns中的首尾空格去除
for rows in range(df.shape[0]):
    for cols in range(df.shape[1]):
        df.iloc[rows,cols] = df.iloc[rows,cols].strip()


# In[7]:


#宣告異常值的符號為symbol
symbol = ['x','#','*','A']
#尋找異常值，將異常值最右和最左的正常值相加除以2的function
def rep(i,j):
    check_left = j
    check_right = j
    left_NF = 0
    right_NF = 0

    #尋找左邊的正常值
    while(left_NF == 0):
        check_left -= 1
        l_check = 1
        for k in range(len(symbol)):
            if (symbol[k]) in df.iloc[i,check_left]:
                l_check = 0
        if(l_check == 1):
            left_NF = 1
            break
            
    #尋找右邊的正常值
    while(right_NF == 0):
        check_right += 1
        r_check = 1
        for k in range(len(symbol)):
            if (symbol[k]) in df.iloc[i,check_right]:
                r_check = 0
        if(r_check == 1):
            right_NF = 1
            break

    new_df = str((float(df.iloc[i,check_left]) + float(df.iloc[rows,check_right]))/2.0)
    df.iloc[i,j] = new_df


# In[8]:


# NR(無降雨)，用0取代，缺失值/無效值則以前後一小時平均值取代
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if (df.iloc[i,j]=="NR"):
            df.iat[i,j]="0"
        for k in range(len(symbol)):
            if (symbol[k]) in df.iloc[i,j]:
                if(j > 2):
                    rep(i,j)


# In[9]:


##將測項的那一列drop掉
df = df.drop(columns=['測項'])


# In[10]:


#將10、11月為訓練集，12月為測試集
train_df = df.iloc[:,:1465]
test_df = df.iloc[:,1465:]


# In[11]:


train_df


# In[12]:


#導入需要使用的模組
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error

#訓練集a-1
y1=[]
for i in range(1458):
    temp= train_df.iloc[9,i+6]
    y1.append(float(temp))
print(y1)
x1=[]
for j in range(1458):
    temp=[]
    for k in range(6):
        temp.append(float(train_df.iloc[9,j+k]))
    x1.append(temp)
print(x1)
#print(len(x1))
#測試集a-1
y1_test=[]
for i in range(738):
    temp= train_df.iloc[9,i+6]
    y1_test.append(float(temp))
print(y1_test)
x1_test=[]
for j in range(738):
    temp=[]
    for k in range(6):
        temp.append(float(train_df.iloc[9,j+k]))
    x1_test.append(temp)
print(x1_test)


# In[13]:


#將未來第一個小時當預測目標，只取PM2.5的特徵(線性回歸)
#建立模型
reg = LinearRegression().fit(x1, y1)  
#預測
y1_test_pred = reg.predict(x1_test)   
print("未來第一個小時當預測目標\n只取PM2.5的特徵\n線性回歸的MAE:")
#計算MAE
mean_absolute_error(y1_test, y1_test_pred)


# In[14]:


#將未來第一個小時當預測目標，只取PM2.5的特徵(隨機森林)
#建立模型
regr = RandomForestRegressor(max_depth=2, random_state=0) 
regr.fit(x1, y1)
#預測
y1_test_pred = regr.predict(x1_test)
print("未來第一個小時當預測目標\n只取PM2.5的特徵\n隨機森林的MAE:")
#計算MAE
mean_absolute_error(y1_test, y1_test_pred)


# In[15]:


#訓練集a-2
y2=[]
for i in range(1453):
    temp= train_df.iloc[9,i+11]
    y2.append(float(temp))
print(y2)
x2=[]
for j in range(1453):
    temp=[]
    for k in range(6):
        temp.append(float(train_df.iloc[9,j+k]))
    x2.append(temp)
print(x2)

#測試集a-2
y2_test=[]
for i in range(733):
    temp= train_df.iloc[9,i+11]
    y2_test.append(float(temp))
print(y2_test)
x2_test=[]
for j in range(733):
    temp=[]
    for k in range(6):
        temp.append(float(train_df.iloc[9,j+k]))
    x2_test.append(temp)
print(x2_test)


# In[16]:


#將未來第六個小時當預測目標(線性回歸)
#建立模型
reg = LinearRegression().fit(x2, y2)  
#預測
y2_test_pred = reg.predict(x2_test)   
print("未來第六個小時當預測目標\n只取PM2.5的特徵\n線性回歸的MAE:")
#計算MAE
mean_absolute_error(y2_test, y2_test_pred) 


# In[17]:


#將未來第六個小時當預測目標，只取PM2.5的特徵(隨機森林)
#建立模型
regr = RandomForestRegressor(max_depth=2, random_state=0) 
regr.fit(x2, y2)
#預測
y2_test_pred = regr.predict(x2_test) 
print("未來第六個小時當預測目標\n只取PM2.5的特徵\n隨機森林的MAE:")
#計算MAE
mean_absolute_error(y2_test, y2_test_pred)


# In[18]:


#訓練集b-1
yb1=[]
for i in range(1458):
    temp= train_df.iloc[9,i+6]
    yb1.append(float(temp))
print(yb1)
xb1=[]
for j in range(1458):
    temp_ar=[]
    for l in range(18):
        for k in range(6):
            temp=train_df.iloc[l,j+k]
            temp_ar.append(float(temp))
    xb1.append(temp_ar)
print(xb1)

#測試集b-1
yb1_test=[]
for i in range(738):
    temp= train_df.iloc[9,i+6]
    yb1_test.append(float(temp))
print(yb1_test)
xb1_test=[]
for j in range(738):
    temp_ar=[]
    for l in range(18):
        for k in range(6):
            temp=train_df.iloc[l,j+k]
            temp_ar.append(float(temp))
    xb1_test.append(temp_ar)
print(xb1_test)


# In[19]:


#將未來第一個小時當預測目標，取所有18*6個特徵(線性回歸)
#建立模型
reg = LinearRegression().fit(xb1, yb1)  
#預測
yb1_test_pred = reg.predict(xb1_test)   
print("未來第一個小時當預測目標\n取所有18種屬性\n線性回歸的MAE:")
#計算MAE
mean_absolute_error(yb1_test, yb1_test_pred) 


# In[20]:


#將未來第一個小時當預測目標，取所有18*6個特徵(隨機森林)
#建立模型
regr = RandomForestRegressor(max_depth=2, random_state=0) 
regr.fit(xb1, yb1)
#預測
yb1_test_pred = regr.predict(xb1_test) 
print("未來第ㄧ個小時當預測目標\nX取所有18種屬性\n隨機森林的MAE:")
#計算MAE
mean_absolute_error(yb1_test, yb1_test_pred)


# In[21]:


#訓練集b-2
yb2=[]
for i in range(1453):
    temp= train_df.iloc[9,i+11]
    yb2.append(float(temp))
print(yb2)
xb2=[]
for j in range(1453):
    temp_ar=[]
    for l in range(18):
        for k in range(6):
            temp=train_df.iloc[l,j+k]
            temp_ar.append(float(temp))
    xb2.append(temp_ar)
print(xb2)

#測試集b-2
yb2_test=[]
for i in range(733):
    temp= train_df.iloc[9,i+11]
    yb2_test.append(float(temp))
print(yb2_test)
xb2_test=[]
for j in range(733):
    temp_ar=[]
    for l in range(18):
        for k in range(6):
            temp=train_df.iloc[l,j+k]
            temp_ar.append(float(temp))
    xb2_test.append(temp_ar)
print(xb2_test)


# In[22]:


#將未來第六個小時當預測目標，取所有18*6個特徵(線性回歸)
#建立模型
reg = LinearRegression().fit(xb2, yb2)  
#預測
yb2_test_pred = reg.predict(xb2_test)   
print("未來第六個小時當預測目標\n取所有18種屬性\n線性回歸的MAE:")
#計算MAE
mean_absolute_error(yb2_test, yb2_test_pred) 


# In[23]:


#將未來第六個小時當預測目標，取所有18*6個特徵(隨機森林)
#建立模型
regr = RandomForestRegressor(max_depth=2, random_state=0) 
regr.fit(xb2, yb2)
#預測
yb2_test_pred = regr.predict(xb2_test) 
print("未來第六個小時當預測目標\n取所有18種屬性\n隨機森林的MAE:")
#計算MAE
mean_absolute_error(yb2_test, yb2_test_pred)


# In[ ]:




