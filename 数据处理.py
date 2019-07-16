
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import time
import datetime
import sys
sys.path.append('/Users/lin/Enterprise/missingno-master/missingno')
import missingno as msno
import math
import os
import re
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.preprocessing import LabelEncoder
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# In[2]:


def drop_dup(df):
    df.drop_duplicates(df.columns,'first',inplace=True)
    df = df.reset_index(drop=True)
    return df


# In[3]:


def combine_sets(df1,df2):
    df1 = df1.append(df2)
    df1 = df1.reset_index(drop=True)
    return df1


# In[4]:


#偏度
def get_kurt(s):
    return s.kurt()
#abs mean
def get_abs_mean(s):
    return np.abs(s).mean()
#square mean
def get_square_mean(s):
    return np.square(s).mean()


# In[5]:


def aggregate_fe(fe,social,data):
    for c in fe:
        print c
        #mean
        gp = social.groupby(by=[u'企业编号'])[c].mean().reset_index().rename(columns={c: c + u'_mean'})
        data = data.merge(gp ,on=[u'企业编号'], how='left')
        #min
        gp = social.groupby(by=[u'企业编号'])[c].max().reset_index().rename(columns={c: c + u'_min'})
        data = data.merge(gp ,on=[u'企业编号'], how='left')
        #max
        gp = social.groupby(by=[u'企业编号'])[c].min().reset_index().rename(columns={c: c + u'_max'})
        data = data.merge(gp ,on=[u'企业编号'], how='left')
        #var
        gp = social.groupby(by=[u'企业编号'])[c].var().reset_index().rename(columns={c: c + u'_var'})
        data = data.merge(gp ,on=[u'企业编号'], how='left')
        #skew
        gp = social.groupby(by=[u'企业编号'])[c].var().reset_index().rename(columns={c: c + u'_skew'})
        data = data.merge(gp ,on=[u'企业编号'], how='left')
        #kurt
        gp = social.groupby(by=[u'企业编号'])[c].apply(get_kurt).reset_index().rename(columns={c: c + u'_kurt'})
        data = data.merge(gp ,on=[u'企业编号'], how='left')
        #abs_mean
        gp = social.groupby(by=[u'企业编号'])[c].apply(get_abs_mean).reset_index().rename(columns={c: c + u'_abs_mean'})
        data = data.merge(gp ,on=[u'企业编号'], how='left')
        #abs_mean
        gp = social.groupby(by=[u'企业编号'])[c].apply(get_square_mean).reset_index().rename(columns={c: c + u'_square_mean'})
        data = data.merge(gp ,on=[u'企业编号'], how='left')
        del gp
        gc.collect()
        print data.shape
    return data


# In[1]:


data_path = u"./赛题1数据集/"


# In[4]:


test_path = u"./赛题1测试数据集/"


# In[3]:


data = pd.read_excel(data_path + u"企业评分.xlsx")


# In[5]:


#read test
test = pd.read_excel(u'赛题1结果_团队名.xlsx',names = {u'企业编号',u"企业评分"},)
tmp = test.iloc[0]
tmp[u'企业编号'] = 4001
test = test.append(tmp)
test = test.sort_values([u'企业编号'])
test = test.reset_index(drop=True)


# In[10]:


#处理企业评分
gp = data.groupby(by=[u'企业编号'])[u'企业总评分'].mean().reset_index().rename(columns={u'企业总评分': u'企业评分'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'企业评分'] = data[u'企业评分'].fillna(0).astype('float')
del gp
gc.collect()


# In[11]:


print data.shape
data = data.drop([u'企业总评分'],axis=1)
data.drop_duplicates(data.columns,'first',inplace=True)
data = data.reset_index(drop=True)
print data.shape


# In[12]:


data = data[data[u'企业评分'] > 70]
data = data.reset_index(drop=True)


# In[13]:


data.shape


# In[14]:


test[u'企业评分'] = 0


# In[15]:


#record len of the test set
len_test = len(test)


# In[16]:


data = data.append(test)
data = data.reset_index(drop=True)
del test


# In[7]:


sns.distplot(data[u'企业总评分'])


# In[17]:


data.shape


# ##  产品

# In[18]:


product = pd.read_excel(data_path+u'产品.xlsx')


# In[19]:


product_test = pd.read_excel(test_path+u'产品.xlsx')


# In[20]:


product = combine_sets(product,product_test)


# In[21]:


product = drop_dup(product)


# In[22]:


gp = product.groupby(by=[u'企业编号'])[u'产品类型'].count().reset_index().rename(columns={u'产品类型': u'产品种类'})
data = data.merge(gp, on=[u'企业编号'], how='left')
data[u'产品种类'] = data[u'产品种类'].fillna(0)
data[u'产品种类'] =  data[u'产品种类'].astype('uint16')
del gp
gc.collect()


# In[23]:


data.shape


# In[24]:


del product
gc.collect()


# ## 工商基本信息表 

# In[25]:


info = pd.read_excel(data_path + u'工商基本信息表.xlsx')


# In[26]:


info_test = pd.read_excel(test_path + u'工商基本信息表.xlsx')


# In[27]:


info = combine_sets(info,info_test)


# In[28]:


info.drop_duplicates(info.columns,'first',inplace=True)
info = info.reset_index(drop=True)


# In[29]:


#缺失信息
info[u'工商信息缺失'] = info.T.isnull().sum()


# In[30]:


#是否在本省登记
info[u'登记机关区域代码'] = info[u'登记机关区域代码'].fillna('000000')
info[u'是否在本地登记'] = (info[u'登记机关区域代码'].astype('int') - info[u'地区代码'].astype('int')).apply(lambda x : 0 if x == 0 else 1)


# In[31]:


#分割城市地区代码
info[u'城市代码'] = info[u'城市代码'].apply(lambda x: re.findall(r'.{2}',str(x))[1])
info[u'地区代码'] = info[u'地区代码'].apply(lambda x: re.findall(r'.{2}',str(x))[2])


# In[32]:


#处理是否注销
info[u'注销原因'] = info[u'注销原因'].fillna(0)
info[u'注销原因'] = info[u'注销原因'].apply(lambda x: 0 if x == 0 else 1)
info[u'注销时间'] = info[u'注销时间'].fillna(0)
info[u'注销时间']  = info[u'注销时间'].apply(lambda x: 0 if x == 0 else 1)
info[u'是否注销'] = info[u'注销时间'] + info[u'注销原因']
info[u'是否注销'] = info[u'是否注销'].apply(lambda x : 0 if x == 0 else 1) # 处理既有原因又有时间的行


# In[33]:


#营业期限
info[u'经营期限至'] = info[u'经营期限至'].apply(lambda x: '9999-09-09' if x in [u'永续经营', u'长期'] else x)
info[u'有效期'] = info[u'经营期限至'].fillna('9999-09-09').apply(lambda x : re.compile(r'\d+').findall(x)[0]).astype('int') - info[u'经营期限自'].fillna('1970-09-09').apply(lambda x : re.compile(r'\d+').findall(x)[0]).astype('int')
info[u'有效期'] = info[u'有效期'].apply(lambda x : 1 if x > 1000 else 0)


# In[34]:


#是否换证 成立日期与发证日期
info[u'发照年'] = info[u'发照日期'].fillna('1970-01-01')
info[u'成立年'] = info[u'成立日期'].fillna('1970-01-01')
info[u'发照年'] = info[u'发照年'].apply(lambda x: re.compile(r'\d+').findall(x)[0])
info[u'成立年'] = info[u'成立年'].apply(lambda x: re.compile(r'\d+').findall(x)[0])
info[u'成立年'] = info[u'成立年'].astype('int')
info[u'发照年'] = info[u'发照年'].astype('int')
info[u'是否换照'] = (info[u'发照年'] - info[u'成立年']).apply(lambda x : 0 if x == 0 else 1)
info = info.drop([u'成立年',u'发照年'],axis=1)


# In[35]:


#label encoder
col_need_encode = [u'注册资本币种(正则)', u'经营状态', u'行业大类（代码）',                   u'是否上市', u'类型',   u'城市代码',                  u'地区代码', u'省份代码', u'行业小类（代码）']
for col in col_need_encode:
    info[col] = info[col].fillna(-1)
    info[col] = info[col].map(dict(zip(info[col].unique(), range(0, info[col].nunique()))))


# In[36]:


fe = [k for k in info.columns if k not in [u'注销时间',u'成立日期', u'登记机关区域代码', u'注销原因',u'经营期限自',u'经营期限至',u'发照日期']]


# In[37]:


data = data.merge(info[fe], on=[u'企业编号'], how='left')
data.shape


# In[38]:


del info
gc.collect()


# ## 作品著作权

# In[39]:


right = pd.read_excel(data_path + u'作品著作权.xlsx')


# In[40]:


right_test = pd.read_excel(test_path + u'作品著作权.xlsx')


# In[41]:


right =combine_sets(right, right_test)


# In[42]:


right.drop_duplicates(right.columns,'first',inplace=True)
right = right.reset_index(drop=True)


# In[43]:


right[u'作品著作权类别'] = right[u'作品著作权类别'].fillna(u'Y 未知')


# In[44]:


right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'文字', u'A 文字')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'文字作品', u'A 文字')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'音乐', u'B 音乐')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'音乐作品', u'B 音乐')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'佛山市顺德区孔雀廊娱乐唱片有限公司', u'B 音乐')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'作品著作权证书', u'C 作品著作权证书')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'赵炜坚', u'C 作品著作权证书')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'张斌', u'C 作品著作权证书')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'李鹏', u'C 作品著作权证书')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'陈宇鹏', u'C 作品著作权证书')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'万全', u'C 作品著作权证书')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'上海浦东新区陆家嘴财富管理培训中心', u'C 作品著作权证书')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'曲艺', u'D 曲艺')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'舞蹈', u'E 舞蹈')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'美术', u'F 美术')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'美术作品', u'F 美术')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'澄心轩', u'F 美术')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'摄影', u'G 摄影')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'摄影作品', u'G 摄影')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'电影', u'H 电影')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'类似摄制电影方法创作的作品', u'I 类似摄制电影方法创作作品')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'电影和类似摄制电影方法创作的作品', u'I 类似摄制电影方法创作作品')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'电影作品和类似摄制电影的方法创造的作品', u'I 类似摄制电影方法创作作品')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'工程设计图、产品设计图', u'J 工程设计图、产品设计图')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'工程设计图、产品设计图作品', u'J 工程设计图、产品设计图')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'地图、示意图', u'K 地图、示意图')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'图形', u'K 地图、示意图')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'其他作品', u'L 其他作品')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'其他', u'L 其他作品')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'变更', u'M 变更')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'建筑', u'N 建筑')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'模型', u'O 模型')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'西域圣果', u'P 食品')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'张记皇冠', u'P 食品')

right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'上海思茸信息科技有限公司', u'X 计算机软件')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'上海明师科技发展有限公司', u'X 计算机软件')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'高学网络科技（上海）有限公司', u'X 计算机软件')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'上海烛龙信息科技有限公司', u'X 计算机软件')
right[u'作品著作权类别'] = right[u'作品著作权类别'].replace(u'上海畅声网络科技有限公司', u'X 计算机软件')


# In[45]:


#drop_dup
right = drop_dup(right)


# In[46]:


gp = right.groupby(by=[u'企业编号'])[u'作品著作权类别'].count().reset_index().rename(columns={u'作品著作权类别': u'作品著作权数量'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'作品著作权数量'] = data[u'作品著作权数量'].fillna(0).astype('int')
del gp
gc.collect()


# In[47]:


gp = right.groupby(by=[u'企业编号'])[u'作品著作权类别'].nunique().reset_index().rename(columns={u'作品著作权类别': u'作品著作权种类'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'作品著作权种类'] = data[u'作品著作权种类'].fillna(0).astype('int')
del gp
gc.collect()


# In[48]:


del right
gc.collect()
data.shape


# ## 专利

# In[49]:


right = pd.read_excel(data_path + u'专利.xlsx')


# In[50]:


right_test = pd.read_excel(test_path + u'专利.xlsx')


# In[51]:


right = combine_sets(right, right_test)


# In[52]:


#drop_dup
right.drop_duplicates(subset=right.columns,keep='first',inplace=True)
rigth = right.reset_index(drop=True)


# In[53]:


gp = right.groupby(by=[u'企业编号'])[u'专利类型'].count().reset_index().rename(columns={u'专利类型': u'专利数量'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'专利数量'] = data[u'专利数量'].fillna(0).astype('int')
del gp
gc.collect()


# In[54]:


del right
gc.collect()
data.shape


# ## 资质认证

# In[55]:


cert = pd.read_excel(data_path + u'资质认证.xlsx')


# In[56]:


cert_test = pd.read_excel(test_path + u'资质认证.xlsx')


# In[57]:


cert = combine_sets(cert,cert_test)


# In[58]:


#只关注有效资质
cert = cert[cert[u'状态'] == u'有效']


# In[59]:


#删除重复列
cert.drop_duplicates(subset=cert.columns,keep='first',inplace=True)
cert = cert.reset_index(drop=True)


# In[60]:


gp = cert.groupby(by=[u'企业编号'])[u'证书名称'].count().reset_index().rename(columns={u'证书名称': u'有效资质数量'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'有效资质数量'] = data[u'有效资质数量'].fillna(0).astype('int')
del gp
gc.collect()


# In[61]:


del cert
gc.collect()
data.shape


# ## 招投标

# In[62]:


tmp_test = pd.read_excel(test_path + u'招投标.xlsx')


# In[63]:


tmp = pd.read_excel(data_path + u'招投标.xlsx')


# In[64]:


tmp = combine_sets(tmp,tmp_test)


# In[65]:


#drop_dup
tmp.drop_duplicates(tmp.columns,'first',inplace=True)
tmp = tmp.reset_index(drop=True)


# In[66]:


#进行了多少次招投标
gp = tmp.groupby(by=[u'企业编号'])[u'省份'].count().reset_index().rename(columns={u'省份': u'招投标次数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'招投标次数'] = data[u'招投标次数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[67]:


#在几个省（种类）进行过招投标
gp = tmp.groupby(by=[u'企业编号'])[u'省份'].nunique().reset_index().rename(columns={u'省份': u'招投标省份数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'招投标省份数'] = data[u'招投标省份数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[68]:


#招标了多少次
tmp1 = tmp[tmp[u'中标或招标'] == u'招标']
gp = tmp1.groupby(by=[u'企业编号'])[u'省份'].count().reset_index().rename(columns={u'省份': u'招标次数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'招标次数'] = data[u'招标次数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[69]:


#在几个省招过标
tmp1 = tmp[tmp[u'中标或招标'] == u'招标']
gp = tmp.groupby(by=[u'企业编号'])[u'省份'].nunique().reset_index().rename(columns={u'省份': u'招标省份数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'招标省份数'] = data[u'招标省份数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[70]:


#中标次数
tmp1 = tmp[tmp[u'中标或招标'] == u'中标']
gp = tmp1.groupby(by=[u'企业编号'])[u'省份'].count().reset_index().rename(columns={u'省份': u'中标次数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'中标次数'] = data[u'中标次数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[71]:


#在几个省中过标
tmp1 = tmp[tmp[u'中标或招标'] == u'中标']
gp = tmp1.groupby(by=[u'企业编号'])[u'省份'].nunique().reset_index().rename(columns={u'省份': u'中标省份数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'中标省份数'] = data[u'中标省份数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[72]:


#公告中标次数
tmp1 = tmp[tmp[u'公告类型'] == u'中标']
gp = tmp1.groupby(by=[u'企业编号'])[u'省份'].count().reset_index().rename(columns={u'省份': u'公告中标次数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'公告中标次数'] = data[u'公告中标次数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[73]:


#在几个省公告中标
tmp1 = tmp[tmp[u'公告类型'] == u'中标']
gp = tmp1.groupby(by=[u'企业编号'])[u'省份'].nunique().reset_index().rename(columns={u'省份': u'公告中标省份数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'公告中标省份数'] = data[u'公告中标省份数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[74]:


#公告合同次数
tmp1 = tmp[tmp[u'公告类型'] == u'合同']
gp = tmp1.groupby(by=[u'企业编号'])[u'省份'].count().reset_index().rename(columns={u'省份': u'公告合同次数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'公告合同次数'] = data[u'公告合同次数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[75]:


#在几个省有公告合同
tmp1 = tmp[tmp[u'公告类型'] == u'合同']
gp = tmp1.groupby(by=[u'企业编号'])[u'省份'].nunique().reset_index().rename(columns={u'省份': u'公告合同省份数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'公告合同省份数'] = data[u'公告合同省份数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[76]:


#公告成交次数
tmp1 = tmp[tmp[u'公告类型'] == u'成交']
gp = tmp1.groupby(by=[u'企业编号'])[u'省份'].count().reset_index().rename(columns={u'省份': u'公告成交次数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'公告成交次数'] = data[u'公告成交次数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[77]:


#在几个省有公告成交
tmp1 = tmp[tmp[u'公告类型'] == u'成交']
gp = tmp1.groupby(by=[u'企业编号'])[u'省份'].nunique().reset_index().rename(columns={u'省份': u'公告成交省份数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'公告成交省份数'] = data[u'公告成交省份数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[78]:


del tmp
gc.collect()


# ## 债券信息 -- 待处理

# In[79]:


#latter = pd.read_excel(data_path + u'债券信息.xlsx')


# In[80]:


#float(len(list(set(data[u'企业编号']).difference(set(latter[u'企业编号']))))) / len(data)


# In[81]:


#超过70%的data的编号不在latter中，暂时放弃！


# ## 一般纳税人 -- 待处理

# In[82]:


#tax = pd.read_excel(data_path + u'一般纳税人.xlsx')


# In[83]:


#msno.matrix(tax)


# In[84]:


#缺失过多


# ## 纳税A级年份

# In[85]:


tax = pd.read_excel(data_path + u'纳税A级年份.xlsx')


# In[86]:


tax_test = pd.read_excel(test_path + u'纳税A级年份.xlsx')


# In[87]:


tax = combine_sets(tax,tax_test)


# In[88]:


#drop_dup
tax.drop_duplicates(tax.columns,'first',inplace=True)
tax = tax.reset_index(drop=True)


# In[89]:


#有几年是纳税A级
gp = tax.groupby(by=[u'企业编号'])[u'纳税A级年份'].nunique().reset_index().rename(columns={u'纳税A级年份': u'纳税A级年份数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'纳税A级年份数'] = data[u'纳税A级年份数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[90]:


del tax
gc.collect()
data.shape


# ## 软著著作权

# In[91]:


soft = pd.read_excel(data_path + u'软著著作权.xlsx')


# In[92]:


soft_test = pd.read_excel(test_path + u'软著著作权.xlsx')


# In[93]:


soft = combine_sets(soft,soft_test)


# In[94]:


#drop_dup
soft.drop_duplicates([u'企业编号',u'软件全称',u'软件著作权版本号'],'first', inplace=True)
soft = soft.reset_index(drop=True)


# In[95]:


#处理年份
soft[u'软件著作权登记批准日期'] = soft[u'软件著作权登记批准日期'].replace(u'上海玄霆娱乐信息科技有限公司中国',u'1970-01-01')
soft[u'软件著作权登记批准日期'] = soft[u'软件著作权登记批准日期'].replace(u'北京华夏晓能石油技术有限公司中国',u'1970-01-01')
soft[u'年份'] = soft[u'软件著作权登记批准日期'].fillna(u'1970-01-01').apply(lambda x : re.findall(r'.{4}',str(x.encode('gbk')))[0])


# In[96]:


soft[u'年份'] = soft[u'年份'].astype('int')


# In[97]:


#有几年有软著批准
gp = soft.groupby(by=[u'企业编号'])[u'年份'].nunique().reset_index().rename(columns={u'年份': u'软著年份数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'软著年份数'] = data[u'软著年份数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[98]:


#共有多少软著批准
gp = soft.groupby(by=[u'企业编号'])[u'年份'].count().reset_index().rename(columns={u'年份': u'软著数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'软著数'] = data[u'软著数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[99]:


del soft
gc.collect()
data.shape


# ## 商标

# In[100]:


band = pd.read_excel(data_path + u'商标.xlsx')


# In[101]:


band_test = pd.read_excel(test_path + u'商标.xlsx')


# In[102]:


band = combine_sets(band,band_test)


# In[103]:


#drop_dup
band.drop_duplicates(band.columns,'first',inplace=True)
band = band.reset_index(drop=True)


# In[104]:


#有几个商标已完成注册
tmp = band[band[u'商标状态'] == u'注册']
gp = tmp.groupby(by=[u'企业编号'])[u'申请日期'].nunique().reset_index().rename(columns={u'申请日期': u'注册商标数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'注册商标数'] = data[u'注册商标数'].fillna(0).astype('int')
print data.shape
del tmp
del gp
gc.collect()


# In[105]:


del band
gc.collect()


# ## 融资信息 

# In[106]:


ipo = pd.read_excel(data_path + u'融资信息.xlsx')


# In[107]:


ipo_test = pd.read_excel(test_path + u'融资信息.xlsx')


# In[108]:


ipo = combine_sets(ipo, ipo_test)


# In[109]:


#drop_dup
ipo.drop_duplicates(ipo.columns,'first',inplace=True)
ipo =ipo.reset_index(drop=True)


# In[110]:


#完成几次融资
gp = ipo.groupby(by=[u'企业编号'])[u'轮次'].nunique().reset_index().rename(columns={u'轮次': u'融资次数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'融资次数'] = data[u'融资次数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[111]:


del ipo
gc.collect()


# ## 项目信息 -- 待处理

# In[112]:


#project = pd.read_excel(data_path + u'项目信息.xlsx')


# In[113]:


# from jieba import analyse
# # 引入TF-IDF关键词抽取接口
# tfidf = analyse.extract_tags
 
# # 原始文本
# text = str(project[project[u'企业编号'] == 1022][u'标签'].values)
 
# # 基于TF-IDF算法进行关键词抽取
# keywords = tfidf(text)
# print "keywords by tfidf:"
# # 输出抽取出的关键词
# for keyword in keywords:
#     print keyword + "/",


# In[114]:


# #词频处理
# from sklearn.feature_extraction.text import CountVectorizer  
# vectorizer=CountVectorizer()
# corpus= [project[u'标签'][1]]
# print vectorizer.fit_transform(corpus)

# vectorizer.get_feature_names()


# ## 竞品 -- 待处理 

# In[115]:


#comp = pd.read_excel(data_path + u'竞品.xlsx')


# In[116]:


#comp.head()


# ## 海关进出口信用

# In[117]:


cert = pd.read_excel(data_path + u'海关进出口信用.xlsx')


# In[118]:


cert_test = pd.read_excel(test_path + u'海关进出口信用.xlsx')


# In[119]:


cert = combine_sets(cert, cert_test)


# In[120]:


#drop_dup 删除全部相同的行
fe = [k for k in cert.columns if k not in [u'信用等级']]
cert.drop_duplicates(fe,'first',inplace=True)
cert = cert.reset_index(drop=True)


# In[121]:


#填补缺失值
cert[u'经济区划'] = cert[u'经济区划'].fillna(u'未知')
cert[u'经营类别'] = cert[u'经营类别'].fillna(u'未知')
cert[u'海关注销标志'] = cert[u'海关注销标志'].fillna(u'未知')
cert[u'年报情况'] = cert[u'年报情况'].fillna(u'未知')


# In[122]:


#编码
fe = [k for k in cert.columns if k not in [u'信用等级',u'企业编号']]
for col in fe:
    cert[col] = cert[col].map(dict(zip(cert[col].unique(), range(0, cert[col].nunique()))))


# In[123]:


train = cert[cert[u'信用等级'].notnull()]
print train.shape


# In[124]:


test = cert[cert[u'信用等级'].isnull()]
print test.shape


# In[125]:


#target 编码
train[u'信用等级'] = train[u'信用等级'].map(dict(zip(train[u'信用等级'].unique(), range(0, train[u'信用等级'].nunique()))))


# In[126]:


train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


# In[127]:


target = train[u'信用等级'].values


# In[128]:


#多分类
target = train[u'信用等级'].values

params={
'booster':'gbtree',
'objective': 'multi:softmax', 
'num_class': 4, 
'max_depth':3, 
'silent':1 ,
'eta': 0.1,
'seed':710,
'eval_metric': 'merror'}

train_X,val_X, train_y, val_y = train_test_split(train[fe].values,target,test_size = 0.3,random_state = 0) 

d_train = xgb.DMatrix(train_X, train_y) 
d_valid = xgb.DMatrix(val_X, val_y) 
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
xgb_model = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=20,maximize=True, verbose_eval=1)

test[u'信用等级'] = xgb_model.predict(xgb.DMatrix(test[fe].values))


# In[129]:


train[u'信用等级'].value_counts()


# In[130]:


train = train.append(test)
train = train.sort_values(by=[u'企业编号'],ascending=1)
train = train.reset_index(drop = True)


# In[131]:


train[u'信用等级'] =train[u'信用等级'].astype('int')


# In[132]:


cert = cert.drop([u'信用等级'],axis=1)


# In[133]:


fe = [k for k in train.columns if k not in [u'信用等级']]


# In[134]:


cert = cert.merge(train,on=fe,how='left')


# In[135]:


#选取第二行
cert.drop_duplicates([u'企业编号'],'last',inplace=True)
cert.shape


# In[136]:


cert = cert.reset_index(drop=True)


# In[137]:


gp = cert[[u'企业编号',u'信用等级']]


# In[138]:


data = data.merge(gp,on=[u'企业编号'],how='left')
data[u'信用等级']  = data[u'信用等级'].fillna(len(data[u'信用等级'].unique()) -1)
data[u'信用等级'] = data[u'信用等级'].astype('int')
data.shape


# In[139]:


del train,test,cert,gp
gc.collect()


# ### 购地 -- 缺失值过多 -- 只做统计特征

# In[140]:


market_sell = pd.read_excel(data_path + u'购地-市场交易-土地转让.xlsx')#(33744, 10)#空值过多
#market_mort = pd.read_excel(data_path + u'购地-市场交易-土地抵押.xlsx')#(33744, 11)
#result = pd.read_excel(data_path + u'购地-结果公告.xlsx')#(33744, 18) #空值过多
#company_sell = pd.read_excel(data_path + u'购地-房地产大地块出让情况.xlsx')#(33744, 10)#空值过多
publication = pd.read_excel(data_path + u'购地-地块公示.xlsx')#(20033, 9)
company_buy = pd.read_excel(data_path + u'购地-房地产大企业购地情况.xlsx')#(20033, 11)


# In[141]:


market_sell_test = pd.read_excel(test_path + u'购地-市场交易-土地转让.xlsx')#(33744, 10)#空值过多
publication_test = pd.read_excel(test_path + u'购地-地块公示.xlsx')#(20033, 9)
company_buy_test = pd.read_excel(test_path + u'购地-房地产大企业购地情况.xlsx')#(20033, 11)


# In[142]:


market_sell = combine_sets(market_sell,market_sell_test)
publication = combine_sets(publication,publication_test)
company_buy = combine_sets(company_buy, company_buy_test)


# In[143]:


market_sell[u'土地面积(公顷)'] = market_sell[u'土地面积(公顷)'].fillna(market_sell[u'土地面积(公顷)'].mean())
market_sell[u'土地用途'] = market_sell[u'土地用途'].fillna(u'未知')


# In[144]:


#出让土地面积
gp = market_sell.groupby(by=[u'企业编号'])[u'土地面积(公顷)'].sum().reset_index().rename(columns={u'土地面积(公顷)': u'出让土地总面积'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'出让土地总面积'] = data[u'出让土地总面积'].fillna(0).astype('float')
print data.shape
del gp
gc.collect()


# In[145]:


#出让土地次数
gp = market_sell.groupby(by=[u'企业编号'])[u'土地用途'].count().reset_index().rename(columns={u'土地用途': u'出让土地次数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'出让土地次数'] = data[u'出让土地次数'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[146]:


#出让土地用途种类
gp = market_sell.groupby(by=[u'企业编号'])[u'土地用途'].nunique().reset_index().rename(columns={u'土地用途': u'出让土地用途种类'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'出让土地用途种类'] = data[u'出让土地用途种类'].fillna(0).astype('int')
print data.shape
del gp
gc.collect()


# In[147]:


del market_sell, publication, company_buy
gc.collect()


# ## 年报

# ### 年报-网站或网点信息 年报-对外投资信息-- 无任何信息
# 

# In[148]:


#web = pd.read_excel(data_path + u'年报-网站或网点信息.xlsx')
#年报-对外投资信息.xlsx


# ### 年报-社保信息

# In[149]:


social = pd.read_excel(data_path + u'年报-社保信息.xlsx')


# In[150]:


social_test = pd.read_excel(test_path + u'年报-社保信息.xlsx')


# In[151]:


social = combine_sets(social,social_test)


# In[152]:


#drop_dup
social.drop_duplicates(social.columns,'first',inplace=True)
social = social.reset_index(drop = True)


# In[153]:


#替换不公示为空值
null_value = social[u'城镇职工基本养老保险人数'][0]
social= social.replace(u'企业选择不公示', null_value)
social = social.replace(u'选择不公示', null_value)


# In[154]:


fe = [k for k in social.columns if k not in [u'年报年份',u'企业编号']]


# In[155]:


#获取每行缺失的比例
social[u'工商信息缺失'] = social[fe].T.isnull().sum() / len(fe)


# In[156]:


#删除全空行
social = social[social[u'工商信息缺失'] < 1]
social.shape


# In[157]:


#单位参加工伤保险缴费基数 缺失过多，删除
social = social.drop([u'单位参加工伤保险缴费基数'],axis=1)
social = social.reset_index(drop=True)


# In[158]:


#分组
value_col = [u'单位参加失业保险缴费基数',u'单位参加生育保险缴费基数',u'单位参加城镇职工基本养老保险缴费基数',      u'参加城镇职工基本养老保险本期实际缴费金额',u'参加失业保险本期实际缴费金额',u'参加职工基本医疗保险本期实际缴费金额',      u'参加生育保险本期实际缴费金额',u'参加工伤保险本期实际缴费金额',u'单位参加城镇职工基本养老保险累计欠缴金额',      u'单位参加工伤保险累计欠缴金额',u'单位参加生育保险累计欠缴金额',u'单位参加职工基本医疗保险缴费基数',      u'单位参加失业保险累计欠缴金额',u'单位参加职工基本医疗保险累计欠缴金额']
human_col = [k for k in social.columns if k not in [u'企业编号',u'年报年份',u'工商信息缺失'] + value_col]


# #### 处理人数列
# 

# In[159]:


#人数 转化成人数，删除人字，保持空值
for c in human_col:
    print c
    social[c] = social[c].fillna(u'无人').apply(lambda x: re.sub("\D", "", x.encode('utf-8')))
    #替换‘’到nan，方便后续处理
    social[c] = social[c].replace('',-1)
    #转化成int
    social[c] = social[c].astype('int')
    #替换成空值
    social[c] = social[c].replace(-1,null_value)


# In[160]:


data = aggregate_fe(human_col,social,data)


# In[161]:


#有效年报年份数
gp = social.groupby(by=[u'企业编号'])[u'年报年份'].count().reset_index().rename(columns={u'年报年份': u'有效年报年份数'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'有效年报年份数'] = data[u'有效年报年份数'].fillna(0)
data[u'有效年报年份数'] = data[u'有效年报年份数'].astype('int')
del gp
gc.collect()


# #### 处理数值列

# In[162]:


#处理“万元” 无数字形式
for c in value_col:
    print c
    social[c] = social[c].fillna(u'-1').apply(lambda x: null_value if x.startswith((u'万元')) else x)
    #转成float
    social[c] =  social[c].fillna(u'-1').apply(lambda x : float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0]))
    #转换回nan
    social[c] = social[c].replace(-1.0,null_value)


# In[163]:


data = aggregate_fe(value_col,social,data)


# In[164]:


del social
gc.collect()


# ### 年报-企业资产状况信息--缺失过多，暂缓

# In[165]:


#funt = pd.read_excel(data_path + u'年报-企业资产状况信息.xlsx')


# In[166]:


#funt = funt.merge(social,on=[u'企业编号',u'年报年份'],how='left')


# In[167]:


#funt.drop_duplicates(funt.columns,'first',inplace=True)


# In[168]:


#funt = funt.replace(u'企业选择不公示',null_value)


# In[169]:


# for c in funt.columns:
#     print c, float(funt[c].isnull().sum()) / len(funt)


# In[170]:


# del funt
# gc.collect()


# ### 年报-企业基本信息

# In[171]:


info = pd.read_excel(data_path+u'年报-企业基本信息.xlsx')


# In[172]:


info_test = pd.read_excel(test_path+u'年报-企业基本信息.xlsx')


# In[173]:


info = combine_sets(info, info_test)


# In[174]:


#drop_dup
print info.shape
info.drop_duplicates(info.columns,'first',inplace=True)
info = info.reset_index(drop=True)
print info.shape


# In[175]:


info = info.replace(u'企业选择不公示',null_value)


# In[176]:


#企业经营状态是否发生改变
gp = info.groupby(by=[u'企业编号'])[u'企业经营状态'].nunique().reset_index().rename(columns={u'企业经营状态': u'企业经营状态种类'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'企业经营状态种类'] = data[u'企业经营状态种类'].apply(lambda x: 1 if x == 1 else 0)
print data.shape


# In[177]:


#网站或网点是否发生改变
gp = info.groupby(by=[u'企业编号'])[u'是否有网站或网点'].nunique().reset_index().rename(columns={u'是否有网站或网点': u'是否有网站或网点种类'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'是否有网站或网点种类'] = data[u'是否有网站或网点种类'].apply(lambda x: 1 if x == 1 else 0)
print data.shape


# In[178]:


#提供对外担保是否发生改变
gp = info.groupby(by=[u'企业编号'])[u'是否提供对外担保'].nunique().reset_index().rename(columns={u'是否提供对外担保': u'是否提供对外担保种类'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'是否提供对外担保种类'] = data[u'是否提供对外担保种类'].apply(lambda x: 1 if x == 1 else 0)
print data.shape


# In[179]:


#有限责任公司本年度是否发生股东股权转
gp = info.groupby(by=[u'企业编号'])[u'有限责任公司本年度是否发生股东股权转'].nunique().reset_index().rename(columns={u'有限责任公司本年度是否发生股东股权转': u'有限责任公司本年度是否发生股东股权转种类'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'有限责任公司本年度是否发生股东股权转种类'] = data[u'有限责任公司本年度是否发生股东股权转种类'].apply(lambda x: 1 if x == 1 else 0)
print data.shape


# In[180]:


#企业是否有投资信息或购买其他公司股权
gp = info.groupby(by=[u'企业编号'])[u'企业是否有投资信息或购买其他公司股权'].nunique().reset_index().rename(columns={u'企业是否有投资信息或购买其他公司股权': u'企业是否有投资信息或购买其他公司股权种类'})
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'企业是否有投资信息或购买其他公司股权种类'] = data[u'企业是否有投资信息或购买其他公司股权种类'].apply(lambda x: 1 if x == 1 else 0)
print data.shape


# In[181]:


#按照年份和编号排序
tmp = info.sort_values([u'企业编号', u'年报年份'], ascending=[True, True])
tmp.drop_duplicates([u'企业编号'],'last',inplace=True)
tmp = tmp.reset_index(drop=True)


# In[182]:


#label encoder
cols = [k for k in tmp.columns if k not in [u'企业编号',u'年报年份',u'发布日期',u'注册资本',u'从业人数']]
for col in cols:
    print col
    tmp[col] = tmp[col].fillna(-1)
    tmp[col] = tmp[col].map(dict(zip(tmp[col].unique(), range(0, tmp[col].nunique()))))
    tmp[col] = tmp[col].astype('int')
    tmp.rename(columns={col: col + u'_最新一年'},inplace= True)


# In[183]:


#接回到data
col = [k for k in tmp.columns if k not in [u'年报年份',u'发布日期',u'注册资本',u'从业人数']]
data = data.merge(tmp[col],on= [u'企业编号'],how = 'left')


# In[184]:


del tmp, info
gc.collect()


# ### 年报-股东（发起人）及出资信息

# In[185]:


boss = pd.read_excel(data_path + u'年报-股东（发起人）及出资信息.xlsx')


# In[186]:


boss_test = pd.read_excel(test_path + u'年报-股东（发起人）及出资信息.xlsx')


# In[187]:


boss = combine_sets(boss,boss_test)


# In[188]:


boss.drop_duplicates(boss.columns,'first',inplace=True)
boss = boss.reset_index(drop=True)


# In[189]:


boss[u'认缴出资信息'] =boss[u'认缴出资信息'].replace(u'认缴出资方式：\n认缴出资额（万元）：\n认缴出资日期：',u'0')


# In[190]:


#认缴出资信息
boss[u'认缴出资信息'] =  boss[u'认缴出资信息'].fillna(u'0').apply(lambda x : float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0]))


# In[191]:


boss[u'实缴出资信息'] = boss[u'实缴出资信息'].fillna(u'0：0：0').apply(lambda x: (u'0：0：0') if x.startswith(u'\n实缴出资日期') else x)


# In[192]:


#实际出资信息
boss[u'实缴出资信息'] = boss[u'实缴出资信息'].fillna(u'0：0：0').apply(lambda x: (u'0：0：0') if x.startswith(u'\n实缴出资日期') else x)
boss[u'实缴出资信息'] = boss[u'实缴出资信息'].fillna(u'0：0：0').apply(lambda x: x.split(u'：')[2])


# In[193]:


boss[u'实缴出资信息'] = boss[u'实缴出资信息'].fillna(u'0：0：0').apply(lambda x: (u'0：0：0') if x.startswith(u'\n实缴出资日期') else x)


# In[194]:


boss[u'实缴出资信息'] =  boss[u'实缴出资信息'].fillna(u'0：0：0').apply(lambda x : float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0]))


# In[195]:


data = aggregate_fe([u'实缴出资信息'],boss,data)


# In[196]:


#认缴实缴是否有不同的记录
boss[u'if_same'] = boss[u'实缴出资信息'] - boss[u'认缴出资信息']
gp = boss.groupby(by=[u'企业编号'])[u'if_same'].nunique().reset_index()
data = data.merge(gp ,on=[u'企业编号'], how='left')
data[u'if_same'] = data[u'if_same'].apply(lambda x: 1 if x == 1 else 0)
print data.shape


# In[197]:


del boss
gc.collect()


# ## 上市信息

# ### 上市信息财务信息资产负债表

# In[198]:


funt = pd.read_excel(data_path + u'上市信息财务信息资产负债表.xlsx')


# In[199]:


funt_test = pd.read_excel(test_path + u'上市信息财务信息资产负债表.xlsx')


# In[200]:


funt_test = funt_test.rename(columns = {u'企业编号s':u'企业编号'})


# In[201]:


funt = combine_sets(funt, funt_test)


# In[202]:


funt = drop_dup(funt)


# In[203]:


fe = [k for k in funt.columns if k not in [u'负债:存货跌价准备(元)',u'资产:累计折旧(元)',u'流动比率',u'日期',u'企业编号',                                          u'标题']]


# In[204]:


#转换亿、万到数字 转换到亿元
def change_to_value(x):
    if(x.endswith(u'亿')):
        return float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0]) 
    elif(x.endswith(u'万')):
        return float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0]) / 10000.0
    else:
        return float(x) / 100000000.0


# In[205]:


funt = funt.replace(u'--',null_value)
funt = funt.replace(u'正无穷大万亿',null_value)


# In[206]:


for c in fe:
    funt[c] = funt[c].fillna(u'0')
    funt[c] = funt[c].apply(change_to_value)


# In[207]:


data = aggregate_fe(fe,funt,data)


# In[208]:


del funt
gc.collect()


# ### 上市信息财务信息运营能力指标 

# In[209]:


funt = pd.read_excel(data_path + u'上市信息财务信息运营能力指标.xlsx')


# In[210]:


funt_test = pd.read_excel(test_path + u'上市信息财务信息运营能力指标.xlsx')


# In[211]:


funt = combine_sets(funt,funt_test)


# In[212]:


funt = drop_dup(funt)


# In[213]:


funt = funt.replace('--',0.0)


# In[214]:


funt.head()


# In[215]:


fe= [k for k in funt.columns if k not in [u'企业编号',u'标题',u'日期']]


# In[216]:


for c in fe:
    funt[c] = funt[c].fillna(0)
    funt[c] = funt[c].astype('float')


# In[217]:


data = aggregate_fe(fe,funt,data)


# In[218]:


del funt
gc.collect()


# ### 上市信息财务信息盈利能力指标

# In[219]:


profit = pd.read_excel(data_path + u'上市信息财务信息盈利能力指标.xlsx')


# In[220]:


profit_test = pd.read_excel(test_path + u'上市信息财务信息盈利能力指标.xlsx')


# In[221]:


profit_test = combine_sets(profit,profit_test)


# In[222]:


def profit_change_to_value(x):
    if(x.startswith(u'--')):
        return 0.0
    elif(x.endswith(u'%')):
        return float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0])


# In[223]:


profit = drop_dup(profit)


# In[224]:


fe = [k for k in profit.columns if k not in [u'企业编号',u'标题',u'日期']]


# In[225]:


for c in fe:
    profit[c] = profit[c].apply(profit_change_to_value) 


# In[226]:


data = aggregate_fe(fe,profit,data)


# In[227]:


del profit
gc.collect()


# ### 上市信息财务信息-现金流量表

# In[228]:


profit = pd.read_excel(data_path + u'上市信息财务信息-现金流量表.xlsx')


# In[229]:


profit_test = pd.read_excel(test_path + u'上市信息财务信息-现金流量表.xlsx')


# In[230]:


profit = combine_sets(profit,profit_test)


# In[231]:


profit = drop_dup(profit)


# In[232]:


fe = [k for k in profit.columns if k not in [u'筹资:吸收投资收到的现金(元)',u'企业编号',u'日期',u'标题']]


# In[233]:


def cash_change_to_value(x):
    if(x.endswith(u'亿')):
        return float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0]) 
    elif(x.endswith(u'万')):
        return float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0]) / 10000.0
    else:
        return float(x) / 100000000.0


# In[234]:


profit = profit.replace('--',u'0')
profit = profit.fillna(u'0')


# In[235]:


for c in fe:
    profit[c] = profit[c].apply(cash_change_to_value)


# In[236]:


data = aggregate_fe(fe,profit,data)


# In[237]:


del profit
gc.collect()


# ### 上市信息财务信息-利润表

# In[238]:


profit = pd.read_excel(data_path + u'上市信息财务信息-利润表.xlsx')


# In[239]:


profit_test = pd.read_excel(test_path + u'上市信息财务信息-利润表.xlsx')


# In[240]:


profit = combine_sets(profit, profit_test)


# In[241]:


profit = drop_dup(profit)


# In[242]:


profit = profit.replace(u'--',u'0')
profit = profit.fillna(u'0')


# In[243]:


fe = [k for k in profit.columns if k not in [u'企业编号',u'标题',u'日期']]


# In[244]:


for c in fe:
    profit[c] = profit[c].apply(change_to_value)


# In[245]:


data = aggregate_fe(fe,profit,data)


# In[246]:


del profit
gc.collect()


# ### 上市信息财务信息-成长能力指标

# In[247]:


profit = pd.read_excel(data_path + u'上市信息财务信息-成长能力指标.xlsx')


# In[248]:


profit_test = pd.read_excel(test_path + u'上市信息财务信息-成长能力指标.xlsx')


# In[249]:


profit = combine_sets(profit,profit_test)


# In[250]:


profit = drop_dup(profit)


# In[251]:


fe = [k for k in profit.columns if k not in [u'企业编号',u'标题',u'日期']]


# In[252]:


profit = profit.replace(u'--',u'0')
profit = profit.fillna(u'0')


# In[253]:


def power_change_to_value(x):
    if(x.endswith(u'亿')):
        return float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0])
    elif(x.endswith(u'万')):
        return float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0]) / 10000.0
    elif(x.startswith(u'--%')):
        return 0.0
    elif(x.endswith(u'%') & x.startswith(u'--') == False):
        return float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0])


# In[254]:


for c in fe:
    profit[c] = profit[c].apply(power_change_to_value)


# In[255]:


data = aggregate_fe(fe,profit,data)


# In[256]:


del profit
gc.collect()


# ### 上市信息财务信息-财务风险指标

# In[257]:


profit = pd.read_excel(data_path + u'上市信息财务信息-财务风险指标.xlsx')


# In[258]:


profit_test = pd.read_excel(test_path + u'上市信息财务信息-财务风险指标.xlsx')


# In[259]:


profit = combine_sets(profit, profit_test)


# In[260]:


profit = profit.replace(u'--',u'0')
profit = profit.fillna(u'0')


# In[261]:


fe = [k for k in profit.columns if k not in [u'企业编号',u'标题',u'日期']]


# In[262]:


def risk_change_to_value(x):
    if(x.startswith(u'--')):
        return 0.0
    elif(x.endswith(u'%')):
        return float(re.findall(r'-?\d+\.?\d*e?-?\d*?',x.encode('utf-8'))[0])
    else:
        return float(x)


# In[263]:


for c in fe:
    profit[c] = profit[c].apply(risk_change_to_value)


# In[264]:


data = aggregate_fe(fe,profit,data)


# In[265]:


del profit
gc.collect()


# ### 上市公司财务信息-每股指标

# In[266]:


profit = pd.read_excel(data_path + u'上市公司财务信息-每股指标.xlsx')


# In[267]:


profit_test = pd.read_excel(test_path + u'上市公司财务信息-每股指标.xlsx')


# In[268]:


profit = combine_sets(profit,profit_test)


# In[269]:


profit = drop_dup(profit)


# In[270]:


profit = profit.replace(u'--',u'0')
profit = profit.fillna(u'0')


# In[271]:


fe = [k for k in profit.columns if k not in [u'企业编号',u'标题',u'日期']]


# In[272]:


for c in fe:
    profit[c] = profit[c].astype('float')


# In[273]:


data = aggregate_fe(fe,profit,data)


# In[274]:


del profit
gc.collect()


# #### 处理完毕

# ## 分离数据

# In[279]:


train = data.iloc[:(len(data) - len_test)]
test = data.iloc[(len(data) - len_test):]
print train.shape, test.shape


# In[276]:


data.shape


# In[281]:


#save
train.to_excel("train.xlsx")
test.to_excel("test.xlsx")

