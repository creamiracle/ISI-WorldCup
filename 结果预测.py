
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
import gc
import time
import datetime
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns
import math
import warnings
import re
warnings.filterwarnings("ignore")
import sys
stdo = sys.stdout
reload(sys)
sys.setdefaultencoding('utf-8')
sys.stdout= stdo
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
import sys
from tqdm import tqdm
import collections
import random


# In[ ]:


def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")


# In[2]:


#读取数据
train = pd.read_excel("train.xlsx")
test = pd.read_excel("test.xlsx")


# In[3]:


print train.shape,test.shape


# In[4]:


train = train.reset_index(drop=True)
test = test.reset_index(drop=True)


# In[5]:


#获取全部预测特征
fe = [k for k in train.columns if k not in [u'企业编号',u'企业评分']]


# In[6]:


#设定target
target = train[u'企业评分'].values


# In[ ]:


# #参数调整
# model = lgb.LGBMRegressor(boosting_type = 'gbdt',random_state=50,
#                           objective = 'regression',subsample=0.6143)

# folds = KFold(n_splits=3, shuffle=True, random_state=1)

# params_spaces = {
#         'learning_rate': (0.01, 1.0, 'log-uniform'),
#         'num_leaves': (10, 100),      
#         'max_depth': (30, 100),
#         'min_child_samples': (0, 50),
#         'max_bin': (100, 1000),
#         'subsample_freq': (0, 10),
#         'min_child_weight': (0, 10),
#         'reg_lambda': (1e-9, 1000, 'log-uniform'),
#         'reg_alpha': (1e-9, 1.0, 'log-uniform'),
#         'scale_pos_weight': (1e-6, 500, 'log-uniform'),
#         'n_estimators': (100, 400),
#     }   

# bayes_cv_tuner = BayesSearchCV(estimator = model,
#                                search_spaces = params_spaces,
#                                scoring = 'neg_mean_squared_error',
#                                cv = folds,
#                                n_iter = 100,   
#                                verbose = 10,
#                                refit = True,
#                                random_state = 1885,
#                                return_train_score = True)

# result_lgb = bayes_cv_tuner.fit(train[fe].values, target, callback=status_print)

# bayes_cv_tuner = BayesSearchCV(
#     estimator = xgb.XGBRegressor(
#         n_jobs = -1,
#         objective = 'reg:linear',
#         eval_metric = 'rmse',
#         silent=1,
#         tree_method='approx'
#     ),
#     search_spaces = {
#         'learning_rate': (0.01, 1.0, 'log-uniform'),
#         'min_child_weight': (0, 10),
#         'max_depth': (0, 50),
#         'max_delta_step': (0, 20),
#         'subsample': (0.01, 1.0, 'uniform'),
#         'colsample_bytree': (0.01, 1.0, 'uniform'),
#         'colsample_bylevel': (0.01, 1.0, 'uniform'),
#         'reg_lambda': (1e-9, 1000, 'log-uniform'),
#         'reg_alpha': (1e-9, 1.0, 'log-uniform'),
#         'gamma': (1e-9, 0.5, 'log-uniform'),
#         'n_estimators': (100, 1000),
#         'scale_pos_weight': (1e-6, 500, 'log-uniform')
#     },    
#     scoring = 'neg_mean_squared_error',
#     cv = folds,
#     n_jobs = 3,
#     n_iter = ITERATIONS,   
#     verbose = 10,
#     refit = True,
#     random_state = 42
# )

# result_xgb = bayes_cv_tuner.fit(X, y, callback=status_print)


# In[7]:


#LGB1 -> 用于特征选择
bst_param = {'objective':'regression',"metric": 'mse',             'num_leaves': 100, 'reg_alpha': 0.1315180967852709,              'subsample_freq': 1, 'scale_pos_weight': 499.99999999999994,              'learning_rate': 0.010747386692855437, 'min_child_weight': 7,              'max_depth': 100, 'n_estimators': 400000,              'reg_lambda': 0.27304074775491205, 'max_bin': 1000,             'min_child_samples': 50}
X = train[fe].values
y = target
folds = KFold(n_splits=10, shuffle=True, random_state=2018)
oof_lgb = np.zeros(len(train))
use_col = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
    val_data = lgb.Dataset(X[val_idx], y[val_idx])

    num_round = 10000
    clf = lgb.train(bst_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
    oof_lgb[val_idx] = clf.predict(X[val_idx], num_iteration=clf.best_iteration)
    fimpot = clf.feature_importance(importance_type='split')
    fimpot_dict = dict(zip(fe, fimpot))
    use_col.extend([k for k in fimpot_dict if fimpot_dict[k] > 0])

print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))


# In[9]:


#选择的特征
select_fe = [k for k in fe if collections.Counter(use_col)[k] >= 7]
print("former: ",len(fe),"after: ",len(select_fe))


# In[11]:


#均分训练集，stacking
train_X,val_X, train_y, val_y = train_test_split(train,target,test_size = 0.5,random_state = 0) 


# In[16]:


# xgb1 -> 用于获取xgb_res
bst_xgb_params = {'objective': 'reg:linear', 'eval_metric': 'rmse','silent': True,
                 'reg_alpha': 0.0077767594036083224, 'colsample_bytree': 0.9897159431480177, 
                 'colsample_bylevel': 0.74211565058812201, 'scale_pos_weight': 0.12388047721433228, 
                 'learning_rate': 0.023685277928811567, 'max_delta_step': 16, 'min_child_weight': 5, 
                 'n_estimators': 579, 'subsample': 0.55530850885179606,'reg_lambda': 0.0083831006990190519, 
                 'max_depth': 19, 'gamma': 2.2921115876217839e-08}
    
X = train_X[select_fe].values
y = train_y
folds = KFold(n_splits=5, shuffle=True, random_state=2017)
xgb_res = np.zeros(len(val_X))
xgb_res_test = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X[trn_idx], y[trn_idx])
    val_data = xgb.DMatrix(X[val_idx], y[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    xgb1 = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=bst_xgb_params)
    xgb_res += xgb1.predict(xgb.DMatrix(val_X[select_fe].values), ntree_limit=xgb1.best_ntree_limit) / 5.0
    xgb_res_test  +=  xgb1.predict(xgb.DMatrix(test[select_fe].values), ntree_limit=xgb1.best_ntree_limit) / 5.0
    


# In[19]:


#接回原始数据
val_X['xgb_res'] = xgb_res
test['xgb_res'] = xgb_res_test


# In[21]:


#LGB2 -> 用于stacking
bst_lgb_param = {'objective':'regression',"metric": 'mse',             'num_leaves': 100, 'reg_alpha': 0.1315180967852709,              'subsample_freq': 1, 'scale_pos_weight': 499.99999999999994,              'learning_rate': 0.010747386692855437, 'min_child_weight': 7,              'max_depth': 100, 'n_estimators': 400000,              'reg_lambda': 0.27304074775491205, 'max_bin': 1000,             'min_child_samples': 50}

X = train_X[select_fe].values
y = train_y
folds = KFold(n_splits=5, shuffle=True, random_state=2016)
lgb_res = np.zeros(len(val_X))
lgb_res_test = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
    val_data = lgb.Dataset(X[val_idx], y[val_idx])
    num_round = 10000
    lgb1 = lgb.train(bst_lgb_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    lgb_res += lgb1.predict(val_X[select_fe].values, num_iteration=lgb1.best_iteration) / 5.0
    lgb_res_test += lgb1.predict(test[select_fe].values, num_iteration=lgb1.best_iteration) / 5.0


# In[22]:


#接回原始数据
val_X['lgb_res'] = lgb_res
test['lgb_res'] = lgb_res_test


# In[23]:


new_fe = select_fe + ['lgb_res','xgb_res']


# In[25]:


# xgb2 -> 用于predict
bst_xgb_params = {'objective': 'reg:linear', 'eval_metric': 'rmse','silent': True,
                 'reg_alpha': 0.0077767594036083224, 'colsample_bytree': 0.9897159431480177, 
                 'colsample_bylevel': 0.74211565058812201, 'scale_pos_weight': 0.12388047721433228, 
                 'learning_rate': 0.023685277928811567, 'max_delta_step': 16, 'min_child_weight': 5, 
                 'n_estimators': 579, 'subsample': 0.55530850885179606,'reg_lambda': 0.0083831006990190519, 
                 'max_depth': 19, 'gamma': 2.2921115876217839e-08}
    
X = val_X[new_fe].values
y = val_y
folds = KFold(n_splits=5, shuffle=True, random_state=2015)
pre_xgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X[trn_idx], y[trn_idx])
    val_data = xgb.DMatrix(X[val_idx], y[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    xgb2 = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=bst_xgb_params)
    pre_xgb += xgb2.predict(xgb.DMatrix(test[new_fe].values), ntree_limit=xgb2.best_ntree_limit) / 5.0
    


# In[27]:


#LGB3 -> 用于predict
bst_lgb_param = {'objective':'regression',"metric": 'mse',             'num_leaves': 100, 'reg_alpha': 0.1315180967852709,              'subsample_freq': 1, 'scale_pos_weight': 499.99999999999994,              'learning_rate': 0.010747386692855437, 'min_child_weight': 7,              'max_depth': 100, 'n_estimators': 400000,              'reg_lambda': 0.27304074775491205, 'max_bin': 1000,             'min_child_samples': 50}

X = val_X[new_fe].values
y = val_y
folds = KFold(n_splits=5, shuffle=True, random_state=2014)
pre_lgb = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
    val_data = lgb.Dataset(X[val_idx], y[val_idx])
    num_round = 10000
    lgb2 = lgb.train(bst_lgb_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    pre_lgb += lgb2.predict(test[new_fe].values, num_iteration=lgb2.best_iteration) / 5.0


# In[28]:


#全数据集的xgb
bst_xgb_params = {'objective': 'reg:linear', 'eval_metric': 'rmse','silent': True,
                 'reg_alpha': 0.0077767594036083224, 'colsample_bytree': 0.9897159431480177, 
                 'colsample_bylevel': 0.74211565058812201, 'scale_pos_weight': 0.12388047721433228, 
                 'learning_rate': 0.023685277928811567, 'max_delta_step': 16, 'min_child_weight': 5, 
                 'n_estimators': 579, 'subsample': 0.55530850885179606,'reg_lambda': 0.0083831006990190519, 
                 'max_depth': 19, 'gamma': 2.2921115876217839e-08}
    
X = train[select_fe].values
y = target
folds = KFold(n_splits=5, shuffle=True, random_state=2013)
pre_xgb_full = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print("fold n°{}".format(fold_+1))
    trn_data = xgb.DMatrix(X[trn_idx], y[trn_idx])
    val_data = xgb.DMatrix(X[val_idx], y[val_idx])

    watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
    xgb3 = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=bst_xgb_params)
    pre_xgb_full += xgb3.predict(xgb.DMatrix(test[select_fe].values), ntree_limit=xgb3.best_ntree_limit) / 5.0
    


# In[29]:


#全数据集的lgb
bst_lgb_param = {'objective':'regression',"metric": 'mse',             'num_leaves': 100, 'reg_alpha': 0.1315180967852709,              'subsample_freq': 1, 'scale_pos_weight': 499.99999999999994,              'learning_rate': 0.010747386692855437, 'min_child_weight': 7,              'max_depth': 100, 'n_estimators': 400000,              'reg_lambda': 0.27304074775491205, 'max_bin': 1000,             'min_child_samples': 50}

X = train[select_fe].values
y = target
folds = KFold(n_splits=5, shuffle=True, random_state=2012)
pre_lgb_full = np.zeros(len(test))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print("fold n°{}".format(fold_+1))
    trn_data = lgb.Dataset(X[trn_idx], y[trn_idx])
    val_data = lgb.Dataset(X[val_idx], y[val_idx])
    num_round = 10000
    lgb3 = lgb.train(bst_lgb_param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    pre_lgb_full += lgb3.predict(test[select_fe].values, num_iteration=lgb3.best_iteration) / 5.0


# In[35]:


sub = (pre_lgb + pre_lgb_full + pre_xgb + pre_xgb_full) / 4.0


# In[44]:


test['sub'] = sub


# In[48]:


test[[u'企业编号','sub']].to_excel("赛题1结果_SUIBIANDA.xlsx",header = 0,index = 0)

