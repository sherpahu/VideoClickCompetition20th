
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np

# In[6]:


import matplotlib.pylab as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth',50)
pd.set_option('display.width',50)
pd.set_option('display.max_info_rows',50)


# # app

# In[181]:


app = pd.read_csv('./dataset/app.csv')


# In[113]:


app.head()


# In[114]:


app.info(verbose=True,null_counts=True)


# In[115]:


import re
applist = []
for x in app['applist']:
    temp = [int(s) for s in re.findall(r'\d+\.?\d*', x)]
    applist  = list(set(applist+temp))
print(applist)


# In[154]:


# 不同app总数：app1-app25730
print(len(applist))


# # user

# In[180]:


user = pd.read_csv('./dataset/user.csv')


# In[158]:


user.head()


# In[159]:


user.info(verbose=True,null_counts=True)


# In[163]:


print(user['gender'].value_counts())


# In[164]:


# user['personalscore'].value_counts()
sns.distplot(user['personalscore'].dropna(axis=0))


# In[165]:


# user['followscore'].value_counts()
sns.distplot(user['followscore'].dropna())


# In[167]:


print(user['personidentification'].value_counts())
# 1表示劣质用户 0表示正常用户


# In[168]:


print(user['level'].value_counts())


# In[169]:


user[user['guid']==user['deviceid']]


# # train & test

# In[7]:


train = pd.read_csv('./dataset/train.csv')


# In[173]:


train.head()


# In[10]:


test = pd.read_csv('./dataset/test.csv')


# In[175]:


test.head()


# In[176]:


data = pd.concat([train, test], ignore_index=True)


# In[177]:


train.info(verbose=True,null_counts=True)


# In[178]:


test.info(verbose=True,null_counts=True)


# In[183]:


print(train['deviceid'].nunique())
print(test['deviceid'].nunique())
print(data['deviceid'].nunique())
print(app['deviceid'].nunique())
print(user['deviceid'].nunique())


# In[184]:


print(train.columns.to_list())


# In[186]:


print(train['pos'].value_counts())


# In[187]:


print(train['netmodel'].value_counts())


# In[188]:


app_version_map = dict(zip(train['app_version'].value_counts().sort_index().index.to_list(), range(train['app_version'].nunique())))


# In[189]:


plt.hist(train['app_version'].map(app_version_map), bins=train['app_version'].nunique())[2]
# plt.yscale('log')
plt.show()
sns.distplot(train['app_version'].map(app_version_map), bins=train['app_version'].nunique())


# In[208]:


print(train['device_vendor'].value_counts())


# In[193]:


# train['osversion'].value_counts()
osversion_map = dict(zip(train['osversion'].value_counts().sort_values().index.to_list(), range(train['osversion'].nunique())))
plt.hist(train['osversion'].map(osversion_map), bins=train['osversion'].nunique())[2]
plt.yscale('log')
plt.show()
sns.distplot(train['osversion'].map(osversion_map), bins=train['osversion'].nunique())


# In[209]:


print(train['device_version'].value_counts())


# In[210]:


print(train['ts'].value_counts())


# In[11]:


import datetime
print(train['ts'].map(lambda x: datetime.datetime.fromtimestamp(x/1000)).sort_values())
print(test['ts'].map(lambda x: datetime.datetime.fromtimestamp(x/1000)).sort_values())


# In[199]:


sns.distplot(train['ts'])


# In[200]:


sns.distplot(test['ts'])


# In[211]:


print(train['target'].value_counts())


# In[18]:


print(train['timestamp'].dropna().map(lambda x: datetime.datetime.fromtimestamp(x/1000)).sort_values())


# In[213]:


sns.distplot(train['timestamp'].dropna())

