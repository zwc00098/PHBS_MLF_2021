#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import arff
import pandas as pd
import numpy as np


# In[2]:


data=arff.loadarff('4year.arff')
df=pd.DataFrame(data[0])


# In[3]:


df.head()


# In[4]:


df['bankruptcy']=(df['class']==b'1')
df.head()


# In[5]:


df=pd.concat([df['Attr1'],df['Attr2'],df['Attr7'],df['Attr10'],df['bankruptcy']],axis=1)
df.head()


# In[6]:


df.columns=['X1','X2','X7','X10','bankruptcy']
df=df.fillna(df.mean())
df.head()


# In[7]:


df_all=df
df_all=df_all.drop('bankruptcy',axis=1)
print('Mean of the 4 features among all companies:','\n',df_all.std())
print('Std of the 4 features among all companies:','\n',df_all.mean())


# In[8]:


df_operating=df[df['bankruptcy']==False]
df_operating=df_operating.drop('bankruptcy',axis=1)
print('Mean of the 4 features among operating companies:','\n',df_operating.std())
print('Std of the 4 features among operating companies:','\n',df_operating.mean())


# In[9]:


df_bankrupt=df[df['bankruptcy']==True]
df_bankrupt=df_bankrupt.drop('bankruptcy',axis=1)
print('Mean of the 4 features among bankrupt companies:','\n',df_bankrupt.std())
print('Std of the 4 features among bankrupt companies:','\n',df_bankrupt.mean())


# In[10]:


df_satisfied=df[(df['X1']<np.mean(df['X1'])-np.std(df['X1']))&(df['X10']<np.mean(df['X10'])-np.std(df['X10']))]
print(len(df_satisfied))


# In[11]:


df_satisfied_bankrupt=df_satisfied[df_satisfied['bankruptcy']==True]
print(len(df_satisfied_bankrupt)/len(df_satisfied))


# In[ ]:




