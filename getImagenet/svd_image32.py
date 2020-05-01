#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle


# In[2]:
import DistributedOI.data_prep as dp

#dp.zeromean_ImageNet32_data()
dp.SVD_ImageNet32_data(40,1024)
