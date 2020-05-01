#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import pickle


# In[2]:


rootdir = os.getcwd()+"/train32"
file_idx=0 
i=0
file_size=5000
data=np.zeros((32*32*3,file_size))

for subdir, dirs, files in os.walk(rootdir):

    for file in files:
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            f=os.path.join(subdir, filename)
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            flat=img.flatten()
            if len(flat)==32*32*3:
            	data[:,i]=flat
            	i+=1

            if i==file_size:
                print(f,file_idx)
                with open('ImageNet32/data{}.pickle'.format(file_idx), 'wb') as handle:
                    pickle.dump(data, handle)
                i=0
           
                file_idx+=1
                
                
            continue
        else:
            continue
            
with open('ImageNet32/data{}.pickle'.format(file_idx), 'wb') as handle:
    pickle.dump(data[:,:i], handle)            

print(i)



