#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("D:/simple-vqa-pylib-master/use_keras")


# In[2]:


from keras.callbacks import ModelCheckpoint
import argparse
from model import build_model
from prepare_data import setup
#pip install easydict
#pip install keras
#pip install tensorflow


# In[3]:


# Support command-line options
'''parser = argparse.ArgumentParser()
parser.add_argument('--big-model', action='store_true', help='Use the bigger model with more conv layers')
parser.add_argument('--use-data-dir', action='store_true', help='Use custom data directory, at /data')
args = parser.parse_args()'''
import easydict
args = easydict.EasyDict({"big_model":False,
                   "use_data_dir":True})


# In[4]:


if args.big_model:
  print('Using big model')
if args.use_data_dir:
  print('Using data directory')


# In[5]:



# Prepare data
train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, _, _, _ = setup(args.use_data_dir)

# Data cardinality is ambiguous 에러 해결
import numpy as np
train_X_ims = np.array(train_X_ims) 
test_X_ims = np.array(test_X_ims)


# In[11]:



print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers, args.big_model)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

print('\n--- Training model...')
model.fit(
  [train_X_ims, train_X_seqs],
  train_Y,
  validation_data=([test_X_ims, test_X_seqs], test_Y),
  shuffle=True,
  epochs=8,
  callbacks=[checkpoint],
)


# In[12]:


model.load_weights("./model.h5")
predictions = model.predict([test_X_ims, test_X_seqs])


# In[14]:


# 정답 확인
predictions[0]

