from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib_venn as venn
import sys
import os
import time
import xgboost
import math
import shap
import matplotlib.dates as mdates
import sklearn
import seaborn as sns
import i2bmi

from utils_data_extraction import *

import torch
from torch import nn, optim
import torch.nn.functional as F
import math
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# cuda settings
print(torch.cuda.is_available())
print ('Available devices ', torch.cuda.device_count())
#print ('Current cuda device ', torch.cuda.current_device())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

import warnings
warnings.filterwarnings("ignore")

#######################################################
########################################################

opath = './output'
npath = opath
SAVENAME = 'FULL'


data={}
data['Full']={}
for fname in os.listdir(opath):
    if fname.endswith('.pickle') and 'cols' not in fname:
        print(fname)
        data['Full'][fname.split('.pickle')[0]] = pd.read_pickle(os.path.join(opath,fname))
        
        
        
# sanity check for object columns
for fname in data['Full']:
    if isinstance(data['Full'][fname], pd.DataFrame):
        for col in data['Full'][fname]:
            if data['Full'][fname][col].dtype=='object':
                print('{}, {}'.format(fname, col))
                
                
                
for i in np.arange(10,16):
    print('2**{} =  {}'.format(i,2**i))
    
print('\n2**{:.1f} total number of unique patients'.format(np.log(data['Full']['y'].index.get_level_values('id').nunique())/np.log(2)))

print('2**{:.1f} is the total number of positive patients'.format(np.log(data['Full']['y'].groupby('id').max().sum())/np.log(2)))

##########################################################
# SUBSET

anypos = data['Full']['y'].groupby('id').max()
posidx = anypos[anypos==True].index
negidx = anypos[anypos==False].index

# ALL POS
#NEGRATIO = 5
#negidx = np.random.choice(negidx, posidx.shape[0] * NEGRATIO, replace=False)
#print(len(posidx))
#print(len(negidx))
#idsubset = list(posidx)+list(negidx)

# SUBSET
#POSSIZE = 2**10
#NEGRATIO = 1
#posidx = np.random.choice(posidx, POSSIZE, replace=False)
#negidx = np.random.choice(negidx, posidx.shape[0] * NEGRATIO, replace=False)
#print(len(posidx))
#print(len(negidx))
#idsubset = list(posidx)+list(negidx)

# FULL
idsubset = list(data['Full']['y'].index.get_level_values('id').unique())
print(len(idsubset))


# split
for df in data['Full']:
    print(df)
    if len(data['Full'][df].index.names)==1:
        data['Full'][df] = data['Full'][df].loc[data['Full'][df].index.isin(idsubset),:].copy()
    else:
        data['Full'][df] = data['Full'][df].loc[data['Full'][df].index.get_level_values('id').isin(idsubset),:].copy()
        
        
        
print('Positive % per Encounter {:.1%}'.format(data['Full']['y'].groupby('id').max().mean()))
print('Positive % per Timestep  {:.1%}'.format(data['Full']['y'].mean()))


##########################################################
####------- Train test validation split ----------------
#########################################################

# First split into Train vs NonTrain
ids={}
ids['Full'] = data['Full']['y'].groupby('id').max()
ids['Train'], ids['NonTrain'] = sklearn.model_selection.train_test_split(ids['Full'].index, train_size=0.5,shuffle=True,stratify=ids['Full'].values)
ids['Train'] =    ids['Full'][ids['Train']]
ids['NonTrain'] = ids['Full'][ids['NonTrain']]

# Then split NonTrain into Test and Val
ids['Test'], ids['Val'] = sklearn.model_selection.train_test_split(ids['NonTrain'].index, train_size=0.5,shuffle=True,stratify=ids['NonTrain'].values)
ids['Test'] = ids['NonTrain'][ids['Test']]
ids['Val']  = ids['NonTrain'][ids['Val']]

for split in ids:
    ids[split] = ids[split].sort_index()

# Sanity check
for split in ['Full','Train','Test','Val']:
    print('{:>6}: n = {:>6,}, Pos = {:>5.2%}'.format(split, ids[split].shape[0],ids[split].mean()))
    
    


# assign into splits
for split in ['Train','Test','Val']:
    # create empty dict per split
    data[split]={}
    # for each full df assign subset into each split
    for df in data['Full']:
        data['Full'][df] = data['Full'][df].sort_index()
        if len(data['Full'][df].index.names)==1:
            data[split][df] = data['Full'][df].loc[data['Full'][df].index.isin(ids[split].index),:].copy()
        else:
            data[split][df] = data['Full'][df].loc[data['Full'][df].index.get_level_values('id').isin(ids[split].index),:].copy()

            
            
            
# scale data based on train

for df in data['Full']:
    if df!='y':
        print(df)
        for split in ['Train','Test','Val']:
            # identify num vs bool columns (there shouldnt be any others)
            cols_num = data[split][df].select_dtypes('number').columns
            cols_SOFA = [i for i in data[split][df] if 'SOFA' in i]
            cols_num = [i for i in cols_num if i not in cols_SOFA]
            cols_bool = data[split][df].select_dtypes('bool').columns
            
            if len(cols_num)>0: # (0,1)
                if split=='Train':
                    scaler_num = sklearn.preprocessing.QuantileTransformer()
                    scaler_num.fit(data[split][df].loc[:,cols_num])
                data[split][df].loc[:,cols_num] = scaler_num.transform(data[split][df].loc[:,cols_num])
                
            if len(cols_bool)>0: # (0,1)
                 data[split][df].loc[:,cols_bool] = data[split][df].loc[:,cols_bool]*1
            
            if len(cols_SOFA)>0: # (0,1)
                if split=='Train':
                    scaler_SOFA = sklearn.preprocessing.MinMaxScaler(feature_range = (0,1))
                    scaler_SOFA.fit(data[split][df].loc[:,cols_SOFA])
                data[split][df].loc[:,cols_SOFA] = scaler_SOFA.transform(data[split][df].loc[:,cols_SOFA])
                
                
                
                
                
# generate mask and delta for missing value imputation

for split in ['Train','Test','Val']:
    print(split)
    # create mask
    data[split]['x_mask'] = data[split]['x_dynamic'].notnull()*1
    data[split]['x_delta'] = data[split]['x_mask'].copy()
    # ones-only pd.series for creating delta
    ones = pd.Series(1,index=data[split]['x_dynamic'].index)
    for col in data[split]['x_mask'].columns:
        temp = data[split]['x_mask'][col]
        # trick from https://stackoverflow.com/questions/53126246/cumulative-sum-of-a-dataframe-column-with-restart
        # basically, create a "group" for each consecutive run of non/missing-measurements, then groupby on those groups and cumsum + minor manipulation
        temp = ones.groupby([temp.index.get_level_values('id'),temp.groupby(level='id').cumsum()]).cumsum().groupby([temp.index.get_level_values('id')]).shift().fillna(0)
        data[split]['x_delta'][col] = temp
        
        
        
        
# ffill and mean-impute 
for split in ['Train','Test','Val']:
    print(split)
    for df in data[split]:
        print('\t{:>10}: {}'.format(df, data[split][df].isnull().any(axis=0).any()))
        if len(data[split][df].index.names)==1:
            data[split][df] = data[split][df].fillna(0.)
        else:
            data[split][df] = data[split][df].groupby('id').ffill().fillna(0.)
            
            
#############################################################            
            
# helper function for converting dataframe to torch
def df2tensor(x):
    return torch.from_numpy(x.values).float()

def df2paddedtensor(x):
    return torch.nn.utils.rnn.pad_sequence(x.groupby(level='id').apply(lambda x:df2tensor(x)).to_list(),batch_first=True,padding_value=0.0)

for split in ['Train','Test','Val']:
    print(split)
    data[split]['maxsteps'] = torch.from_numpy(data[split]['y'].groupby('id').size().values).int()
    for df in data[split]:
        if df!='maxsteps':
            print('\t{}'.format(df))
            if len(data[split][df].index.names)==1:
                data[split][df] = df2tensor(data[split][df])
            else:
                data[split][df] = df2paddedtensor(data[split][df])
                
                
                
for split in ['Train','Test','Val']:
    data[split]['y'] = data[split]['y'].unsqueeze(dim=2)
    
data['Full']['ids'] = ids

torch.save(data, os.path.join(npath,'{}.pt'.format(SAVENAME)))


for split in data:
    print(split)
    for df in data[split]:
        if type(data[split][df]) in [pd.core.frame.DataFrame,torch.Tensor]:
            print('\t{:>10}: {}'.format(df,data[split][df].shape))
            
            
#### Distributions-------            
# for split in ['Train','Test','Val']:
#     print(split)
#     for df in ['x_static']:
#         print(df)
#         display(pd.DataFrame(data[split][df].numpy()).agg(['min','max','mean','std']))
#     for df in ['x_dynamic','y_aux']:
#         print(df)
#         display(pd.DataFrame(data[split][df].numpy().reshape((-1,data[split][df].shape[2]))).agg(['min','max','mean','std']))

    