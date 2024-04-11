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

##############################################################  
##------------------------------------------------------------
###############################################################


# ylen is there to keep track of the true unpadded length (time steps) of each sample
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x_static, x_dynamic, x_mask, x_delta, y, y_aux, maxsteps, ids):
        self.x_static = x_static
        self.x_dynamic = x_dynamic
        self.x_mask = x_mask
        self.x_delta = x_delta
        self.y = y
        self.y_aux = y_aux
        self.maxsteps = maxsteps
        self.ids = ids
    def __len__(self):
        return len(self.x_static)

    def __getitem__(self, idx):
        return {
            'x_dynamic': self.x_dynamic[idx,:,:],
            'x_static': self.x_static[idx,:],
            'y': self.y[idx,:],
            'y_aux': self.y_aux[idx,:,:],
            'x_mask': self.x_mask[idx,:,:],
            'x_delta': self.x_delta[idx,:,:],
            'maxsteps': self.maxsteps[idx],
            'ids':self.ids[idx]
        }
    
##------------------------------------------------------------------------

# from https://github.com/NIPS-BRITS/BRITS/blob/master/models/rits.py
class TemporalDecay(nn.Module):
    def __init__(self, input_size, rnn_hid_size):
        super(TemporalDecay, self).__init__()
        self.rnn_hid_size = rnn_hid_size
        self.build(input_size)

    def build(self, input_size):
        self.W = torch.nn.Parameter(torch.Tensor(self.rnn_hid_size, input_size))
        self.b = torch.nn.Parameter(torch.Tensor(self.rnn_hid_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma
    
#-----------------------------------------------------------------------


def BuildSequential(sizes, p_dropout, size_input):
    ret = []
    for i in range(len(sizes)):
        if i==0:
            cur_size_input = size_input
        else:
            cur_size_input = sizes[i-1]
        ret.append(torch.nn.Linear(cur_size_input, sizes[i]))
        ret.append(torch.nn.Dropout(p=p_dropout))
        ret.append(torch.nn.ReLU())
    return torch.nn.Sequential(*ret)




class BaselineModel(nn.Module):
    def __init__(self,n_static,n_dynamic, n_rnn_hidden = 2**8, n_rnn_layers = 3, p_dropout = 0.50, ns_linear=[2**8,2**7,2**6]):
        super(type(self),self).__init__()
        self.n_static = n_static
        self.n_dynamic = n_dynamic
        self.n_rnn_hidden = n_rnn_hidden
        self.n_rnn_layers = n_rnn_layers
        self.p_dropout = p_dropout
        self.ns_linear = ns_linear
        
        self.layer_rnn = torch.nn.LSTM(input_size = self.n_dynamic, hidden_size = self.n_rnn_hidden, num_layers = self.n_rnn_layers, dropout = self.p_dropout, batch_first = True)
        self.layers_linear = BuildSequential(self.ns_linear, self.p_dropout, self.n_rnn_hidden + self.n_static)
        self.layer_final = torch.nn.Linear(self.ns_linear[-1],1)
        
        
    def forward(self,batch):
        MAXSTEP = batch['x_dynamic'].shape[1]
        # Initialize RNN
        rnn_h = torch.zeros(self.n_rnn_layers, batch['x_dynamic'].shape[0], self.n_rnn_hidden, device = device)
        rnn_c = torch.zeros(self.n_rnn_layers, batch['x_dynamic'].shape[0], self.n_rnn_hidden, device = device)
        # RNN step
        x_dynamic, (rnn_h,rnn_c)  = self.layer_rnn(batch['x_dynamic'], (rnn_h,rnn_c))
        # Cat
        x = torch.cat([x_dynamic, batch['x_static'].unsqueeze(1).repeat(1, MAXSTEP, 1)],dim=2)
        # Linear
        y_proba = self.layer_final(self.layers_linear(x))
        return {'y_proba':y_proba}
    
    
    
    
class Model(nn.Module):
    def __init__(self, n_static, n_dynamic, n_y_aux, 
                 n_impute_hidden = 2**8, n_impute_linear = 2**8, n_impute_rnn = 2**8,
                 n_rnn_hidden = 2**8, n_rnn_layers = 3, p_dropout = 0.50,
                 ns_aux = [2**8, 2**8, 2**8], ns_att = [2**8, 2**8, 2**8], idx_exp_static=[], idx_exp_dynamic=[]):
        super(type(self),self).__init__()
        self.n_static = n_static
        self.n_dynamic = n_dynamic
        self.n_y_aux = n_y_aux
        self.n_explanation_static = len(idx_exp_static)
        self.n_explanation_dynamic = len(idx_exp_dynamic)
        self.n_impute_hidden = n_impute_hidden
        self.n_impute_linear = n_impute_linear
        self.n_impute_rnn = n_impute_rnn
        self.n_rnn_hidden = n_rnn_hidden
        self.n_rnn_layers = n_rnn_layers
        self.p_dropout  = p_dropout 
        self.ns_aux = ns_aux
        self.ns_att = ns_att
        self.idx_exp_static = idx_exp_static
        self.idx_exp_dynamic = idx_exp_dynamic
        
        # Imputation layers (dynamic only)
        self.layer_impute_linear = torch.nn.Linear(self.n_impute_hidden + self.n_dynamic, self.n_dynamic)
        self.layer_impute_gamma = TemporalDecay(self.n_dynamic, self.n_impute_hidden)
        self.layer_impute_rnn = torch.nn.LSTMCell(self.n_dynamic*2, self.n_impute_hidden, bias=True)
        
        # Recurrent layers
        self.layer_rnn = torch.nn.LSTM(input_size = self.n_dynamic, hidden_size = self.n_rnn_hidden, num_layers = self.n_rnn_layers, dropout = self.p_dropout, batch_first = True)
        
        # Auxillery
        self.layers_aux = BuildSequential(self.ns_aux, self.p_dropout, self.n_rnn_hidden + self.n_static)
        self.layer_aux = torch.nn.Linear(self.ns_aux[-1], self.n_y_aux)

        # Attention
        self.layers_att = BuildSequential(self.ns_att, self.p_dropout, self.n_rnn_hidden + self.n_static)
        self.layer_att = torch.nn.Linear(self.ns_att[-1], self.n_y_aux + self.n_explanation_static + self.n_explanation_dynamic)

        # Final layer
        self.layer_final = torch.nn.Linear(self.n_y_aux + self.n_explanation_static + self.n_explanation_dynamic, 1)
        
        
    def forward(self, batch):
        # define maximum number of timesteps
        MAXSTEP = batch['x_dynamic'].shape[1]
        
        # Tensor for predicted x_dynamic
        x_h = torch.zeros_like(batch['x_dynamic'], device=device)
        
        # Tensor for imputed x_dynamic (x_dynamic with missing values filled in with predicted x_dynamic)
        x_c = torch.zeros_like(batch['x_dynamic'], device=device)
        
        # Initialize RNN for imputation
        impute_h = torch.zeros(batch['x_dynamic'].shape[0], self.n_impute_hidden, device = device)
        impute_c = torch.zeros(batch['x_dynamic'].shape[0], self.n_impute_hidden, device = device)
        
        # Iterate through timesteps and impute
        for t in range(MAXSTEP):
            # Slice off timestep
            x_slice = batch['x_dynamic'][:,t,:]
            m_slice = batch['x_mask'][:,t,:]
            d_slice = batch['x_delta'][:,t,:]
            
            # Temporal decay
            gamma = self.layer_impute_gamma(d_slice)
            
            # Update hidden
            impute_h = impute_h * gamma
            
            # Predict x based on x (which has been forward filled and represents most recent value) and hidden
            impute_input = torch.cat([x_slice, impute_h],dim=1)
            x_h_slice = self.layer_impute_linear(impute_input)
            
            # Populate x_h
            x_h[:,t,:] = x_h_slice
            
            # Fill in missing x with x_h
            x_c_slice = (m_slice * x_slice) + ((1 - m_slice) * x_h_slice)
            
            # Populate x_c
            x_c[:,t,:] = x_c_slice
                        
            # RNN step
            rnn_input = torch.cat([x_c_slice, m_slice],dim=1)            
            impute_h, impute_c = self.layer_impute_rnn(rnn_input, (impute_h, impute_c))
        
        # Initialize RNN
        rnn_h = torch.zeros(self.n_rnn_layers, batch['x_dynamic'].shape[0], self.n_rnn_hidden, device = device)
        rnn_c = torch.zeros(self.n_rnn_layers, batch['x_dynamic'].shape[0], self.n_rnn_hidden, device = device)
        
        # RNN step
        x_dynamic, (rnn_h,rnn_c)  = self.layer_rnn(x_c, (rnn_h,rnn_c))
        
        # Append static
        x_dynamic = torch.cat([x_dynamic,batch['x_static'].unsqueeze(1).repeat(1, MAXSTEP, 1)],dim=2)

        # Auxillery
        y_aux_proba = torch.sigmoid(self.layer_aux(self.layers_aux(x_dynamic))) # (0, 1)
        
        # Attention
        att = torch.clamp(self.layer_att(self.layers_att(x_dynamic)),min = 0) # (0, inf)
        
        # Explanations (Auxillery + Static + Dynamic)
        exp = torch.cat([y_aux_proba, 
                         batch['x_static'][:,self.idx_exp_static].unsqueeze(1).repeat(1, MAXSTEP, 1), 
                         batch['x_dynamic'][:,:,self.idx_exp_dynamic]], dim=2)
        
        # Final
        y_proba = torch.sum(exp.detach() * att, dim=2, keepdim=True) # (0, inf)
        #y_proba = torch.clamp(y_proba, min=0, max=1) 
        y_proba = 1-1/(torch.exp(y_proba)) # (0, 1)
        #y_proba = torch.log(torch.exp(y_proba+1e-15)-1) # (-inf, inf)
        
        return {'x_h':x_h, 'x_c':x_c, 'y_aux_proba':y_aux_proba, 'att':att, 'y_proba':y_proba, 'exp':exp}       
    
    
#################################################################
#################################################################


# LOSSES

def L_pri(logit=False, **kwargs):
    if logit:
        func = torch.nn.BCEWithLogitsLoss
    else:
        func = torch.nn.BCELoss
    func = func(reduction='none', **kwargs)
    def L_pri_inner(batch):
        # Create mask based on maxsteps
        mask = torch.zeros(batch['y'].shape, device=device)
        for i in range(batch['y'].shape[0]):
            mask[i, :batch['maxsteps'][i]] = 1
        unmaskedloss = func(batch['y_proba'],batch['y'])
        maskedloss = unmaskedloss * mask
        pixels = mask.sum()
        return maskedloss.sum()/pixels
    return L_pri_inner



def L_aux(batch):
    # Create mask based on maxsteps
    mask = torch.zeros(batch['y_aux'].shape, device=device)
    for i in range(batch['y_aux'].shape[0]):
        mask[i, :batch['maxsteps'][i], :] = 1
    unmaskedloss = torch.nn.MSELoss(reduction='none')(batch['y_aux_proba'],batch['y_aux'])
    maskedloss = unmaskedloss * mask
    pixels = mask.sum()
    return maskedloss.sum()/pixels



def L_imp(batch):
    # grab imputation weight (we care more about imputing rare values, and non-misssing variables shouldnt count)
    #imputeweight = []
    #for idx in range(x_mask.shape[0]):
    #    imputeweight.append(x_mask[idx,:maxsteps[idx],:])
    #imputeweight = torch.cat(imputeweight)
    #imputeweight = (1/imputeweight.mean(dim=0))**2
    #imputeweight = imputeweight.repeat((x.shape[0],x.shape[1],1))
    
    # create mask based on timesteps
    timestepmask = torch.zeros(batch['x_dynamic'].shape, device=device)
    for i in range(batch['x_dynamic'].shape[0]):
        timestepmask[i, :batch['maxsteps'][i], :] = 1
    unmaskedloss = torch.nn.MSELoss(reduction='none')(batch['x_dynamic'],batch['x_h'])
    maskedloss = unmaskedloss * batch['x_mask'] * timestepmask
    pixels = (batch['x_mask'] * timestepmask).sum()          
    return maskedloss.sum()/pixels

LossFunc = {
    'L_pri':L_pri,
    'L_aux':L_aux,
    'L_imp':L_imp,
}


