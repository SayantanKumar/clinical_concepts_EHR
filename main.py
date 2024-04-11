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
from model import *
from data_splitter import *
from train_evaluate import *

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


projname = 'PlausibleExplanations'
pt = 'FULL.pt'
opath = './output'
npath = os.path.join(opath,projname)
os.makedirs(npath,exist_ok=True)

data = torch.load(os.path.join(npath, 'FULL.pt'))

for split in data:
    print(split)
    for df in data[split]:
        if type(data[split][df]) in [pd.core.frame.DataFrame,torch.Tensor]:
            print('\t{:>10}: {}'.format(df,data[split][df].shape))
            
            
for split in ['Train','Test','Val']:
    print(split)
    data[split]['ids'] = torch.Tensor(data['Full']['ids'][split].index)
    print('\t{:>10}: {}'.format('ids',data[split]['ids'].shape))
    
    
###############################################################
#---------- Training model --------------------
###############################################################

defaultdim = 2**6
param_model = {
    'n_impute_hidden': defaultdim,
    'n_impute_linear': defaultdim,
    'n_impute_rnn': defaultdim,
    'n_rnn_hidden': defaultdim,
    'n_rnn_layers': 1,
    'p_dropout': 0.50,
    'ns_aux': [defaultdim],
    'ns_att': [defaultdim],
}
param_optim = {'lr':3e-5,'betas':(0.9, 0.999), 'weight_decay':0, 'amsgrad':False}
param_training = {'epochs':501,'batch_size':2**8}
print(f"Batch size: {param_training['batch_size']}, train batches per epoch: {data['Train']['x_dynamic'].shape[0]/param_training['batch_size']:.0f}, val batches per epoch: {data['Val']['x_dynamic'].shape[0]/param_training['batch_size']:.0f}")

param_ES = {
    'epochperval':1, # Check validation performance every X epochs
    'patience': 50,
    'miniter': 10,
}

param_loss = {'Primary':{'Func':L_pri(logit=False),'Weight':1},'Auxiliary':{'Func':L_aux,'Weight':1},'Imputation':{'Func':L_imp,'Weight':.1}}

model = Model(data['Train']['x_static'].shape[1], data['Train']['x_dynamic'].shape[2],  data['Train']['y_aux'].shape[2],
                      idx_exp_static=list(data['Full']['x_static_expcols']['iloc'].values), idx_exp_dynamic=list(data['Full']['x_dynamic_expcols']['iloc'].values),
                      **param_model)
model.to(device)
#optimizer = optim.SGD(model.parameters(), **param_optim)
optimizer = optim.Adam(model.parameters(), **param_optim)

# Initialize dataset & data loader
LoadOrder = ['x_static', 'x_dynamic', 'x_mask', 'x_delta', 'y', 'y_aux', 'maxsteps']
DataSet = {split:CustomDataset(*[data[split][i] for i in LoadOrder]) for split in ['Train','Test','Val']}
DataLoader = {split:torch.utils.data.DataLoader(DataSet[split], batch_size=param_training['batch_size'], shuffle=True, num_workers=0, pin_memory =True) for split in ['Train','Test','Val']}

res = NNTrain(model, optimizer, DataLoader, param_training, param_ES, param_loss,fname="novel_training.png", baseline = False)



###############################################################
#---------- Evaluating trained model --------------------
###############################################################

model = torch.load(os.path.join(npath,'bestmodel.ckpt'))

results = GeneratePerformanceData(DataLoader, model, sigmoid=False)
flatresults = TruncateAndFlatten(results)
ROCPRC(flatresults,'novel_performance.png')
Calibration(flatresults,'novel_calibration.png')
picklesave(results,os.path.join(npath,'proposed_results.pickle'))



###############################################################
#---------- Performance on auxiliary tasks --------------------
###############################################################

splits = ['Test']
auxcols = data['Full']['y_aux'].columns
fig,axs = plt.subplots(figsize=(16, 10), nrows=2, ncols=3,sharex=True,squeeze=False)

auxcol_up = ['SOFA Respiratory Future Max 24h','SOFA Cardiovascular Future Max 24h','SOFA Neurological Future Max 24h']
auxcol_down = ['SOFA Hepatic Future Max 24h','SOFA Hematologic Future Max 24h','SOFA Renal Future Max 24h']

aux = pd.concat((pd.DataFrame(flatresults[split]['y_aux'],columns = auxcols),pd.DataFrame(flatresults[split]['y_aux_proba'],columns = [f'{i}_proba' for i in auxcols])),axis=1)

for j,auxcol in enumerate(auxcol_up):
    curaux = aux.groupby(auxcol)[f'{auxcol}_proba'].quantile([.25,.50,.75]).unstack()
    curaux[.25] = curaux[.5]-curaux[.25]
    curaux[.75] = curaux[.75]-curaux[.5]
    axs[0,j].errorbar(curaux.index,curaux[.5], yerr=(curaux[.25],curaux[.75]), capsize=10, color='tab:blue',marker='o')
    axs[0,j].plot([0,1],[0,1],color='black',ls=':')
    axs[0,j].set_xlabel('Actual value', FontSize = 16)
    axs[0,j].set_ylabel('Predicted value', FontSize = 16)
    axs[0,j].set_title(auxcol, FontSize = 20)
    
    
for j,auxcol in enumerate(auxcol_down):
    curaux = aux.groupby(auxcol)[f'{auxcol}_proba'].quantile([.25,.50,.75]).unstack()
    curaux[.25] = curaux[.5]-curaux[.25]
    curaux[.75] = curaux[.75]-curaux[.5]
    axs[1,j].errorbar(curaux.index,curaux[.5], yerr=(curaux[.25],curaux[.75]), capsize=10, color='tab:blue',marker='o')
    axs[1,j].plot([0,1],[0,1],color='black',ls=':')
    axs[1,j].set_xlabel('Actual value', FontSize = 16)
    axs[1,j].set_ylabel('Predicted value', FontSize = 16)
    axs[1,j].set_title(auxcol, FontSize = 20)
    
    
plt.tight_layout()
#plt.savefig(os.path.join(npath,'Aux_MSE.pdf'),bbox_inches='tight', dpi=300)



###############################################################
#------------------- Explanation - relevance visualization ------
###############################################################

results = pickleload(os.path.join(npath,'proposed_results.pickle'))

#expcols = list(data['Full']['y_aux'].columns) + list(data['Full']['x_static_expcols'].index) + list(data['Full']['x_dynamic_expcols'].index)
expcols = list(data['Full']['y_aux'].columns)

#pd.DataFrame(results[split]['att'].reshape((-1,results[split]['att'].shape[2])), columns = expcols).agg(['min','max','mean','median'])


split='Test'
n_ids=20

cmap = matplotlib.cm.Reds #inferno
cmapalpha = 1.0
num_exp_aux = data['Full']['y_aux'].shape[1]
#num_exp_static = data['Full']['x_static_expcols'].shape[0]
#num_exp_dynamic = data['Full']['x_dynamic_expcols'].shape[0]

#totrows = 1 + num_exp_aux + num_exp_static + num_exp_dynamic
totrows = 1 + num_exp_aux

deadids = (results[split]['y'].sum(axis=1)>0).flatten().nonzero()[0]
reqd_patid = data['Full']['ids']['Test'].to_frame().reset_index(drop = False).iloc[deadids].id.to_numpy()
randids = np.random.choice(deadids,n_ids,replace=False)

#randids = np.random.choice(data[split]['y'].shape[0],n_ids,replace=False)

#randids = [932, 448, 950, 1088, 304, 1163, 645, 305, 1389]

for randid in randids:
    fig,axs = plt.subplots(nrows=totrows,figsize=(15,30),sharex=True)
    
    maxstep = (results[split]['maxsteps'])[randid]
    # y and y_proba
    axi = 0
    y = results[split]['y'][randid,:maxstep].flatten()
    y_proba = results[split]['y_proba'][randid,:maxstep].flatten()
    axs[axi].plot([i for i in range(maxstep)],y,label='True',marker='o',color='black',markersize=10,fillstyle='none')
    axs[axi].plot([i for i in range(maxstep)],y_proba,label='Predicted',marker='o',color='red')
    #axs[axi].fill_between(range(maxstep),y1=y,y2=y_proba,color='tab:orange',alpha=0.5)
    axs[axi].set_ylim((0,1))
    #axs[axi].set_yticks(np.arange(0,1.1,.25), fontsize = 16)
    axs[axi].set_ylabel('Mortality \n within 24h',rotation=0,ha='right', FontSize = 24)
    axs[axi].grid()
    axs[axi].legend(loc='lower right',bbox_to_anchor=(1.3, 0),fontsize = 24)
    #axs[axi].set_title(f'{split}: {randid}')
    axs[axi].tick_params(axis='y', labelsize=16)
    
    # predicted
    y_aux = results[split]['y_aux'][randid,:maxstep,:]
    y_aux_proba = results[split]['y_aux_proba'][randid,:maxstep,:]
    exp_att = (results[split]['y_aux_proba'] * results[split]['att'])[randid,:maxstep,:]
    
    #auxnames = []
    
    for auxi, auxname in enumerate(data['Full']['y_aux'].columns):
        axi+=1
        auxval = y_aux[:,auxi]
        auxprobaval = y_aux_proba[:,auxi]
        exp_att_val = exp_att[:, auxi]
        
        axs[axi].plot([i for i in range(maxstep)],auxval, label='True', marker='o',color='black',markersize=10,fillstyle='none')
        
        axs[axi].plot([i for i in range(maxstep)],auxprobaval, label='Predicted',marker='o',color='red')
        
        #axs[axi].plot([i for i in range(maxstep)], exp_att_val, label='Contribution',marker='o',color='blue')
        
        #axs[axi].fill_between(range(maxstep),y1=auxval,y2=auxprobaval,color='tab:orange',alpha=0.5)
        axs[axi].set_ylabel(auxname,rotation=0,ha='right', FontSize = 24)
        axs[axi].legend(loc='lower right',bbox_to_anchor=(1.3, 0),fontsize = 24)
        
        axs[axi].tick_params(axis='y', labelsize=16)
    
        # att,  Auxillery
        for x in [i for i in range(maxstep)]:
            axs[axi].axvspan(x-0.5, x+0.5, alpha=cmapalpha, color=cmap(results[split]['att'][randid, x, auxi]))
        axs[axi].set_ylim((0,1))
        
        axs[axi].grid()
    
    axs[0].set_xlim((0,maxstep))
    axs[axi].tick_params(axis='x', labelsize=20)
    
    #axs[0].set_xticklabels(range(maxstep), fontsize = 16)
    axs[axi].set_xlabel('Hours into ICD Admission', FontSize = 24)
    #plt.show()

#plt.xticks(fontsize = 20)
#axs[axi].tick_params(axis='x', labelsize=24)

plt.tight_layout()
plt.savefig(os.path.join(npath,'Attention_map.pdf'),bbox_inches='tight', dpi=300)



###############################################################################################
#----- Correlation between missing value % and their corresponding imputation loss (MSE loss)
#############################################################################################

# imputation per dynamic variable
# for each dynamic, how relevant was the imputation (% missing), and how correct were the imputations (MSE)

fig,axs  = plt.subplots(figsize=(20,20),ncols=6,sharey=True)
for spliti, split in enumerate(['Train','Test','Val']):
    imp = pd.DataFrame((1-flatresults[split]['x_mask'].mean(axis=0))*100,columns=['Missing %'],index=data['Full']['x_dynamic'].columns)
    MSE = pd.DataFrame(((flatresults[split]['x_mask'] * flatresults[split]['x_dynamic']) - (flatresults[split]['x_mask'] * flatresults[split]['x_h']))**2,columns=data['Full']['x_dynamic'].columns)
    MSE = MSE.sum(axis=0)
    MSE = MSE/flatresults[split]['x_mask'].sum(axis=0)
    imp['MSE'] = MSE
    if split=='Train':
        impindex = imp.sort_values(by=['Missing %','MSE']).index
        y = range(len(impindex))
    imp = imp.loc[impindex,:].copy()
    
    axs[spliti*2+0].barh(y,imp['Missing %'])
    axs[spliti*2+1].barh(y,imp['MSE'])

    if split=='Train':
        axs[0].set_yticks(y)
        axs[0].set_yticklabels(imp.index)

    axs[spliti*2+0].xaxis.tick_top()
    axs[spliti*2+0].xaxis.set_label_position('top') 
    axs[spliti*2+1].xaxis.tick_top()
    axs[spliti*2+1].xaxis.set_label_position('top') 
    axs[spliti*2+0].set_xlabel(f'{split}\nMissing %')
    axs[spliti*2+1].set_xlabel(f'{split}\nMSE')
    
    axs[spliti*2+1].set_xlim((0,1.5))
    
    axs[spliti*2+0].grid(axis='x')
    axs[spliti*2+1].grid(axis='x')
    
axs[0].set_ylim((0,len(imp.index)))
plt.subplots_adjust(wspace=0.1)
plt.show()



###############################################################################################
#----- Imputation performance of selected variables ----------------------------------
#############################################################################################

# example imputation

numpat = 5
fnames = ['Platelet Count','Creatinine (serum)','GCS - Verbal Response','Arterial Blood Pressure mean','Non Invasive Blood Pressure mean','SBP','Heart Rate']

for fname in fnames:
    fidx = data['Full']['x_dynamic'].columns.get_loc(fname)
    
    fig,axs = plt.subplots(nrows=numpat,ncols=3,figsize=(30,8))
    
    for spliti,split in enumerate(['Train','Test','Val']):
        patidx = np.random.choice(range(results[split]['x_dynamic'].shape[0]),numpat,replace=False)

            
        for pati,patidx in enumerate(patidx):
            
            maxstep = results[split]['maxsteps'][patidx]
            
            validsteps = np.argwhere(results[split]['x_mask'][pati,:maxstep,fidx]).flatten()
            axs[pati,spliti].plot(validsteps,
                                  results[split]['x_dynamic'][pati,validsteps,fidx],
                                  color='black',label='Real',marker='o',alpha=1,fillstyle='none',markersize=10)
            axs[pati,spliti].plot(range(maxstep),results[split]['x_h'][pati,:maxstep,fidx],color='tab:red',label='Predicted',marker='o',alpha=1,markersize=5)
            
            if pati==0:
                axs[pati,spliti].set_title(split)
            axs[pati,spliti].set_ylim((-3,3))
            axs[pati,spliti].grid(axis='y')
            axs[pati,spliti].legend(loc='upper right')
    plt.subplots_adjust(hspace=0.02,wspace=0.02)
    missing = 1-flatresults[split]['x_mask'][:,fidx].mean()
    plt.suptitle(f'{fname}, missing: {missing:.1%}')
    plt.show()
    
    
###############################################################################################
#----- Example of a baseline model 
# Logistic regresion and XGboost----------------------------------
#############################################################################################
'''
x_static = pd.read_pickle(os.path.join(opath,'static.pickle'))
x_dynamic = pd.read_pickle(os.path.join(opath,'dynamic.pickle'))
y = pd.read_pickle(os.path.join(opath,'task.pickle'))

numcols = flatx.select_dtypes('number').columns

Scaler = sklearn.preprocessing.StandardScaler()
flatx.loc[:,numcols] = Scaler.fit_transform(flatx.loc[:,numcols])
flatx = (flatx*1).fillna(0)

import sklearn
from sklearn import linear_model
model = sklearn.linear_model.LogisticRegression(solver='saga',verbose=3,n_jobs=-1)
model.fit(flatx,y.iloc[:,0])

#pd.DataFrame(model.coef_,columns=flatx.columns,index=['coef']).T.sort_values(by='coef',ascending=False).head(20)

model = xgboost.XGBClassifier()
model.fit(flatx,y.iloc[:,0])

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(flatx,check_additivity=True,approximate=False)
shap.summary_plot(shap_values, features=flatx, show=False)
plt.show()
'''

######### Proposed model without concepts and relevance scores ##########

class BaselineModel(nn.Module):
    def __init__(self,n_static,n_dynamic, n_rnn_hidden = 2**8, n_rnn_layers = 3, p_dropout = 0.5, ns_linear=[2**8,2**8,2**8]):
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
        y_proba = torch.nn.functional.sigmoid(y_proba)
        return {'y_proba':y_proba}




#Baseline model with only temporal SOFA subscores as input followed by sequence modeling and fully connected layers 

class BaselineModel_SOFA(nn.Module):
    def __init__(self, n_y_aux,  n_rnn_hidden = 2**8, n_rnn_layers = 3, p_dropout = 0.5, ns_linear=[2**8,2**8,2**8]):
        super(type(self),self).__init__()
        #self.n_static = n_static
        self.n_y_aux = n_y_aux
        self.n_rnn_hidden = n_rnn_hidden
        self.n_rnn_layers = n_rnn_layers
        self.p_dropout = p_dropout
        self.ns_linear = ns_linear
        
        self.layer_rnn = torch.nn.LSTM(input_size = self.n_y_aux, hidden_size = self.n_rnn_hidden, num_layers = self.n_rnn_layers, dropout = self.p_dropout, batch_first = True)
        self.layers_linear = BuildSequential(self.ns_linear, self.p_dropout, self.n_rnn_hidden)
        self.layer_final = torch.nn.Linear(self.ns_linear[-1],1)
    
    
    def forward(self,batch):
        MAXSTEP = batch['y_aux'].shape[1]
        # Initialize RNN
        rnn_h = torch.zeros(self.n_rnn_layers, batch['y_aux'].shape[0], self.n_rnn_hidden, device = device)
        rnn_c = torch.zeros(self.n_rnn_layers, batch['y_aux'].shape[0], self.n_rnn_hidden, device = device)
        # RNN step
        x_dynamic, (rnn_h,rnn_c)  = self.layer_rnn(batch['y_aux'], (rnn_h,rnn_c))
        # Cat
        #x = torch.cat([x_dynamic, batch['x_static'].unsqueeze(1).repeat(1, MAXSTEP, 1)],dim=2)
        # Linear
        y_proba = self.layer_final(self.layers_linear(x_dynamic))
        y_proba = torch.nn.functional.log_softmax(y_proba)
        return {'y_proba':y_proba}
        