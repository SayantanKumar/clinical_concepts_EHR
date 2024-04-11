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
###############################################################


def NNTrain(model, optimizer, DataLoader, param_training, param_ES, param_loss,fname, baseline = False):
    # define return dictionary
    res = {
        'Train':{'Epochs':[],'Loss':{},'Time':[],},
        'Val':{'Epochs':[],'Loss':{},'Time':[],}
    }
    for loss in list(param_loss)+['Total']:
        for split in res:
            res[split]['Loss'][loss]={'Unweighted':[],'Weighted':[]}
    
    # Plotting
    fig,axs = plt.subplots(nrows=len(param_loss)+2,figsize=(28,12),sharex=True)
    
    # initalize ES params
    param_ES['minloss'] = np.inf
    param_ES['earlystopping']= 0
    
    for e in range(param_training['epochs']):
        split='Train'
        
        # Housekeeping
        res[split]['Epochs'].append(e)
        curtime = time.time()
        
        # Training mode
        model.train()
       
        # Initialize loss list for this epoch
        for loss in list(param_loss)+['Total']:
            res[split]['Loss'][loss]['Unweighted'].append([])
            res[split]['Loss'][loss]['Weighted'].append([])
        
        # batch
        for b,loadedbatch in enumerate(DataLoader[split]):
            # Load batch (into GPU)
            batch = {batchitem:loadedbatch[batchitem].to(device) for batchitem in loadedbatch}
        
            # Zero out gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(batch)
            batch.update(output)

            # Log batch losses
            res[split]['Loss']['Total']['Weighted'][-1].append(0)
            res[split]['Loss']['Total']['Unweighted'][-1].append(0)
            for loss in param_loss:
                temploss = param_loss[loss]['Func'](batch)
                res[split]['Loss']['Total']['Weighted'][-1][-1] += (temploss * param_loss[loss]['Weight'])
                temploss = temploss.detach().item()
                res[split]['Loss']['Total']['Unweighted'][-1][-1] += temploss
                res[split]['Loss'][loss]['Unweighted'][-1].append(temploss)
                res[split]['Loss'][loss]['Weighted'][-1].append(temploss * param_loss[loss]['Weight'])

            # Backward pass
            res[split]['Loss']['Total']['Weighted'][-1][-1].backward()

            # Update parameters
            optimizer.step()
            
            # Detach Last Total Weighted Loss
            res[split]['Loss']['Total']['Weighted'][-1][-1] = res[split]['Loss']['Total']['Weighted'][-1][-1].detach().item()
            
        # Housekeeping
        res[split]['Time'].append(time.time()-curtime)
    
        # Validation
        if e%param_ES['epochperval']==0:
            split='Val'
            
            # Housekeeping
            res[split]['Epochs'].append(e)
            curtime = time.time()
            
            # Validation mode
            model.eval()
            
             # Initialize loss list for this epoch
            for loss in list(param_loss)+['Total']:
                res[split]['Loss'][loss]['Unweighted'].append([])
                res[split]['Loss'][loss]['Weighted'].append([])
        
            # batch
            for b,loadedbatch in enumerate(DataLoader[split]):
                # Load batch (into GPU)
                batch = {batchitem:loadedbatch[batchitem].to(device) for batchitem in loadedbatch}

                # Forward pass
                output = model(batch)
                batch.update(output)

                # Log batch losses
                res[split]['Loss']['Total']['Weighted'][-1].append(0)
                res[split]['Loss']['Total']['Unweighted'][-1].append(0)
                for loss in param_loss:
                    temploss = param_loss[loss]['Func'](batch)
                    res[split]['Loss']['Total']['Weighted'][-1][-1] += (temploss * param_loss[loss]['Weight'])
                    temploss = temploss.detach().item()
                    res[split]['Loss']['Total']['Unweighted'][-1][-1] += temploss
                    res[split]['Loss'][loss]['Unweighted'][-1].append(temploss)
                    res[split]['Loss'][loss]['Weighted'][-1].append(temploss * param_loss[loss]['Weight'])
                
                # Detach Last Total Weighted Loss
                res[split]['Loss']['Total']['Weighted'][-1][-1] = res[split]['Loss']['Total']['Weighted'][-1][-1].detach().item()
                
            # Housekeeping
            res[split]['Time'].append(time.time()-curtime)
            torch.save(model,os.path.join(npath,'finalmodel.ckpt'))
            
            # Early Stoppings
            val_epoch_loss = np.median(res[split]['Loss']['Total']['Weighted'][-1])
            
            if (val_epoch_loss < param_ES['minloss']) or (e < param_ES['miniter']):
                if (val_epoch_loss < param_ES['minloss']):
                    param_ES['minloss'] = val_epoch_loss
                    param_ES['bestiter'] = e
                param_ES['earlystopping'] = 0
                
                if baseline == False:
                    torch.save(model,os.path.join(npath,'bestmodel.ckpt'))
                else:
                    torch.save(model,os.path.join(npath,'bestmodel_baseline.ckpt'))
                
            else:
                param_ES['earlystopping'] += param_ES['epochperval']
                if param_ES['earlystopping']>param_ES['patience']:
                    plt.gcf()
                    plt.savefig(os.path.join(npath,fname),dpi=300)
                    return res
        
        # Plotting
        curtime = time.time()
        for ax in axs:
            ax.cla()
        axi=0
        # Plot losses
        colors = {'Train':'black','Val':'tab:red'}
        for loss in list(param_loss)+['Total']:
            # Unweighted
            for split in colors:
                p25,p50,p75 = np.percentile(np.array(res[split]['Loss'][loss]['Unweighted']),[25,50,75],axis=1)
                axs[axi].errorbar(x=res[split]['Epochs'], y=p50, yerr=[p50-p25, p75-p50], color=colors[split],label=split,capsize=10, marker='o')
            axs[axi].set_yscale('log')
            # Weighted
            #twinx = axs[axi].twinx()
            for split in colors:
                p25,p50,p75 = np.percentile(np.array(res[split]['Loss'][loss]['Weighted']),[25,50,75],axis=1)
                axs[axi].errorbar(x=res[split]['Epochs'], y=p50, yerr=[p50-p25, p75-p50], color=colors[split],label=f"{split} Weighted",capsize=10, marker='x')
            axs[axi].set_ylabel(loss,rotation=0,ha='right')
            #twinx.set_yscale('log')
            axi+=1
        # Plot time taken to train/validate
        for split in colors:
            axs[axi].plot(res[split]['Epochs'],res[split]['Time'],markersize=10, marker='o',color=colors[split],label=split)
        axs[axi].set_ylabel('Time (s)',rotation=0,ha='right')
        # Draw vertical line for best iteration thus far
        for ax in axs:
            ax.axvline(param_ES['bestiter'],color='tab:blue')
        for vlineaxi, loss in enumerate(list(param_loss)+['Total']):
            axs[vlineaxi].text(param_ES['bestiter'], axs[vlineaxi].get_ylim()[1], f" {np.percentile(np.array(res['Val']['Loss'][loss]['Unweighted'][param_ES['bestiter']]),50):.3E}",va='top',ha='left')
        
        for ax in axs:
            ax.legend(loc='upper left')
            ax.grid(which='both', axis='both')
        axs[0].set_title(f"Epoch {e} out of {param_training['epochs']} ({time.time()-curtime:.1f}s)")
        display(plt.gcf())
        clear_output(wait=True)
    
    # save figure
    plt.gcf()
    plt.savefig(os.path.join(npath,fname),dpi=300)
    
    # text when finished
    
    return res



#################################################################
#################################################################


# run through batches and generate data for evaluation
def sigmoidfunc(z):
    return 1/(1 + np.exp(-z))


def GeneratePerformanceData(dataloaders,model,sigmoid=True):
    # run through batches
    # strip extra timesteps
    # stack vertically
    results={}
    model.eval()
    with torch.no_grad():
        for split in dataloaders:
            results[split]={}
            for b,loadedbatch in enumerate(dataloaders[split]):
                # load batch into GPU
                batch = {}
                for batchitem in loadedbatch:
                    batch[batchitem] = loadedbatch[batchitem].to(device)
                # get outputs from each batch
                output = model(batch)
                for outputkey in output:
                    batch[outputkey] = output[outputkey]
                for outitem in batch:
                    if outitem not in results[split]:
                        results[split][outitem]=[]
                    results[split][outitem].append(batch[outitem].detach().cpu())
            for outitem in results[split]:
                results[split][outitem] = torch.cat(results[split][outitem],dim=0).numpy()
            # Apply sigmoid
            if sigmoid:
                results[split]['y_proba'] = sigmoidfunc(results[split]['y_proba'])
        return results
    
    
    
    
def TruncateAndFlatten(results):
    ret={}
    for split in results:
        ret[split]={}
        for df in results[split]:
            ret[split][df]={}
            temp=[]
            if len(results[split][df].shape)==3:
                for idx in range(results[split][df].shape[0]):
                    maxstep = results[split]['maxsteps'][idx]
                    temp.append(results[split][df][idx,:maxstep,:])
                ret[split][df] = np.concatenate(temp,axis=0)
    return ret



# Generate auroc/auprc plots
def ROCPRC(results,fname):
    fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(12, 6))
    for split in results:
        results[split]['fpr'], results[split]['tpr'], _ = sklearn.metrics.roc_curve(results[split]['y'], results[split]['y_proba'])
        results[split]['precision'], results[split]['recall'], _ = sklearn.metrics.precision_recall_curve(results[split]['y'], results[split]['y_proba'])
        results[split]['auroc'] = sklearn.metrics.roc_auc_score(results[split]['y'], results[split]['y_proba'])
        results[split]['auprc'] = sklearn.metrics.average_precision_score(results[split]['y'], results[split]['y_proba'])
        
        axs[0].plot(results[split]['fpr'],results[split]['tpr'],         label='{:>5}: {:.3f}'.format(split,results[split]['auroc']))
        axs[1].plot(results[split]['recall'],results[split]['precision'],label='{:>5}: {:.3f}'.format(split,results[split]['auprc']))
    for ax in axs:
        ax.set_xticks(np.arange(0,1.1,0.1))
        ax.set_yticks(np.arange(0,1.1,0.1))
        ax.set_xlim((-.02,1.02))
        ax.set_ylim((-.02,1.02))
        ax.grid()
        ax.legend()
    plt.savefig(os.path.join(npath,fname),dpi=300)
    plt.show()
    
    
    
def Calibration(results, fname):
    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(4*3,4))
    for axi,split in enumerate(results):
        calib = pd.DataFrame({
            'y':results[split]['y'].flatten(),
            'y_proba':results[split]['y_proba'].flatten(),
        })
        calib['cut'] = pd.IntervalIndex(pd.cut(calib['y_proba'],bins=np.arange(0,1.1,0.1))).left
        calib = calib.groupby('cut')['y'].agg(['sum','size'])
        calib['perc'] = calib['sum']/calib['size']
        axs[axi].bar(calib.index+0.05,calib['size'],width=0.08,color='grey',alpha=0.4)
        axs[axi].set_yscale('log')
        axs[axi].set_xticks(np.arange(0,1.1,0.1))
        axs[axi].set_xlim((0,1)) 
        twinx = axs[axi].twinx()
        twinx.plot(calib.index+0.05,calib.perc,color='red',markersize=3,marker='o')
        twinx.set_yticks(np.arange(0,1.1,0.1))
        twinx.tick_params(axis='y',color='tab:red',labelcolor='tab:red')
        axs[axi].set_title(split)
        twinx.plot([0,1],[0,1],color='grey',linestyle=':')
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(os.path.join(npath,fname),dpi=300)
    plt.show()
    
    


################################################
################################################

def picklesave(file,path):
    with open(os.path.join(path), 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def pickleload(path):
    with open(path, 'rb') as handle:
        file = pickle.load(handle)
    return file


