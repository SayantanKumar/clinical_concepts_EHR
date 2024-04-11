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

import time
start_time = time.time()

################################################
################################################

data={}
opath = 'C:/Users/Windows/Desktop/Sayantan/Research/MultiTask_Mortality_Prediction/Data/pickled_reqd'
for table in os.listdir(opath):
    tablename = table.split('.pickle')[0]
    print('Loading {:>20}...'.format(tablename),end='')
    curtime = time.time()
    data[tablename] = pd.read_pickle(os.path.join(opath,table))
    print('({:6.1f}s)'.format(time.time()-curtime))
    
##-----------------------------------------

# define path to save progress

npath = os.path.join('./output')
os.makedirs(npath,exist_ok=True)

print('.....Admission and patients information...')
adm_pat_cohort, dx_long = admission_patients_table(data)


print('.....Chartevents...')
cohort_ce = chartevents(data, adm_pat_cohort)


print('.....Static feature generation...')
cohort_static, static, priorcomo = static_feature_generation(data, cohort_ce, dx_long)


# print('.....Adding elixhauser comorbidity information to static feature set...')
# static_como = add_elixhauser_static(data, cohort_static, static, priorcomo)


print('....Time-series (dynamic) feature generation...')
ce_dynamic, cohort_dynamic = dynamic_feature_generation(data, static, cohort_static)


print('.....Resample time-series data to hourly interval...')
ce_hourly = resample_time_series(ce_dynamic, cohort_dynamic, static, data)


print('.....Adding mortality outcome label to data...')
ce_hourly_mort = mortality_outcome(ce_hourly, cohort_dynamic)


HORIZON_HRS = 24
HORIZON = pd.Timedelta(HORIZON_HRS,'h')
var_death = 'Death w/in {}h'.format(HORIZON_HRS)

#####################################################################
#####################################################################

print('.....SOFA organ scores (time-series outcome label generation)...')
#ce_hourly_sofa = SOFA_organ_scores(ce_hourly_mort)

HORIZON_HRS = 24
HORIZON = pd.Timedelta(HORIZON_HRS,'h')
var_death = 'Death w/in {}h'.format(HORIZON_HRS)

ce_hourly = ce_hourly_mort.copy()
    
# SOFA Respiratory (PF Ratio + Mechanical Ventilation)

PlotRelationship(ce_hourly,var_death,'SpO2 FiO2 Ratio',bins=5)

# cutoffs
# https://www.sciencedirect.com/science/article/pii/S0016508513002916?via%3Dihub
# https://bmcanesthesiol.biomedcentral.com/articles/10.1186/s12871-017-0304-8/tables/1

var_SOFA_Resp = 'SOFA Respiratory'
var_SF = 'SpO2 FiO2 Ratio'
SF_ffill = ce_hourly[var_SF].groupby('id').ffill()
ce_hourly[var_SOFA_Resp] = 0
ce_hourly.loc[SF_ffill<=(100+100*3),var_SOFA_Resp] = 1 # 300
ce_hourly.loc[SF_ffill<=(100+100*2),var_SOFA_Resp] = 2 # 220
ce_hourly.loc[SF_ffill<=(100+100*1),var_SOFA_Resp] = 3 # 141
ce_hourly.loc[SF_ffill<=(100+100*0),var_SOFA_Resp] = 4  # 67

# visualize
TaskAssess(ce_hourly,var_death,var_SOFA_Resp)

# rolling & future
var_rollingmax = '{} Rolling Max {}h'.format(var_SOFA_Resp,HORIZON_HRS)
var_futuremax  = '{} Future Max {}h'.format(var_SOFA_Resp,HORIZON_HRS)
ce_hourly[var_rollingmax] = RollingMax(ce_hourly[var_SOFA_Resp],N=HORIZON_HRS)
ce_hourly[var_futuremax]  = FutureMax(ce_hourly[var_rollingmax],N=HORIZON_HRS)
TaskAssess(ce_hourly,var_death,var_rollingmax)
TaskAssess(ce_hourly,var_death,var_futuremax)

##-------------------------------------------------------------------------------
##---------------------------------------------------------------------------------

# SOFA Cardiovascular (MAP + Pressors)
var_PAR = 'PAR'
var_SOFA_Cardio = 'SOFA Cardiovascular'

PlotRelationship(ce_hourly,var_death,var_PAR,bins=20)

# cutoffs
# https://www.sciencedirect.com/topics/medicine-and-dentistry/organ-dysfunction-score
PAR_ffill = ce_hourly[var_PAR].groupby('id').ffill()
ce_hourly[var_SOFA_Cardio] = 0
ce_hourly.loc[PAR_ffill>10,var_SOFA_Cardio] = 1 # 300
ce_hourly.loc[PAR_ffill>15,var_SOFA_Cardio] = 2 # 220
ce_hourly.loc[PAR_ffill>20,var_SOFA_Cardio] = 3 # 141
ce_hourly.loc[PAR_ffill>30,var_SOFA_Cardio] = 4  # 67

# visualize
TaskAssess(ce_hourly,var_death,var_SOFA_Cardio)

# rolling & future
var_rollingmax = '{} Rolling Max {}h'.format(var_SOFA_Cardio,HORIZON_HRS)
var_futuremax  = '{} Future Max {}h'.format(var_SOFA_Cardio,HORIZON_HRS)
ce_hourly[var_rollingmax] = RollingMax(ce_hourly[var_SOFA_Cardio],N=HORIZON_HRS)
ce_hourly[var_futuremax]  = FutureMax(ce_hourly[var_rollingmax],N=HORIZON_HRS)
TaskAssess(ce_hourly,var_death,var_rollingmax)
TaskAssess(ce_hourly,var_death,var_futuremax)

##-------------------------------------------------------------------------------
##---------------------------------------------------------------------------------

# SOFA Neurologic (GCS)
var_SOFA_Neuro = 'SOFA Neurological'
var_GCS = 'GCS Total'
vars_GCS = [i for i in ce_hourly if 'GCS' in i and (i!=var_GCS)]
ce_hourly[var_GCS] = ce_hourly.loc[:,vars_GCS].groupby('id').ffill().fillna(ce_hourly.loc[:,vars_GCS].max()).sum(axis=1)

PlotRelationship(ce_hourly,var_death,var_GCS,bins=10)

# cutoffs
ce_hourly[var_SOFA_Neuro] = 0
ce_hourly.loc[ce_hourly[var_GCS]<15,var_SOFA_Neuro] = 1 
ce_hourly.loc[ce_hourly[var_GCS]<13,var_SOFA_Neuro] = 2 
ce_hourly.loc[ce_hourly[var_GCS]<10,var_SOFA_Neuro] = 3 
ce_hourly.loc[ce_hourly[var_GCS]<6,var_SOFA_Neuro] = 4

# visualize
TaskAssess(ce_hourly,var_death,var_SOFA_Neuro)

# rolling & future
var_rollingmax = '{} Rolling Max {}h'.format(var_SOFA_Neuro,HORIZON_HRS)
var_futuremax  = '{} Future Max {}h'.format(var_SOFA_Neuro,HORIZON_HRS)
ce_hourly[var_rollingmax] = RollingMax(ce_hourly[var_SOFA_Neuro],N=HORIZON_HRS)
ce_hourly[var_futuremax]  = FutureMax(ce_hourly[var_rollingmax],N=HORIZON_HRS)
TaskAssess(ce_hourly,var_death,var_rollingmax)
TaskAssess(ce_hourly,var_death,var_futuremax)

##-------------------------------------------------------------------------------
##--------------------------------------------------------------------------------

# SOFA Hepatic (Bilirubin)

var_SOFA_Hepatic = 'SOFA Hepatic'
var_Bili = 'Total Bilirubin'
PlotRelationship(ce_hourly,var_death,var_Bili,bins=5)

Bili = ce_hourly[var_Bili].groupby('id').ffill()

# cutoffs
ce_hourly[var_SOFA_Hepatic] = 0
ce_hourly.loc[Bili>=1.2,var_SOFA_Hepatic] = 1 
ce_hourly.loc[Bili>=2.0,var_SOFA_Hepatic] = 2 
ce_hourly.loc[Bili>=6.0,var_SOFA_Hepatic] = 3 
ce_hourly.loc[Bili>=12.0,var_SOFA_Hepatic] = 4

# visualize
TaskAssess(ce_hourly,var_death,var_SOFA_Hepatic)

# rolling & future
var_rollingmax = '{} Rolling Max {}h'.format(var_SOFA_Hepatic,HORIZON_HRS)
var_futuremax  = '{} Future Max {}h'.format(var_SOFA_Hepatic,HORIZON_HRS)
ce_hourly[var_rollingmax] = RollingMax(ce_hourly[var_SOFA_Hepatic],N=HORIZON_HRS)
ce_hourly[var_futuremax]  = FutureMax(ce_hourly[var_rollingmax],N=HORIZON_HRS)
TaskAssess(ce_hourly,var_death,var_rollingmax)
TaskAssess(ce_hourly,var_death,var_futuremax)

##-------------------------------------------------------------------------------
##--------------------------------------------------------------------------------

# SOFA Hematologic (PLT)

var_SOFA_Hematologic = 'SOFA Hematologic'
var_INR = 'INR'
PlotRelationship(ce_hourly,var_death,var_INR,bins=10)

INR = ce_hourly[var_INR].groupby('id').ffill()

# cutoffs
# https://www.sciencedirect.com/science/article/pii/S0016508513002916?via%3Dihub
ce_hourly[var_SOFA_Hematologic] = 0
ce_hourly.loc[INR>=1.1,var_SOFA_Hematologic] = 1 
ce_hourly.loc[INR>=1.25,var_SOFA_Hematologic] = 2 
ce_hourly.loc[INR>=1.5,var_SOFA_Hematologic] = 3 
ce_hourly.loc[INR>=2.5,var_SOFA_Hematologic] = 4

# visualize
TaskAssess(ce_hourly,var_death,var_SOFA_Hematologic)

# rolling & future
var_rollingmax = '{} Rolling Max {}h'.format(var_SOFA_Hematologic,HORIZON_HRS)
var_futuremax  = '{} Future Max {}h'.format(var_SOFA_Hematologic,HORIZON_HRS)
ce_hourly[var_rollingmax] = RollingMax(ce_hourly[var_SOFA_Hematologic],N=HORIZON_HRS)
ce_hourly[var_futuremax]  = FutureMax(ce_hourly[var_rollingmax],N=HORIZON_HRS)
TaskAssess(ce_hourly,var_death,var_rollingmax)
TaskAssess(ce_hourly,var_death,var_futuremax)

##-------------------------------------------------------------------------------
##--------------------------------------------------------------------------------

# Renal (Creatinine)
var_SOFA_Renal = 'SOFA Renal'
var_Creat = 'Creatinine (serum)'
PlotRelationship(ce_hourly,var_death,var_Creat,bins=5)

Creat = ce_hourly[var_Creat].groupby('id').ffill()

# cutoffs
ce_hourly[var_SOFA_Renal] = 0
ce_hourly.loc[Creat>=1.2,var_SOFA_Renal] = 1 
ce_hourly.loc[Creat>=2.0,var_SOFA_Renal] = 2 
ce_hourly.loc[Creat>=3.5,var_SOFA_Renal] = 3 
ce_hourly.loc[Creat>=5.0,var_SOFA_Renal] = 4

# visualize
TaskAssess(ce_hourly,var_death,var_SOFA_Renal)

# rolling & future
var_rollingmax = '{} Rolling Max {}h'.format(var_SOFA_Renal,HORIZON_HRS)
var_futuremax  = '{} Future Max {}h'.format(var_SOFA_Renal,HORIZON_HRS)
ce_hourly[var_rollingmax] = RollingMax(ce_hourly[var_SOFA_Renal],N=HORIZON_HRS)
ce_hourly[var_futuremax]  = FutureMax(ce_hourly[var_rollingmax],N=HORIZON_HRS)
TaskAssess(ce_hourly,var_death,var_rollingmax)
TaskAssess(ce_hourly,var_death,var_futuremax)



#####################################################################
#####################################################################

print('.....Saving static, time-series features, auxiliary tables (SOFA) and mortality outcome...')
#save = saveitems(data, static, cohort_static, ce_hourly_sofa, npath)

############# Save items #####################

cohort = cohort_static.copy()

temp = static.loc[:,[i for i in static if 'Prior' in i]].reset_index().merge(cohort.loc[:,['id','Death w/in 24h of ICU disch']],how='inner',on='id').set_index('id')
temp = i2bmi.cohort_comparison(temp,'Death w/in 24h of ICU disch').sort_values(by='-log10p',ascending=False)
temp = temp.drop([i for i in temp if 'Missing' in i],axis=1)

#-----------------------------------------------------------------------------------------

static_cols_keep = ['age', 'ED admit', 'BMI', 'gender, Female',
       'gender, Male', 'Admission type, DIRECT EMER.',
       'Admission type, ELECTIVE', 'Admission type, EW EMER.',
       'Admission type, OBSERVATION ADMIT',
       'Admission type, SURGICAL SAME DAY ADMISSION', 'Admission type, URGENT',
       'Admission location, EMERGENCY ROOM', 'Admission location, OTHER',
       'Admission location, PHYSICIAN REFERRAL',
       'Admission location, TRANSFER FROM HOSPITAL', 'Insurance, Medicaid',
       'Insurance, Medicare', 'Insurance, Other', 'Ethnicity, ASIAN',
       'Ethnicity, BLACK/AFRICAN AMERICAN', 'Ethnicity, HISPANIC/LATINO',
       'Ethnicity, OTHER', 'Ethnicity, WHITE']


#static_cols_keep = ['age', 'ED admit', 'BMI', 'gender, Female', 'gender, Male', '(Prior Elixhauser) Moore17']

save={}

# static
static_selected = static[static_cols_keep]
save['x_static']=static_selected


# primary task
save['y']=ce_hourly[var_death]

# explanation, static
#cols_exp_s = ['age','(Prior Elixhauser) Moore17']
#save['x_static_expcols']=pd.DataFrame([static.columns.get_loc(i) for i in cols_exp_s],index=cols_exp_s,columns=['iloc'])
# explanation, predicted

#-------------------------------------------------------------------------------------------------

cefreq = ce_hourly.isnull().mean().sort_values(ascending=False)
exclcols_dynamic = cefreq.loc[cefreq>.95].index
ce_hourly = ce_hourly.drop(exclcols_dynamic,axis=1)

cols_exp_p = ['SOFA Respiratory Future Max 24h','SOFA Cardiovascular Future Max 24h','SOFA Neurological Future Max 24h','SOFA Hepatic Future Max 24h','SOFA Hematologic Future Max 24h','SOFA Renal Future Max 24h']
save['y_aux'] = ce_hourly.loc[:,cols_exp_p]


# dyanmic

dynamic_cols_drop = ['SOFA Respiratory', 'SOFA Respiratory Rolling Max 24h',
       'SOFA Cardiovascular', 'SOFA Cardiovascular Rolling Max 24h',
       'GCS Total', 'SOFA Neurological', 'SOFA Neurological Rolling Max 24h',
       'SOFA Hepatic', 'SOFA Hepatic Rolling Max 24h', 'SOFA Hematologic',
       'SOFA Hematologic Rolling Max 24h', 'SOFA Renal',
       'SOFA Renal Rolling Max 24h']

cols_x_d = [i for i in ce_hourly if i not in cols_exp_p and i!=var_death]
save['x_dynamic'] = ce_hourly.loc[:,cols_x_d]

# explanation, dynamic
#cols_exp_d = ['SOFA Respiratory Rolling Max 24h','SOFA Cardiovascular Rolling Max 24h','SOFA Neurological Rolling Max 24h','SOFA Hepatic Rolling Max 24h','SOFA Hematologic Rolling Max 24h','SOFA Renal Rolling Max 24h','SOFA Lactic Acidosis Rolling Max 24h']
#cols_exp_d = []
#save['x_dynamic_expcols'] = pd.DataFrame([save['x_dynamic'].columns.get_loc(i) for i in cols_exp_d],index=cols_exp_d,columns=['iloc'])


##--------------------------------------------------------------------------------------------

for saveitem in save:
    print('{}: {}'.format(saveitem, save[saveitem].shape))
    #display(save[saveitem].sample(5))
    save[saveitem].to_pickle(os.path.join(npath,'{}.pickle'.format(saveitem)))
