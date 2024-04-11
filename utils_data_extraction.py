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

npath = os.path.join('./output')

####################################################
#-------- Admission and patient information
####################################################

def admission_patients_table(data):
    
    # Globals

    GAP_TOL_HOURS = 24
    GAP_TOL = pd.Timedelta(GAP_TOL_HOURS,'h')

    MINICULOS_HOURS = 48 # 2 days
    MINICULOS = pd.Timedelta(MINICULOS_HOURS,'h')

    MAXICULOS_HOURS = 24*14 # 14 days
    MAXICULOS = pd.Timedelta(MAXICULOS_HOURS,'h')

    #####------------------------------------------------------------

    # first contiguous ICU stay per patient

    cohort = data['transfers'].copy()
    cohort = cohort.loc[cohort['careunit'].str.contains('ICU\)$|CCU\)$',na=False,regex=True,case=True),['subject_id','hadm_id','careunit','intime','outtime']].sort_values(by=['hadm_id','intime','outtime'])

    # cohort.shift(1) = previous row
    cohort['delta']=0
    cohort.loc[(cohort['hadm_id']!=cohort.shift(1)['hadm_id']) | ((cohort['intime']-cohort.shift(1)['outtime'])>GAP_TOL),'delta']=1
    cohort['id'] = cohort['delta'].cumsum()

    cohort['careunit'] = cohort['careunit'].str.extract('.*\((.*)\)$')
    careunitdummies = pd.get_dummies(cohort['careunit'])                      
    careunitcols = list(careunitdummies.columns)
    cohort = pd.concat([cohort.drop('careunit',axis=1),careunitdummies],axis=1)

    aggdict = {'intime':'min','outtime':'max'}
    for careunitcol in careunitcols:
        aggdict[careunitcol] = 'sum'

    cohort = cohort.groupby(['subject_id','hadm_id','id']).agg(aggdict)

    cohort.loc[:,careunitcols] = cohort.loc[:,careunitcols]>0

    cohort['los'] = cohort.outtime - cohort.intime
    cohort = cohort.loc[(cohort.los>=MINICULOS) & (cohort.los<=MAXICULOS),:].copy()

    cohort = cohort.reset_index()
    cohort = cohort.sort_values(by='intime')

    #print(cohort.shape)
    cohort = cohort.drop_duplicates(subset=['subject_id'],keep='first')
    #print(cohort.shape)

    #-----------------------------------------------------------
    #print((cohort.loc[:,careunitcols].mean()*100).round(1).sort_values(ascending=False).to_frame('%').T)

    # LOS distribution
    plt.figure(figsize = (7,5))
    (cohort['los'].dt.total_seconds()/60/60/24).hist()
    plt.title('LOS distribution')
    plt.xlabel('Length of Stay (days)')
    #-----------------------------------------------------------

    birth = data['patients'].loc[:,['subject_id','anchor_year','anchor_age']].copy()
    birth['birthyear'] = birth['anchor_year'].astype(int) - birth['anchor_age'].astype(int)
    cohort = cohort.merge(birth.loc[:,['subject_id','birthyear']],how='left',on='subject_id')
    cohort['age'] = (cohort['intime'].dt.year-cohort['birthyear']).astype('float')
    cohort = cohort.drop('birthyear',axis=1)
    cohort = cohort.loc[cohort['age']>=18,:].copy()

    # age distribution
    plt.figure(figsize = (7,5))
    cohort['age'].hist()
    plt.title('Age distribution')
    plt.xlabel('Age (years)')

    #-------------------------------------------------------------

    # from patients

    # gender
    gender = data['patients'].loc[data['patients'].subject_id.isin(cohort.subject_id),['subject_id','gender']]
    gender['gender'] = gender['gender'].map({'M':'Male','F':'Female'})
    gender['gender'] = gender['gender'].astype('category')
    cohort = cohort.merge(gender,how='left',on='subject_id')

    ##----------------------------------------------------------

    # from admissions

    # admission type, location, insurance, ethnicity, ed admit

    adm = data['admissions'].loc[data['admissions']['hadm_id'].isin(cohort.hadm_id),['hadm_id','admittime','dischtime','deathtime','admission_type','admission_location','discharge_location','insurance','ethnicity','edregtime']].copy()

    for catcol in ['admission_type','admission_location','discharge_location','ethnicity','insurance']:
        adm[catcol] = adm[catcol].fillna('UNKNOWN').astype('category')
    adm['edregtime'] = adm['edregtime'].notnull()
    adm = adm.rename(columns={
        'admission_type':'Admission type',
        'admission_location':'Admission location',
        'insurance':'Insurance',
        'ethnicity':'Ethnicity',
        'edregtime':'ED admit',
    })

    cohort = cohort.merge(adm,how='left',on='hadm_id')

    ###------------------------------------------------------------

    # from diagnoses

    diag = data['diagnoses_icd'].drop('seq_num',axis=1)
    diag =  diag.merge(data['d_icd_diagnoses'],how='left',on=['icd_code','icd_version'])
    diag['icd_code'] = diag['icd_code'].str.strip()
    diag['icd_version'] = pd.to_numeric(diag['icd_version'])
    dx_long, dx_wide = i2bmi.assign_comorbidities(diag,column_code='icd_code',column_version='icd_version',columns_id=['subject_id','hadm_id'])
    cohort = cohort.loc[cohort['hadm_id'].isin(data['diagnoses_icd']['hadm_id'])].merge(dx_wide.drop('subject_id',axis=1),how='left',on='hadm_id')

    ###-----------------------------------------------------------
    
   
    # oddly, a decent chunk (3000) of patients have a deathtime after outtime

    cohort.loc[cohort['outtime']>cohort['deathtime'],'outtime'] = cohort.loc[cohort['outtime']>cohort['deathtime'],'deathtime']
    cohort.loc[cohort['outtime']>cohort['dischtime'],'outtime'] = cohort.loc[cohort['outtime']>cohort['dischtime'],'dischtime']
    cohort['los'] = cohort['outtime'] - cohort['intime']
    print(cohort.shape)
    cohort = cohort.loc[cohort['los']>=MINICULOS,:].copy()
    print(cohort.shape)

    ##--------------------------------------------------------------

    cohort['Death w/in 24h of ICU disch'] = cohort['deathtime']<=(cohort['outtime']+pd.Timedelta(24,'h'))
    #(cohort['Death w/in 24h of ICU disch'].mean()*100).round(1)

    ###-------------------------------------------------------------

    # calculate LOS for cohort comparison
    cohort['LOS (h)'] = cohort['los'].dt.total_seconds()/60/60

    ###-----------------------------------------------------------------
    cohort =cohort.loc[cohort['LOS (h)'] >= 7*24]

    return cohort, dx_long



####################################################
#-------- Chartevents
####################################################

def chartevents(data, cohort):
    
    
    bmimap = data['d_items'].copy()
    bmimap.loc[bmimap.itemid.isin(['226730','226707']),'type']='Height'
    bmimap.loc[bmimap.itemid.isin(['224639','226512']),'type']='Weight'
    bmimap = bmimap.loc[bmimap['type'].notnull(),:].copy()
    bmi = data['chartevents'].loc[data['chartevents']['subject_id'].isin(cohort.subject_id),:].merge(bmimap,how='inner',on='itemid')
    bmi['valuenum'] = pd.to_numeric(bmi['valuenum'],errors='coerce')
    bmi = bmi.loc[bmi['valuenum'].notnull(),:].copy()
    bmi.loc[bmi['valueuom']=='Inch','valuenum'] = bmi.loc[bmi['valueuom']=='Inch','valuenum'] * 2.54
    bmi.loc[bmi['valueuom']=='Inch','valueuom'] = 'cm'
    bmi.loc[bmi['valueuom']=='cm','valuenum'] = bmi.loc[bmi['valueuom']=='cm','valuenum']/100
    bmi.loc[bmi['valueuom']=='cm','valueuom'] = 'm'

    height = bmi.loc[bmi.type=='Height'].groupby(['subject_id'])['valuenum'].median().to_frame(name='Height (m)')
    weight = bmi.loc[bmi.type=='Weight'].groupby(['hadm_id'])['valuenum'].median().to_frame(name='Weight (kg)')

    weight = weight.loc[weight['Weight (kg)']>25.].copy()
    height = height.loc[height['Height (m)']>1.].copy()

    cohort = cohort.merge(height,how='left',on='subject_id').merge(weight,how='left',on='hadm_id')
    print('{:.1%} missing height'.format(cohort['Height (m)'].isnull().mean()))
    print('{:.1%} missing weight'.format(cohort['Weight (kg)'].isnull().mean()))

    cohort['BMI'] = cohort['Weight (kg)']/(cohort['Height (m)']**2)

    fig,axs = plt.subplots(nrows=1,ncols=3,figsize=(20,4))
    axs[0].hist(cohort['Height (m)'],bins=100)
    axs[1].hist(cohort['Weight (kg)'],bins=100)
    axs[2].hist(cohort['BMI'],bins=100)
    #plt.show()

    ####-----------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------------------

    # simplify rare columns

    #display((cohort.select_dtypes(bool).mean()*100).sort_values(ascending=True).head(10).to_frame('%').T)
    #cohort = cohort.drop('NICU',axis=1)

    catcols = cohort.select_dtypes('category')

    for catcol in catcols:
        cohort[catcol] = cohort[catcol].astype(str)

    cohort.loc[cohort['Admission type'].isin(['EU OBSERVATION','DIRECT OBSERVATION','AMBULATORY OBSERVATION']),'Admission type'] = 'OBSERVATION ADMIT'
    cohort.loc[~cohort['Admission location'].isin(['EMERGENCY ROOM','PHYSICIAN REFERRAL','TRANSFER FROM HOSPITAL']),'Admission location'] = 'OTHER'
    cohort.loc[cohort['discharge_location'].isin(['HOME HEALTH CARE']),'discharge_location'] = 'HOME'
    cohort.loc[cohort['discharge_location'].isin(['OTHER FACILITY','UNKNOWN','ASSISTED LIVING','HEALTHCARE FACILITY','AGAINST ADVICE']),'discharge_location'] = 'OTHER'
    cohort.loc[cohort['Ethnicity'].isin(['UNABLE TO OBTAIN','AMERICAN INDIAN/ALASKA NATIVE','UNKNOWN']),'Ethnicity'] = 'OTHER'

    for catcol in catcols:
        cohort[catcol] = cohort[catcol].astype('category')

#     for col in cohort.select_dtypes('category'):
#         display(cohort[col].value_counts().to_frame('n').T)

    ##-------------------------------------------------------------------------------------------------
    ##-------------------------------------------------------------------------------------------------

    exclude=['subject_id','hadm_id','id','intime','outtime','admittime','dischtime','deathtime','los']
    ret = i2bmi.cohort_comparison(cohort,'Death w/in 24h of ICU disch',[i for i in cohort if i not in exclude])
    ret = ret.loc[:,[i for i in ret if 'Missing' not in i]].copy()
    #display(ret.head(60))
    ret.to_excel(os.path.join(npath,'cohort_comparison.xlsx'))

    ###----------------------------------------------------------------------------------------------

    return cohort




####################################################
#-------- Static (time-invariant) feature generation---------------------------
####################################################

def static_feature_generation(data, cohort, dx_long):
    
    keepcols=['CCU','CVICU','MICU','MICU/SICU','Neuro SICU', 'SICU', 'TSICU', 'age', 'gender', 'Admission type', 'Admission location','Insurance','Ethnicity','ED admit','Height (m)','Weight (kg)','BMI']

    static = cohort.set_index('id').loc[:,keepcols].copy()
    for catcol in static.select_dtypes('category').columns:
        dummy = pd.get_dummies(static[catcol])==1
        dummy.columns = ['{}, {}'.format(catcol,i) for i in dummy]
        static = pd.concat([static.drop(catcol,axis=1),dummy],axis=1)

    ##-----------------------------------------------------------------------------

    static['Height (m)'] = static['Height (m)'].fillna(static['Height (m)'].median())
    static['Weight (kg)'] = static['Weight (kg)'].fillna(static['Weight (kg)'].median())
    static['BMI'] = static['Weight (kg)']/(static['Height (m)']**2)

    ##-------------------------------------------------------------------------------
    # prior como

    priorcomo = dx_long.loc[:,['subject_id','hadm_id']+[i for i in dx_long if '(Elixhauser)' in i]].copy()
    priorcomo = priorcomo.merge(data['admissions'].loc[:,['hadm_id','dischtime']],how='inner',on='hadm_id').drop('hadm_id',axis=1)
    cohortwithdischtime = cohort.loc[:,['subject_id','hadm_id','id']].merge(data['admissions'].loc[:,['hadm_id','dischtime']],how='inner',on='hadm_id')
    priorcomo = cohortwithdischtime.merge(priorcomo.rename(columns={'dischtime':'priordischtime'}),how='left',on='subject_id')
    priorcomo = priorcomo.loc[(priorcomo['dischtime']>priorcomo['priordischtime']),:].copy()
    priorcomo = priorcomo.drop(['subject_id','hadm_id','dischtime','priordischtime'],axis=1).groupby('id').max()
    priorcomo.columns = [i.replace('(Elixhauser)','(Prior Elixhauser)') for i in priorcomo]

    display((priorcomo.mean()*100).to_frame('%').round(1).sort_values(by='%',ascending=False))
    
    #################################################################
    
    elixmap = {
        'Congestive heart failure':{
            10:['I09.9', 'I11.0', 'I13.0', 'I13.2', 'I25.5', 'I42.0', 'I42.5-I42.9', 'I43.x', 'I50.x', 'P29.0'],
            9:['398.91', '402.01', '402.11', '402.91', '404.01', '404.03', '404.11', '404.13', '404.91', '404.93', '425.4-425.9', '428.x'],
            'Moore17':9,
            'vanWalraven09':7,
        },
        'Cardiac arrhythmias':{
            10:['I44.1-I44.3', 'I45.6', 'I45.9', 'I47.x-I49.x', 'R00.0', 'R00.1', 'R00.8', 'T82.1', 'Z45.0', 'Z95.0'],
            9:['426.0', '426.13', '426.7', '426.9', '426.10', '426.12', '427.0-427.4', '427.6-427.9', '785.0', '996.01', '996.04', 'V45.0', 'V53.3'],
            'Moore17':0,
            'vanWalraven09':5,
        },
        'Valvular disease':{
            10:['A52.0', 'I05.x-I08.x', 'I09.1', 'I09.8', 'I34.x-I39.x', 'Q23.0-Q23.3', 'Z95.2-Z95.4'],
            9:['093.2', '394.x-397.x', '424.x', '746.3-746.6', 'V42.2', 'V43.3'],
            'Moore17':0,
            'vanWalraven09':-1,
        },
        'Pulmonary circulation disorders':{
            10:['I26.x', 'I27.x', 'I28.0', 'I28.8', 'I28.9'],
            9:['415.0', '415.1', '416.x', '417.0', '417.8', '417.9'],
            'Moore17':6,
            'vanWalraven09':4,
        },
        'Peripheral vascular disorders':{
            10:['I70.x', 'I71.x', 'I73.1', 'I73.8', 'I73.9', 'I77.1', 'I79.0', 'I79.2', 'K55.1', 'K55.8', 'K55.9', 'Z95.8', 'Z95.9'],
            9:['093.0', '437.3', '440.x', '441.x', '443.1-443.9', '447.1', '557.1', '557.9', 'V43.4'],
            'Moore17':3,
            'vanWalraven09':2,
        },
        'Hypertension, (complicated and uncomplicated)':{
            10:['I10.x','I11.x-I13.x', 'I15.x'],
            9:['401.x','402.x-405.x'],
            'Moore17':-1,
            'vanWalraven09':0,
        },
        'Paralysis':{
            10:['G04.1', 'G11.4', 'G80.1', 'G80.2', 'G81.x', 'G82.x', 'G83.0-G83.4', 'G83.9'],
            9:['334.1', '342.x', '343.x', '344.0-344.6', '344.9'],
            'Moore17':5,
            'vanWalraven09':7,
        },
        'Other neurological disorders':{
            10:['G10.x-G13.x', 'G20.x-G22.x', 'G25.4', 'G25.5', 'G31.2', 'G31.8', 'G31.9', 'G32.x', 'G35.x-G37.x', 'G40.x', 'G41.x', 'G93.1', 'G93.4', 'R47.0', 'R56.x'],
            9:['331.9', '332.0', '332.1', '333.4', '333.5', '333.92', '334.x-335.x', '336.2', '340.x', '341.x', '345.x', '348.1', '348.3', '780.3', '784.3'],
            'Moore17':5,
            'vanWalraven09':6,
        },
        'Chronic pulmonary disease':{
            10:['I27.8', 'I27.9', 'J40.x-J47.x', 'J60.x-J67.x', 'J68.4', 'J70.1', 'J70.3'],
            9:['416.8', '416.9', '490.x -505.x', '506.4', '508.1', '508.8'],
            'Moore17':3,
            'vanWalraven09':3,
        },
        'Diabetes, uncomplicated':{
            10:['E10.0', 'E10.1', 'E10.9', 'E11.0', 'E11.1', 'E11.9', 'E12.0', 'E12.1', 'E12.9', 'E13.0', 'E13.1', 'E13.9', 'E14.0', 'E14.1', 'E14.9'],
            9:['250.0-250.3'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Diabetes, complicated':{
            10:['E10.2-E10.8', 'E11.2-E11.8', 'E12.2-E12.8', 'E13.2-E13.8', 'E14.2-E14.8'],
            9:['250.4-250.9'],
            'Moore17':-3,
            'vanWalraven09':0,
        },
        'Hypothyroidism':{
            10:['E00.x-E03.x', 'E89.0'],
            9:['240.9', '243.x', '244.x', '246.1', '246.8'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Renal failure':{
            10:['I12.0', 'I13.1', 'N18.x', 'N19.x', 'N25.0', 'Z49.0-Z49.2', 'Z94.0', 'Z99.2'],
            9:['403.01', '403.11', '403.91', '404.02', '404.03', '404.12', '404.13', '404.92', '404.93', '585.x', '586.x', '588.0', 'V42.0', 'V45.1', 'V56.x'],
            'Moore17':6,
            'vanWalraven09':5,
        },
        'Liver disease':{
            10:['B18.x', 'I85.x', 'I86.4', 'I98.2', 'K70.x', 'K71.1', 'K71.3-K71.5', 'K71.7', 'K72.x-K74.x', 'K76.0', 'K76.2-K76.9', 'Z94.4'],
            9:['070.22', '070.23', '070.32', '070.33', '070.44', '070.54', '070.6', '070.9', '456.0-456.2', '570.x', '571.x', '572.2-572.8', '573.3', '573.4', '573.8', '573.9', 'V42.7'],
            'Moore17':4,
            'vanWalraven09':11,
        },
        'Peptic ulcer disease excluding bleeding':{
            10:['K25.7', 'K25.9', 'K26.7', 'K26.9', 'K27.7', 'K27.9', 'K28.7', 'K28.9'],
            9:['531.7', '531.9', '532.7', '532.9', '533.7', '533.9', '534.7', '534.9'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'AIDS/HIV':{
            10:['B20.x-B22.x', 'B24.x'],
            9:['042.x-044.x'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Lymphoma':{
            10:['C81.x-C85.x', 'C88.x', 'C96.x', 'C90.0', 'C90.2'],
            9:['200.x-202.x', '203.0', '238.6'],
            'Moore17':6,
            'vanWalraven09':9,
        },
        'Metastatic cancer':{
            10:['C77.x-C80.x'],
            9:['196.x-199.x'],
            'Moore17':14,
            'vanWalraven09':12,
        },
        'Solid tumor without metastasis':{
            10:['C00.x-C26.x', 'C30.x-C34.x', 'C37.x-C41.x', 'C43.x', 'C45.x-C58.x', 'C60.x-C76.x', 'C97.x'],
            9:['140.x-172.x', '174.x-195.x'],
            'Moore17':7,
            'vanWalraven09':4,
        },
        'Rheumatoid arthritis/collagen vascular diseases':{
            10:['L94.0', 'L94.1', 'L94.3', 'M05.x', 'M06.x', 'M08.x', 'M12.0', 'M12.3', 'M30.x', 'M31.0-M31.3', 'M32.x-M35.x', 'M45.x', 'M46.1', 'M46.8', 'M46.9'],
            9:['446.x', '701.0', '710.0-710.4', '710.8', '710.9', '711.2', '714.x', '719.3', '720.x', '725.x', '728.5', '728.89', '729.30'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Coagulopathy':{
            10:['D65-D68.x', 'D69.1', 'D69.3-D69.6'],
            9:['286.x', '287.1', '287.3-287.5'],
            'Moore17':11,
            'vanWalraven09':3,
        },
        'Obesity':{
            10:['E66.x'],
            9:['278.0'],
            'Moore17':-5,
            'vanWalraven09':-4,
        },
        'Weight loss':{
            10:['E40.x-E46.x', 'R63.4', 'R64'],
            9:['260.x-263.x', '783.2', '799.4'],
            'Moore17':9,
            'vanWalraven09':6,
        },
        'Fluid and electrolyte disorders':{
            10:['E22.2', 'E86.x', 'E87.x'],
            9:['253.6', '276.x'],
            'Moore17':11,
            'vanWalraven09':5,
        },
        'Blood loss anemia':{
            10:['D50.0'],
            9:['280.0'],
            'Moore17':-3,
            'vanWalraven09':-2,
        },
        'Deficiency anemia':{
            10:['D50.8', 'D50.9', 'D51.x-D53.x'],
            9:['280.1-280.9', '281.x'],
            'Moore17':-2,
            'vanWalraven09':-2,
        },
        'Alcohol abuse':{
            10:['F10', 'E52', 'G62.1', 'I42.6', 'K29.2', 'K70.0', 'K70.3', 'K70.9', 'T51.x', 'Z50.2', 'Z71.4', 'Z72.1'],
            9:['265.2', '291.1-291.3', '291.5-291.9', '303.0', '303.9', '305.0', '357.5', '425.5', '535.3', '571.0-571.3', '980.x', 'V11.3'],
            'Moore17':-1,
            'vanWalraven09':0,
        },
        'Drug abuse':{
            10:['F11.x-F16.x', 'F18.x', 'F19.x', 'Z71.5', 'Z72.2'],
            9:['292.x', '304.x', '305.2-305.9', 'V65.42'],
            'Moore17':-7,
            'vanWalraven09':-7,
        },
        'Psychoses':{
            10:['F20.x', 'F22.x-F25.x', 'F28.x', 'F29.x', 'F30.2', 'F31.2', 'F31.5'],
            9:['293.8', '295.x', '296.04', '296.14', '296.44', '296.54', '297.x', '298.x'],
            'Moore17':-5,
            'vanWalraven09':0,
        },
        'Depression':{
            10:['F20.4', 'F31.3-F31.5', 'F32.x', 'F33.x', 'F34.1', 'F41.2', 'F43.2'],
            9:['296.2', '296.3', '296.5', '300.4', '309.x', '311'],
            'Moore17':-5,
            'vanWalraven09':-3,
        },
    }
    elixmap = pd.DataFrame(elixmap).T['Moore17']
    elixmap.index = ['(Prior Elixhauser) {}'.format(i) for i in elixmap.index]
    priorcomo['(Prior Elixhauser) Moore17'] = (priorcomo * elixmap.loc[priorcomo.columns]).sum(axis=1)

    ##_-----------------------------------------------------------------------------------------

    static = static.merge(priorcomo,how='left',left_index=True,right_index=True)
    static.loc[:,priorcomo.columns] = static.loc[:,priorcomo.columns].fillna(False)
    static['(Prior Elixhauser) Moore17'] = static['(Prior Elixhauser) Moore17'].replace(False,0)


    return cohort, static, priorcomo


####################################################
#--Add Elixhauser comorbidity to the static feature set---------------------------
####################################################

def add_elixhauser_static(data, cohort, static, priorcomo):
    
    elixmap = {
        'Congestive heart failure':{
            10:['I09.9', 'I11.0', 'I13.0', 'I13.2', 'I25.5', 'I42.0', 'I42.5-I42.9', 'I43.x', 'I50.x', 'P29.0'],
            9:['398.91', '402.01', '402.11', '402.91', '404.01', '404.03', '404.11', '404.13', '404.91', '404.93', '425.4-425.9', '428.x'],
            'Moore17':9,
            'vanWalraven09':7,
        },
        'Cardiac arrhythmias':{
            10:['I44.1-I44.3', 'I45.6', 'I45.9', 'I47.x-I49.x', 'R00.0', 'R00.1', 'R00.8', 'T82.1', 'Z45.0', 'Z95.0'],
            9:['426.0', '426.13', '426.7', '426.9', '426.10', '426.12', '427.0-427.4', '427.6-427.9', '785.0', '996.01', '996.04', 'V45.0', 'V53.3'],
            'Moore17':0,
            'vanWalraven09':5,
        },
        'Valvular disease':{
            10:['A52.0', 'I05.x-I08.x', 'I09.1', 'I09.8', 'I34.x-I39.x', 'Q23.0-Q23.3', 'Z95.2-Z95.4'],
            9:['093.2', '394.x-397.x', '424.x', '746.3-746.6', 'V42.2', 'V43.3'],
            'Moore17':0,
            'vanWalraven09':-1,
        },
        'Pulmonary circulation disorders':{
            10:['I26.x', 'I27.x', 'I28.0', 'I28.8', 'I28.9'],
            9:['415.0', '415.1', '416.x', '417.0', '417.8', '417.9'],
            'Moore17':6,
            'vanWalraven09':4,
        },
        'Peripheral vascular disorders':{
            10:['I70.x', 'I71.x', 'I73.1', 'I73.8', 'I73.9', 'I77.1', 'I79.0', 'I79.2', 'K55.1', 'K55.8', 'K55.9', 'Z95.8', 'Z95.9'],
            9:['093.0', '437.3', '440.x', '441.x', '443.1-443.9', '447.1', '557.1', '557.9', 'V43.4'],
            'Moore17':3,
            'vanWalraven09':2,
        },
        'Hypertension, (complicated and uncomplicated)':{
            10:['I10.x','I11.x-I13.x', 'I15.x'],
            9:['401.x','402.x-405.x'],
            'Moore17':-1,
            'vanWalraven09':0,
        },
        'Paralysis':{
            10:['G04.1', 'G11.4', 'G80.1', 'G80.2', 'G81.x', 'G82.x', 'G83.0-G83.4', 'G83.9'],
            9:['334.1', '342.x', '343.x', '344.0-344.6', '344.9'],
            'Moore17':5,
            'vanWalraven09':7,
        },
        'Other neurological disorders':{
            10:['G10.x-G13.x', 'G20.x-G22.x', 'G25.4', 'G25.5', 'G31.2', 'G31.8', 'G31.9', 'G32.x', 'G35.x-G37.x', 'G40.x', 'G41.x', 'G93.1', 'G93.4', 'R47.0', 'R56.x'],
            9:['331.9', '332.0', '332.1', '333.4', '333.5', '333.92', '334.x-335.x', '336.2', '340.x', '341.x', '345.x', '348.1', '348.3', '780.3', '784.3'],
            'Moore17':5,
            'vanWalraven09':6,
        },
        'Chronic pulmonary disease':{
            10:['I27.8', 'I27.9', 'J40.x-J47.x', 'J60.x-J67.x', 'J68.4', 'J70.1', 'J70.3'],
            9:['416.8', '416.9', '490.x -505.x', '506.4', '508.1', '508.8'],
            'Moore17':3,
            'vanWalraven09':3,
        },
        'Diabetes, uncomplicated':{
            10:['E10.0', 'E10.1', 'E10.9', 'E11.0', 'E11.1', 'E11.9', 'E12.0', 'E12.1', 'E12.9', 'E13.0', 'E13.1', 'E13.9', 'E14.0', 'E14.1', 'E14.9'],
            9:['250.0-250.3'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Diabetes, complicated':{
            10:['E10.2-E10.8', 'E11.2-E11.8', 'E12.2-E12.8', 'E13.2-E13.8', 'E14.2-E14.8'],
            9:['250.4-250.9'],
            'Moore17':-3,
            'vanWalraven09':0,
        },
        'Hypothyroidism':{
            10:['E00.x-E03.x', 'E89.0'],
            9:['240.9', '243.x', '244.x', '246.1', '246.8'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Renal failure':{
            10:['I12.0', 'I13.1', 'N18.x', 'N19.x', 'N25.0', 'Z49.0-Z49.2', 'Z94.0', 'Z99.2'],
            9:['403.01', '403.11', '403.91', '404.02', '404.03', '404.12', '404.13', '404.92', '404.93', '585.x', '586.x', '588.0', 'V42.0', 'V45.1', 'V56.x'],
            'Moore17':6,
            'vanWalraven09':5,
        },
        'Liver disease':{
            10:['B18.x', 'I85.x', 'I86.4', 'I98.2', 'K70.x', 'K71.1', 'K71.3-K71.5', 'K71.7', 'K72.x-K74.x', 'K76.0', 'K76.2-K76.9', 'Z94.4'],
            9:['070.22', '070.23', '070.32', '070.33', '070.44', '070.54', '070.6', '070.9', '456.0-456.2', '570.x', '571.x', '572.2-572.8', '573.3', '573.4', '573.8', '573.9', 'V42.7'],
            'Moore17':4,
            'vanWalraven09':11,
        },
        'Peptic ulcer disease excluding bleeding':{
            10:['K25.7', 'K25.9', 'K26.7', 'K26.9', 'K27.7', 'K27.9', 'K28.7', 'K28.9'],
            9:['531.7', '531.9', '532.7', '532.9', '533.7', '533.9', '534.7', '534.9'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'AIDS/HIV':{
            10:['B20.x-B22.x', 'B24.x'],
            9:['042.x-044.x'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Lymphoma':{
            10:['C81.x-C85.x', 'C88.x', 'C96.x', 'C90.0', 'C90.2'],
            9:['200.x-202.x', '203.0', '238.6'],
            'Moore17':6,
            'vanWalraven09':9,
        },
        'Metastatic cancer':{
            10:['C77.x-C80.x'],
            9:['196.x-199.x'],
            'Moore17':14,
            'vanWalraven09':12,
        },
        'Solid tumor without metastasis':{
            10:['C00.x-C26.x', 'C30.x-C34.x', 'C37.x-C41.x', 'C43.x', 'C45.x-C58.x', 'C60.x-C76.x', 'C97.x'],
            9:['140.x-172.x', '174.x-195.x'],
            'Moore17':7,
            'vanWalraven09':4,
        },
        'Rheumatoid arthritis/collagen vascular diseases':{
            10:['L94.0', 'L94.1', 'L94.3', 'M05.x', 'M06.x', 'M08.x', 'M12.0', 'M12.3', 'M30.x', 'M31.0-M31.3', 'M32.x-M35.x', 'M45.x', 'M46.1', 'M46.8', 'M46.9'],
            9:['446.x', '701.0', '710.0-710.4', '710.8', '710.9', '711.2', '714.x', '719.3', '720.x', '725.x', '728.5', '728.89', '729.30'],
            'Moore17':0,
            'vanWalraven09':0,
        },
        'Coagulopathy':{
            10:['D65-D68.x', 'D69.1', 'D69.3-D69.6'],
            9:['286.x', '287.1', '287.3-287.5'],
            'Moore17':11,
            'vanWalraven09':3,
        },
        'Obesity':{
            10:['E66.x'],
            9:['278.0'],
            'Moore17':-5,
            'vanWalraven09':-4,
        },
        'Weight loss':{
            10:['E40.x-E46.x', 'R63.4', 'R64'],
            9:['260.x-263.x', '783.2', '799.4'],
            'Moore17':9,
            'vanWalraven09':6,
        },
        'Fluid and electrolyte disorders':{
            10:['E22.2', 'E86.x', 'E87.x'],
            9:['253.6', '276.x'],
            'Moore17':11,
            'vanWalraven09':5,
        },
        'Blood loss anemia':{
            10:['D50.0'],
            9:['280.0'],
            'Moore17':-3,
            'vanWalraven09':-2,
        },
        'Deficiency anemia':{
            10:['D50.8', 'D50.9', 'D51.x-D53.x'],
            9:['280.1-280.9', '281.x'],
            'Moore17':-2,
            'vanWalraven09':-2,
        },
        'Alcohol abuse':{
            10:['F10', 'E52', 'G62.1', 'I42.6', 'K29.2', 'K70.0', 'K70.3', 'K70.9', 'T51.x', 'Z50.2', 'Z71.4', 'Z72.1'],
            9:['265.2', '291.1-291.3', '291.5-291.9', '303.0', '303.9', '305.0', '357.5', '425.5', '535.3', '571.0-571.3', '980.x', 'V11.3'],
            'Moore17':-1,
            'vanWalraven09':0,
        },
        'Drug abuse':{
            10:['F11.x-F16.x', 'F18.x', 'F19.x', 'Z71.5', 'Z72.2'],
            9:['292.x', '304.x', '305.2-305.9', 'V65.42'],
            'Moore17':-7,
            'vanWalraven09':-7,
        },
        'Psychoses':{
            10:['F20.x', 'F22.x-F25.x', 'F28.x', 'F29.x', 'F30.2', 'F31.2', 'F31.5'],
            9:['293.8', '295.x', '296.04', '296.14', '296.44', '296.54', '297.x', '298.x'],
            'Moore17':-5,
            'vanWalraven09':0,
        },
        'Depression':{
            10:['F20.4', 'F31.3-F31.5', 'F32.x', 'F33.x', 'F34.1', 'F41.2', 'F43.2'],
            9:['296.2', '296.3', '296.5', '300.4', '309.x', '311'],
            'Moore17':-5,
            'vanWalraven09':-3,
        },
    }
    elixmap = pd.DataFrame(elixmap).T['Moore17']
    elixmap.index = ['(Prior Elixhauser) {}'.format(i) for i in elixmap.index]
    priorcomo['(Prior Elixhauser) Moore17'] = (priorcomo * elixmap.loc[priorcomo.columns]).sum(axis=1)

    ##_-----------------------------------------------------------------------------------------

    static = static.merge(priorcomo,how='left',left_index=True,right_index=True)
    static.loc[:,priorcomo.columns] = static.loc[:,priorcomo.columns].fillna(False)
    static['(Prior Elixhauser) Moore17'] = static['(Prior Elixhauser) Moore17'].replace(False,0)

    return static


####################################################
#-------- Time-series feature generation---------------------------
####################################################

def dynamic_feature_generation(data, static, cohort):
    
    cefreq = pd.read_excel('cefreq_reviewed.xlsx')
    cefreq = cefreq.loc[cefreq['REALITEM'].notnull(),['itemid','REALITEM']].rename(columns={'REALITEM':'label'})
    ce = data['chartevents'].loc[(data['chartevents']['hadm_id'].isin(cohort['hadm_id'])),:].copy()
    ce = ce.merge(cefreq,how='inner',on='itemid')
    ce['valuenum'] = pd.to_numeric(ce['valuenum'],errors='coerce')

    ###-----------------------------------------------------------------------------

    # fixing temp

    ce.loc[ce.label=='Temperature Fahrenheit','valuenum'] = (ce.loc[ce.label=='Temperature Fahrenheit','valuenum']-32)*5/9
    ce.loc[ce.label=='Temperature Fahrenheit','label']='Temperature'
    ce.loc[ce.label=='Temperature Celsius','label']='Temperature'

    ###------------------------------------------------------------------------------

    # fixing o2 delivery device

    ce = ce.loc[~((ce.label=='O2 Delivery Device(s)') & (ce['value']=='None')),:].copy()

    # convert 'O2 Delivery Device(s)', a categorical column, to multiple numerical one-hot columns
    idx = ce['label']=='O2 Delivery Device(s)'

    #for i in ce.loc[idx,'label'].value_counts().index:
    #    print('\'{}\':\'{}\','.format(i,i))
    ce.loc[idx,'label'] = ce.loc[idx,'value'].map({'Endotracheal tube':'O2 Delivery Device: IMV', 'CPAP mask ':'O2 Delivery Device: NIPPV', 'Bipap mask ':'O2 Delivery Device: NIPPV', 'High flow neb':'O2 Delivery Device: HFNC', 'High flow nasal cannula':'O2 Delivery Device: HFNC', 'Tracheostomy tube':'O2 Delivery Device: Trach', 'Trach mask ':'O2 Delivery Device: Trach', 'Nasal cannula':'O2 Delivery Device: Supplemental Oxygen', 'Aerosol-cool':'O2 Delivery Device: Supplemental Oxygen', 'Face tent':'O2 Delivery Device: Supplemental Oxygen', 'Non-rebreather':'O2 Delivery Device: Supplemental Oxygen', 'Venti mask ':'O2 Delivery Device: Supplemental Oxygen', 'Medium conc mask ':'O2 Delivery Device: Supplemental Oxygen', 'Other':'O2 Delivery Device: Supplemental Oxygen', 'T-piece':'O2 Delivery Device: Supplemental Oxygen', 'Oxymizer':'O2 Delivery Device: Supplemental Oxygen', 'Vapomist':'O2 Delivery Device: Supplemental Oxygen', 'Ultrasonic neb':'O2 Delivery Device: Supplemental Oxygen',})
    ce.loc[idx,'valuenum']=1

    ###---------------------------------------------------------------------------

    # fixing code

    idx = (ce.label=='Code Status') & (ce.value=='Full code')
    ce.loc[idx,'label'] = 'Full code'
    ce.loc[idx,'valuenum'] = 1

    idx = (ce.label=='Code Status')
    ce.loc[idx,'label'] = 'DNR / DNI'
    ce.loc[idx,'valuenum'] = 1

    ###------------------------------------------------------------------------------

    # filter by intime and outtime

    ce = ce.loc[:,['hadm_id','charttime','valuenum','label']].merge(cohort.loc[:,['hadm_id','id','intime','outtime']],how='left',on='hadm_id')

    print(ce.shape)
    ce = ce.loc[(ce.charttime>=ce.intime) & (ce.charttime<=ce.outtime),:].copy()
    print(ce.shape)

    ##-------------------------------------------------------------------------------

    # identify outliers (<1st and> 99th percentile) except for columns where values are all = 1

    onecols = [i for i in ce.label.unique() if 'O2 Delivery Device' in i] + ['DNR / DNI','Full code']

    ce['EXCLUDE']=False
    for variable in ce.label.unique():
        if variable not in onecols:
            #print(variable)
            varidx = (ce.label==variable)
            (minlim,maxlim) = ce.loc[varidx,'valuenum'].quantile([.01,.99])
            ce.loc[(varidx) & ((ce['valuenum']<minlim) | (ce['valuenum']>maxlim)),'EXCLUDE']=True

    ##---------------------------------------------------------------------------------
    # drop outliers

    print(ce.shape)
    ce = ce.loc[ce['EXCLUDE']==False,:].drop('EXCLUDE',axis=1)
    print(ce.shape)
    
    return ce, cohort



####################################################
#-------- Resample time-series---------------------------
####################################################

def resample_time_series(ce, cohort, static, data):
    
    # resample into hourly bins, take median if multiple

    FREQUENCYSTRING = '1H'

    ce_hourly = ce.sort_values(by='charttime').pivot_table(index=['id','charttime'],columns=['label'],values='valuenum').reset_index()

    ce_hourly = ce_hourly.set_index(['charttime']).groupby('id').resample(FREQUENCYSTRING).median().drop('id',axis=1)

    ##-----------------------------------------------------------------------------------

    # add time since X as features

    # [time since icu admit]
    ce_hourly['Hours since ICU admit'] = ce_hourly.groupby(level='id').cumcount()
    #ce_hourly['Hours since ICU admit'] = ce_hourly['Hours since ICU admit']*6

    # [time since admit] = [time since icu admit] + [time from admit to icu admit]
    ce_hourly = ce_hourly.reset_index().merge(cohort.loc[:,['id','admittime']],how='left',on='id')
    ce_hourly['Hours since admit'] = (ce_hourly['charttime'] - ce_hourly['admittime']).dt.total_seconds()/60/60
    ce_hourly = ce_hourly.set_index(['id','charttime'])
    ce_hourly = ce_hourly.drop('admittime',axis=1)

    ###------------------------------------------------------------------------------------

    def plotoverlap(df,var1,var2):
        fig,axs= plt.subplots(figsize=(8,4))
        set1 = set(np.where(df[var1].notnull())[0])
        set2 = set(np.where(df[var2].notnull())[0])
        matplotlib_venn.venn2((set1,set2),(var1,var2),ax=axs)
        axs.set_title('{} total'.format(df.shape[0]))
        plt.show()

    ##--------------------------------------------------------------------------------------

    # SBP
    var_NISBP = 'Non Invasive Blood Pressure systolic'
    var_ASBP = 'Arterial Blood Pressure systolic'
    var_SBP = 'SBP'
    ce_hourly[var_SBP] = ce_hourly[var_ASBP]
    missidx = ce_hourly[var_SBP].isnull()
    ce_hourly.loc[missidx,var_SBP] = ce_hourly.loc[missidx,var_NISBP]

    # DBP
    var_NIDBP = 'Non Invasive Blood Pressure diastolic'
    var_ADBP = 'Arterial Blood Pressure diastolic'
    var_DBP = 'DBP'
    ce_hourly[var_DBP] = ce_hourly[var_ADBP]
    missidx = ce_hourly[var_DBP].isnull()
    ce_hourly.loc[missidx,var_DBP] = ce_hourly.loc[missidx,var_NIDBP]

    ##------------------------------------------------------------------------------------

    # SI = HR/SBP Heart Rate Non Invasive Blood Pressure systolic, Arterial Blood Pressure systolic
    var_HR = 'Heart Rate'
    var_SI = 'Shock Index'
    ce_hourly[var_SI] = ce_hourly[var_HR]/ce_hourly[var_SBP]

    ##-------------------------------------------------------------------------------------

    # BunCr = BUN/Cr
    var_BUN = 'BUN'
    var_Cr = 'Creatinine (serum)'
    var_BUNCr = 'BUN Cr Ratio'
    ce_hourly[var_BUNCr] = ce_hourly[var_BUN]/ce_hourly[var_Cr]

    ##----------------------------------------------------------------------------------------

    # MAP = 2/3*DBP + 1/3*SBP
    var_MAP = 'MAP'
    ce_hourly[var_MAP] = 2/3 * ce_hourly[var_DBP] + 1/3 * ce_hourly[var_SBP]

    var_AMAP = 'Arterial Blood Pressure mean'
    var_NIMAP = 'Non Invasive Blood Pressure mean'

    #sns.pairplot(ce_hourly.loc[:,[var_MAP,var_AMAP,var_NIMAP]],diag_kind='hist',kind='scatter',plot_kws={'alpha':0.005},corner=True)
    #plt.show()

    ##-----------------------------------------------------------------------------------------

    # PAR = HR * CVP / MAP

    var_CVP = 'Central Venous Pressure'
    var_HR = 'Heart Rate'
    var_MAP = 'MAP'
    var_PAR = 'PAR'

    temp = ce_hourly.loc[:,[var_CVP,var_HR,var_MAP]].groupby('id').ffill().fillna(ce_hourly.loc[:,[var_CVP,var_HR,var_MAP]].median())
    ce_hourly[var_PAR] = temp[var_HR] * temp[var_CVP] / temp[var_MAP]

    ##-------------------------------------------------------------------------------------------

    # PF Ratio = PaO2 / FiO2
    var_PaO2 = 'Arterial O2 pressure'
    var_FiO2 = 'Inspired O2 Fraction'
    var_FlowRate = 'Flow Rate (L/min)'

    # fill forward
    var_PaO2Impute = 'PaO2 Imputed'
    ce_hourly[var_PaO2Impute] = ce_hourly.groupby(level='id')[var_PaO2].ffill().fillna(100)

    # FiO2 from FlowRate LPM*3+21 capped at 100
    var_FiO2Impute = 'FiO2 Imputed'
    ce_hourly[var_FiO2Impute] = ce_hourly.groupby(level='id')[var_FiO2].ffill()
    missidx = ce_hourly[var_FiO2Impute].isnull()
    ce_hourly.loc[missidx,var_FiO2Impute] = ce_hourly.loc[missidx,var_FlowRate]*3+21
    ce_hourly[var_FiO2Impute] = ce_hourly[var_FiO2Impute].clip(upper=100)
    ce_hourly[var_FiO2Impute] = ce_hourly.groupby(level='id')[var_FiO2Impute].ffill()
    ce_hourly[var_FiO2Impute] = ce_hourly[var_FiO2Impute].fillna(21)
    ce_hourly[var_FiO2Impute] = ce_hourly[var_FiO2Impute]/100

    # PF Ratio
    var_PF = 'PaO2 FiO2 Ratio'
    ce_hourly[var_PF] = ce_hourly[var_PaO2Impute]/ce_hourly[var_FiO2Impute]

    ##--------------------------------------------------------------------------------------------

    # SF Ratio = SpO2 / FiO2
    var_SpO2 = 'O2 saturation pulseoxymetry'
    var_FiO2 = 'Inspired O2 Fraction'
    var_FlowRate = 'Flow Rate (L/min)'

    # PF Ratio
    var_SF = 'SpO2 FiO2 Ratio'
    ce_hourly[var_SF] = ce_hourly[var_SpO2]/ce_hourly[var_FiO2Impute]

    ##-----------------------------------------------------------------------------------------------

    # fix o2cols
    o2cols = [i for i in ce_hourly if 'O2 Delivery Device' in i]
    ce_hourly.loc[:,o2cols] = ce_hourly.groupby(level='id')[o2cols].ffill(limit=14).fillna(0) == 1

    # fix code status
    ce_hourly['Code'] = None
    ce_hourly.loc[ce_hourly['DNR / DNI']==1,'Code']=1
    ce_hourly.loc[ce_hourly['Full code']==1,'Code']=2
    ce_hourly['Code'] = ce_hourly.groupby(level='id')['Code'].ffill()
    ce_hourly['DNR / DNI'] = ce_hourly['Code']==1
    ce_hourly['Full code'] = ce_hourly['Code']==2
    ce_hourly = ce_hourly.drop('Code',axis=1)

    ##-----------------------------------------------------------------------------------------------
    ##-----------------------------------------------------------------------------------------------


    # inputevents
    ie = data['inputevents'].merge(data['d_items'].loc[:,['itemid','label']],how='left',on='itemid').merge(cohort.loc[:,['hadm_id','id']],how='inner',on='hadm_id').drop('hadm_id',axis=1)
    ie['start'] = ie['starttime'].dt.floor('H')
    ie['end'] = ie['endtime'].dt.floor('H')
    ie['hourlyamount'] = ie['totalamount']/(((ie['end']-ie['start']).dt.total_seconds()/60/60)+1)

    ieitems=['Packed Red Blood Cells','Fresh Frozen Plasma','Platelets',]

    for ieitem in ieitems:
        ieslice = ie.loc[ie['label']==ieitem,['id','label','starttime','endtime','start','end','totalamount','hourlyamount',]].copy()
        (minlim,maxlim) = ieslice['hourlyamount'].quantile([.01,.99])
        ieslice = ieslice.loc[(ieslice['hourlyamount']>=minlim) & (ieslice['hourlyamount']<=maxlim),:].copy()
        ieslice['daterange'] = ieslice.apply(lambda x:pd.date_range(start=x['start'], end=x['end'],freq='H'),axis=1)
        ieslice = ieslice.loc[:,['id','hourlyamount','daterange']].explode('daterange').rename(columns={'hourlyamount':ieitem,'daterange':'charttime'}).groupby(['id','charttime']).sum()
        if ieitem in ce_hourly:
            ce_hourly = ce_hourly.drop(ieitem,axis=1)
        ce_hourly = ce_hourly.merge(ieslice,how='left',left_index=True,right_index=True)
        ce_hourly[ieitem] = ce_hourly[ieitem].fillna(0)

    ieitems ={
        'Fluid Crystalloids':'02-Fluids (Crystalloids)',
        'Fluid Colloids':'04-Fluids (Colloids)'
    }
    for ieitem in ieitems:
        ieslice = ie.loc[ie['ordercategoryname']==ieitems[ieitem],['id','label','starttime','endtime','start','end','totalamount','hourlyamount',]].copy()
        (minlim,maxlim) = ieslice['hourlyamount'].quantile([.01,.99])
        ieslice = ieslice.loc[(ieslice['hourlyamount']>=minlim) & (ieslice['hourlyamount']<=maxlim),:].copy()
        ieslice['daterange'] = ieslice.apply(lambda x:pd.date_range(start=x['start'], end=x['end'],freq='H'),axis=1)
        ieslice = ieslice.loc[:,['id','hourlyamount','daterange']].explode('daterange').rename(columns={'hourlyamount':ieitem,'daterange':'charttime'}).groupby(['id','charttime']).sum()
        if ieitem in ce_hourly:
            ce_hourly = ce_hourly.drop(ieitem,axis=1)
        ce_hourly = ce_hourly.merge(ieslice,how='left',left_index=True,right_index=True)
        ce_hourly[ieitem] = ce_hourly[ieitem].fillna(0)


       ##----------------------------------------------------------------------------------------------------------------

    # outputevents
    oe = data['outputevents'].merge(data['d_items'].loc[:,['itemid','label']],how='left',on='itemid')
    # from https://arxiv.org/pdf/1710.08531.pdf
    UrineItems = ['226559','226560','226561','226584','226563','226564','226565','226567','226557','226558','227488','227489',]
    colname = 'Urine Output'

    urine = oe.loc[oe.itemid.isin(UrineItems),:].copy()
    (minlim,maxlim) = urine['value'].quantile([.1,.99])
    urine = urine.loc[(urine['value']>=minlim) & (urine['value']<=maxlim),:].copy()

    urine = urine.loc[:,['hadm_id','charttime','value']].merge(cohort.loc[:,['id','hadm_id']],how='inner',on='hadm_id').drop('hadm_id',axis=1)
    ce_hourly_indexdf = ce_hourly.reset_index().loc[:,['id','charttime']]
    ce_hourly_indexdf['newcharttime'] = ce_hourly_indexdf['charttime']
    ce_hourly_indexdf = ce_hourly_indexdf.set_index('charttime').sort_index()
    urine = pd.merge_asof(urine.set_index('charttime').sort_index(),ce_hourly_indexdf,by='id',on='charttime',direction='nearest',tolerance=pd.Timedelta(1,'h'),allow_exact_matches=True)
    urine = urine.drop('charttime',axis=1).rename(columns={'newcharttime':'charttime','value':colname}).groupby(['id','charttime']).sum()
    urine = urine.reindex(ce_hourly_indexdf.reset_index().set_index(['id','charttime']).sort_index().index)
    urine = urine.reset_index().sort_values(by='charttime').set_index('charttime').groupby('id')['Urine Output'].rolling(window=24,min_periods=1).mean().to_frame().fillna(0)

    if colname in ce_hourly:
        ce_hourly = ce_hourly.drop(colname,axis=1)
    ce_hourly = ce_hourly.merge(urine,how='left',left_index=True,right_index=True)

    ##---------------------------------------------------------------------------------------------------------------

    return ce_hourly


####################################################
#-------- Primary outcome (mortality)---------------------------
####################################################

def mortality_outcome(ce_hourly, cohort):
    
    HORIZON_HRS = 24
    HORIZON = pd.Timedelta(HORIZON_HRS,'h')

    var_death = 'Death w/in {}h'.format(HORIZON_HRS)

    ce_hourly = ce_hourly.reset_index().merge(cohort.loc[:,['id','deathtime']],how='left',on='id')
    ce_hourly[var_death] = (ce_hourly['deathtime']-ce_hourly['charttime']) <= HORIZON
    ce_hourly = ce_hourly.set_index(['id', 'charttime'])
    ce_hourly = ce_hourly.drop('deathtime',axis=1)

    return ce_hourly


####################################################
#-------- SOFA organ scores---------------------------
####################################################

HORIZON_HRS = 24
HORIZON = pd.Timedelta(HORIZON_HRS,'h')
var_death = 'Death w/in {}h'.format(HORIZON_HRS)
   
def PlotRelationship(df,primarytask,secondarytask,bins=10):
    if df[secondarytask].dtype==bool:
        print('{}; True: {:.1%}; False: {:.1%}'.format(col,ce_hourly.loc[ce_hourly[col]==True,var_death].mean(),ce_hourly.loc[ce_hourly[col]==False,var_death].mean()))
    else:
        missingperc = df[secondarytask].isnull().mean()*100
        temp = df.loc[df[secondarytask].notnull(),[primarytask,secondarytask]].copy()
        temp['bin'] = pd.IntervalIndex(pd.qcut(temp[secondarytask], bins, duplicates='drop'))
        return temp
        temp = temp.groupby('bin')[primarytask].mean()
        if temp.shape[0]>1:
            print('ERROR')
            return
        fig,axs = plt.subplots(figsize=(6,2))
        axs.plot(range(temp.shape[0]),temp.values,marker='o')
        axs.set_xticks(range(temp.shape[0]))
        axs.set_xticklabels(['{} - {}'.format(i.left,i.right) for i in temp.index],rotation=-45,ha='left')

        color='black'
        if (temp.values.max()>0.06) & (missingperc<90):
            color='red'

        axs.set_title('{} (missing {:.1f})'.format(secondarytask,missingperc),color=color)
        axs.set_ylabel('{} %'.format(primarytask))
        axs.grid(axis='y')
        axs.set_ylim((0,0.1))
        plt.show()
        
##---------------------------------------------------------------------------------------------------

# for col in ce_hourly:
#     PlotRelationship(ce_hourly,var_death,col,bins=20)
    
##--------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------

# compare task vs. outcome
def TaskAssess(df,primarytask,secondarytask):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(df[primarytask], df[secondarytask])
    auroc = sklearn.metrics.roc_auc_score(df[primarytask], df[secondarytask])
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(df[primarytask], df[secondarytask])
    auprc = sklearn.metrics.average_precision_score(df[primarytask], df[secondarytask])
    
    calib = df.loc[:,[primarytask,secondarytask]].groupby(secondarytask)[primarytask].agg(['size','mean'])
    
    fig,axs = plt.subplots(ncols=3,nrows=1,figsize=(4*3,4))
    
    axs[0].scatter(calib.index,calib['mean'],color='red')
    twinax0 = axs[0].twinx()
    twinax0.bar(calib.index,calib['size'],color='black',alpha=0.2)
    twinax0.set_yscale('log')
    axs[0].set_title('Calibration')
    axs[0].set_xlabel(secondarytask)
    axs[0].set_ylabel(primarytask)
    
    axs[1].plot(fpr,tpr,marker='o',markersize=5)
    axs[1].set_title('AUROC: {:.3f}'.format(auroc))
    axs[1].set_xlabel('fpr')
    axs[1].set_ylabel('tpr')
    
    axs[2].plot(recall,precision,marker='o',markersize=5)
    axs[2].set_title('AUPRC: {:.3f}'.format(auprc))
    axs[2].set_xlabel('recall')
    axs[2].set_ylabel('precision')
    plt.show()
# Feature = rolling-24h max
def RollingMax(series,N=24):
    return series.groupby('id').rolling(window=N,min_periods=0).max().reset_index(level=0, drop=True)
# Task = max within 24h
def FutureMax(series,N=24):
    return series[::-1].groupby('id',sort=False).rolling(window=N,min_periods=1).max()[::-1].reset_index(level=0, drop=True)

##------------------------------------------------------------------------------------------------------------------------


def SOFA_organ_scores(ce_hourly):
    
    HORIZON_HRS = 24
    HORIZON = pd.Timedelta(HORIZON_HRS,'h')
    var_death = 'Death w/in {}h'.format(HORIZON_HRS)
    
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
    
    return ce_hourly




def saveitems(data, static, cohort, ce_hourly, npath):
    
    ############# Save items #####################

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
        
        return save
