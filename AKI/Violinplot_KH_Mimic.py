import pandas as pd
#import mimic_methods as su
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
import random
import math

# all data

diag=pd.read_csv('C:/Users/lisan/Documents/Analyse_Krankenhaus_Gesamtdaten/Validierung_mimic/Daten_endgueltig/preproc_diag.csv')
patients=pd.read_csv('C:/Users/lisan/Documents/Analyse_Krankenhaus_Gesamtdaten/Validierung_mimic/Daten_endgueltig/patients.csv')

aki_cases=pickle.load(open('E:/lbrueggemann/Neuer_Ansatz_Mimic/aki_cases.pkl','rb'))
lab=pickle.load(open('E:/lbrueggemann/Neuer_Ansatz_Mimic/lab_val.pkl','rb'))
patients=pickle.load(open('E:/lbrueggemann/Neuer_Ansatz_Mimic/patients_val.pkl','rb'))
patients=patients.drop_duplicates()
lab_24h=pickle.load(open('E:/lbrueggemann/Neuer_Ansatz_Mimic/lab_24h_val.pkl','rb'))
corr_help_24h=pickle.load(open('E:/lbrueggemann/Neuer_Ansatz_Mimic/corr_help_24h_val_nan.pkl','rb'))
non_aki_cases=pickle.load(open('E:/lbrueggemann/Neuer_Ansatz_Mimic/non_aki_cases.pkl','rb'))
aki_cases_proof=pickle.load(open('E:/lbrueggemann/Neuer_Ansatz_Mimic/aki_cases_proof.pkl','rb'))

data_m=pd.read_pickle('E:/lbrueggemann/Neuer_Ansatz_Mimic/data_val.pkl')


#wilcoxon_aki_24_m=pd.read_csv('C:/Users/lisan/Documents/Mimic_neuer_Ansatz/Daten/wilcoxon_aki_m.csv',sep=';',index_col=0)

ttest_aki_24=pd.read_csv('C:/Users/lisan/Documents/Mimic_neuer_Ansatz/Daten/ttest_aki_24.csv',sep=';',index_col=0)

lower_bound=pickle.load(open('C:/Users/lisan/Documents/Mimic_neuer_Ansatz/Daten/lower_bound.pkl','rb'))
upper_bound=pickle.load(open('C:/Users/lisan/Documents/Mimic_neuer_Ansatz/Daten/upper_bound.pkl','rb'))



# kh data

data_KH=pickle.load(open('C:/Users/lisan/Documents/Mimic_neuer_Ansatz/Modelle_Mimic_KH/Daten KH/data_KH.pkl','rb'))
aki_cases_KH=pickle.load(open('C:/Users/lisan/Documents/Mimic_neuer_Ansatz/Modelle_Mimic_KH/Daten KH/aki_cases_KH.pkl','rb'))
lower_bound_KH=pickle.load(open('C:/Users/lisan/Documents/Mimic_neuer_Ansatz/Modelle_Mimic_KH/Daten KH/lower_bound_KH.pkl','rb'))
upper_bound_KH=pickle.load(open('C:/Users/lisan/Documents/Mimic_neuer_Ansatz/Modelle_Mimic_KH/Daten KH/upper_bound_KH.pkl','rb'))
ttest_aki_24_m_KH=pickle.load(open('C:/Users/lisan/Documents/Mimic_neuer_Ansatz/Modelle_Mimic_KH/Daten KH/ttest_aki_24_m_KH.pkl','rb'))

aki_cases_KH=aki_cases_KH.tolist()


# data_KH=data_KH[['HK','Hb','MCH','MCHC','MCV','Thromb','RDW','Ery','Leuko','HCO3','Ca','Cl','BZ','Mg','K','Na','Hst','Sex','Age']]


data_KH=data_KH[['Age','Sex','HK','Hb','MCH','MCHC','MCV','Thromb','RDW','Ery','Leuko','Ca','BZ','K','Na','Hst']]


data=data[['alter', 'geschlecht', 51221, 50811, 51248, 51249, 51250, 51265, 51277, 51279, 51301, 50893, 50809, 50971, 50983, 51006]] 


# Removing patients younger than 18 years

alter=data_KH['Age']

alter=alter.to_frame()

ausschliessen=alter<=18

ausschliessen.columns = ['new_col']

aa=ausschliessen[ausschliessen['new_col']==True]

ausschliessen_list=aa.index.values.tolist() 

data_KH['index_neu'] = data_KH.index

data_KH=data_KH[[i not in ausschliessen_list for i in data_KH['index_neu']]] 

ausschliessen_list = [float(i) for i in ausschliessen_list]

aki_cases_KH = [i for i in aki_cases_KH if i not in ausschliessen_list]        
        
data_KH=data_KH.drop('index_neu', axis=1)  

# Variables:
# 51221 Hematocrit HK
# 50811 Hemoglobin Hb
# 51248 MCH
# 51249 MCHC
# 51250 MCV
# 51265 Platelet Count / Thrombozyten Thromb
# 51277 RDW
# 51279 Red Blood Cells / Erythrozyten Ery
# 51301 White Blood Cells / Leokozyten Leukp
# 50882 Bicarbonate / HCO3 raus
# 50893 Calcium Ca
# 50902 Chloride raus
# 50809 Glucose / BZ 
# 50960 Magnesium raus
# 50971 Potassium / Kalium K
# 50983 Sodium / Natrium Na
# 51006 Urea Nitrogen (umgewandelt zu Harnstoff) / Hst 

data_m=data
data=data_KH

aki_cases_m=aki_cases
aki_cases=aki_cases_KH

data_m=data_m.drop(25074766.0)

# # creatinine

# BIGGER_SIZE = 13
# plt.rc('font', size=BIGGER_SIZE+2)       
# plt.rc('axes', titlesize=27)    
# plt.rc('axes', labelsize=BIGGER_SIZE)  
# plt.rc('xtick', labelsize=BIGGER_SIZE)   
# plt.rc('ytick', labelsize=20)   
# plt.rc('legend', fontsize=BIGGER_SIZE)   
# plt.rc('figure', titlesize=20)  

# # fig, axes = plt.subplots(3,4,sharex=False,figsize=(20,11),dpi=400)
# # fig.tight_layout()

# y_help = [int(i) in np.array(aki_cases) for i in data.index]
# y_help = [int(i) for i in y_help]

# y_help_m = [int(i) in np.array(aki_cases_m) for i in data_m.index]
# y_help_m = [int(i) for i in y_help_m]

# sum=0
# for i,value in enumerate(y_help):
#     if value==0:
#         sum+=1
#         y_help[i]='Non-AKI'
#     if value==1:
#         y_help[i]='AKI'
        
# sum_m=0
# for i,value in enumerate(y_help_m):
#     if value==0:
#         sum_m+=1
#         y_help_m[i]='Non-AKI'
#     if value==1:
#         y_help_m[i]='AKI'        
  
# violin = pd.DataFrame(data['Krea'])
# violin['aki'] = y_help
# violin['aki_'] = y_help
# violin['naki_'] = y_help

# for i in range(violin.shape[0]):
#         if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
#                 violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Krea')]   
#                 violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
#         if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
#                 violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Krea')]     
#                 violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
# violin=violin.drop('Krea', axis=1)
# violin=violin.drop('aki', axis=1)


# violin_m = pd.DataFrame(data_m[50912])
# violin_m['aki'] = y_help_m
# violin_m['aki_m'] = y_help_m
# violin_m['naki_m'] = y_help_m

# for i in range(violin_m.shape[0]):
#         if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
#                 violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50912)]   
#                 violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
#         if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
#                 violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50912)]     
#                 violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
# violin_m=violin_m.drop(50912, axis=1)
# violin_m=violin_m.drop('aki', axis=1)    
    
# #data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

# violin_z=pd.concat([violin,violin_m])

# violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

# order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

# fig=sb.violinplot(data=violin_z,cut=0, order=order)
# fig.set_title('Creatinine'); fig.set_xlabel(''); fig.set_ylabel('mg/dl'); fig.set_ylim(bottom=0, top=6);



# fig.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_speichern/Krea.png')

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

y_help_m = [int(i) in np.array(aki_cases_m) for i in data_m.index]
y_help_m = [int(i) for i in y_help_m]




# urea

BIGGER_SIZE = 9
plt.rc('font', size=BIGGER_SIZE)       
plt.rc('axes', titlesize=11)    
plt.rc('axes', labelsize=BIGGER_SIZE)  
plt.rc('xtick', labelsize=BIGGER_SIZE)   
plt.rc('ytick', labelsize=14)   
plt.rc('legend', fontsize=BIGGER_SIZE)   
plt.rc('figure', titlesize=13)  

fig, axes = plt.subplots(4,4,sharex=False,figsize=(20,11),dpi=400)
fig.tight_layout()


sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Hst')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Hst')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Hst', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51006])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51006)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51006)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51006, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[0,0],data=violin_z,cut=0, order=order)
axes[0,0].set_title('Urea'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('mg/dl'); axes[0,0].set_ylim(bottom=0, top=350);




# Calcium


sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Ca')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Ca')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Ca', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50893])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50893)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50893)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50893, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[0,1],data=violin_z,cut=0, order=order)
axes[0,1].set_title('Calcium'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel('mmol/l'); axes[0,1].set_ylim(bottom=0, top=5);





# Age

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Age')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Age')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Age', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m['alter'])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc('alter')]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc('alter')]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop('alter', axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[0,2],data=violin_z,cut=0, order=order)
axes[0,2].set_title('Age'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('years');





# Hemoglobin

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Hb'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Hb')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Hb')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Hb', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50811])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50811)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50811)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50811, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[0,3],data=violin_z,cut=0, order=order)
axes[0,3].set_title('Hemoglobin'); axes[0,3].set_xlabel(''); axes[0,3].set_ylabel('g/dl'); axes[0,3].set_ylim(top=30);




# Glucose

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['BZ'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('BZ')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('BZ')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('BZ', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50809])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50809)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50809)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50809, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[1,0],data=violin_z,cut=0, order=order)
axes[1,0].set_title('Blood sugar'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('mg/dl'); axes[1,0].set_ylim(top=500,bottom=0);





# Hematocrit

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['HK'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('HK')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('HK')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('HK', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51221])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51221)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51221)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51221, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[1,1],data=violin_z,cut=0, order=order)
axes[1,1].set_title('Hematocrit'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('%'); #fig.set_ylim(top=250);




# Erythrocytes

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Ery'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Ery')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Ery')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Ery', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51279])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51279)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51279)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51279, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[1,2],data=violin_z,cut=0, order=order)
axes[1,2].set_title('Erythrocytes'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('/pl'); #fig.set_ylim(top=250);



# RDW

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['RDW'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('RDW')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('RDW')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('RDW', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51277])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51277)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51277)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51277, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[1,3],data=violin_z,cut=0, order=order)
axes[1,3].set_title('RDW'); axes[1,3].set_xlabel(''); axes[1,3].set_ylabel('%'); axes[1,3].set_ylim(top=30);





# MCH

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['MCH'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('MCH')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('MCH')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('MCH', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51248])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51248)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51248)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51248, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[2,0],data=violin_z,cut=0, order=order)
axes[2,0].set_title('MCH'); axes[2,0].set_xlabel(''); axes[2,0].set_ylabel('pg'); axes[2,0].set_ylim(top=50);





# MCHC

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['MCHC'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('MCHC')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('MCHC')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('MCHC', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51249])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51249)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51249)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51249, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[2,1],data=violin_z,cut=0, order=order)
axes[2,1].set_title('MCHC'); axes[2,1].set_xlabel(''); axes[2,1].set_ylabel('g/dl'); axes[2,1].set_ylim(top=50);





# MCV

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['MCV'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('MCV')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('MCV')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('MCV', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51250])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51250)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51250)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51250, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[2,2],data=violin_z,cut=0, order=order)
axes[2,2].set_title('MCV'); axes[2,2].set_xlabel(''); axes[2,2].set_ylabel('fl'); axes[2,2].set_ylim(top=125);


# Thromb

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Thromb'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Thromb')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Thromb')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Thromb', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51265])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51265)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51265)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51265, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[2,3],data=violin_z,cut=0, order=order)
axes[2,3].set_title('Thrombocytes'); axes[2,3].set_xlabel(''); axes[2,3].set_ylabel('K/µL'); axes[2,3].set_ylim(top=1000);




# Leuko

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Leuko'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Leuko')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Leuko')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Leuko', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51301])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51301)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51301)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51301, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[3,0],data=violin_z,cut=0, order=order)
axes[3,0].set_title('Leukocytes'); axes[3,0].set_xlabel(''); axes[3,0].set_ylabel('K/µL'); axes[3,0].set_ylim(top=50, bottom=0);



# K

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['K'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('K')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('K')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('K', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50971])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50971)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50971)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50971, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[3,1],data=violin_z,cut=0, order=order)
axes[3,1].set_title('Potassium'); axes[3,1].set_xlabel(''); axes[3,1].set_ylabel('mmol/l'); axes[3,1].set_ylim(top=10);



# Sodium

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Na'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Na')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Na')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Na', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50983])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50983)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50983)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50983, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

sb.violinplot(ax=axes[3,2],data=violin_z,cut=0, order=order)
axes[3,2].set_title('Sodium'); axes[3,2].set_xlabel(''); axes[3,2].set_ylabel('mmol/l'); axes[3,2].set_ylim(top=200);





# Sex


violin = pd.DataFrame(data['Sex'])
y_help = [int(i) in np.array(aki_cases) for i in data.index]
violin['aki'] = y_help
violin_w = violin[violin['Sex']==0]
violin_m = violin[violin['Sex']==1]
pw = violin_w[violin_w['aki']==True].shape[0]/violin_w.shape[0]
pm = violin_m[violin_m['aki']==True].shape[0]/violin_m.shape[0]

violin = pd.DataFrame(data_m['geschlecht'])
y_help = [int(i) in np.array(aki_cases_m) for i in data_m.index]
violin['aki'] = y_help
violin_w = violin[violin['geschlecht']==0]
violin_m = violin[violin['geschlecht']==1]
pw_m = violin_w[violin_w['aki']==True].shape[0]/violin_w.shape[0]
pm_m = violin_m[violin_m['aki']==True].shape[0]/violin_m.shape[0]
sb.barplot(x=['Male','Male Mimic','Female','Female Mimic'],y=[pm,pm_m,pw,pw_m],ax=axes[3,3])
axes[3,3].set_title('Sex'); axes[3,3].set_xlabel(''); axes[3,3].set_ylabel('');




fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_speichern/Violinplot_alleVariablen.png')






# violin = pd.DataFrame(data['Sex'])
# y_help = [int(i) in np.array(aki_cases) for i in data.index]
# violin['aki'] = y_help
# violin_w = violin[violin['Sex']==0]
# violin_m = violin[violin['Sex']==1]
# pw = violin_w[violin_w['aki']==True].shape[0]/violin_w.shape[0]
# pm = violin_m[violin_m['aki']==True].shape[0]/violin_m.shape[0]
# sb.barplot(x=['Male','Female'],y=[pm,pw],ax=axes[2,0])
# axes[2,0].set_title('Sex'); axes[2,0].set_xlabel(''); axes[2,0].set_ylabel('');
# y_help = [int(i) in np.array(aki_cases) for i in data.index]
# y_help = [int(i) for i in y_help]
# for i,value in enumerate(y_help):
#     if value==0:
#         y_help[i]='Non-AKI'
#     if value==1:
#         y_help[i]='AKI'
# violin = pd.DataFrame(data['RDW'])
# violin['aki'] = y_help
# sb.violinplot(ax=axes[2,1],y='RDW',x='aki',data=violin,cut=0)
# axes[2,1].set_title('RDW'); axes[2,1].set_xlabel(''); axes[2,1].set_ylabel('%');
# violin = pd.DataFrame(data['PTT'])
# violin['aki'] = y_help
# sb.violinplot(ax=axes[2,2],y='PTT',x='aki',data=violin,cut=0)
# axes[2,2].set_title('PTT'); axes[2,2].set_xlabel(''); axes[2,2].set_ylabel('sec');
# axes[2,3].axis('off')

# # Mimic


# violin = pd.DataFrame(data[51279])
# violin['aki'] = y_help
# sb.violinplot(ax=axes[1,3],y=51279,x='aki',data=violin,cut=0)
# axes[1,3].set_title('Erythrocytes'); axes[1,3].set_xlabel(''); axes[1,3].set_ylabel('/pl');
# violin = pd.DataFrame(data['gender'])
# y_help = [int(i) in np.array(aki_cases) for i in data.index]
# violin['aki'] = y_help
# violin_w = violin[violin['gender']==0]
# violin_m = violin[violin['gender']==1]
# pw = violin_w[violin_w['aki']==True].shape[0]/violin_w.shape[0]
# pm = violin_m[violin_m['aki']==True].shape[0]/violin_m.shape[0]
# sb.barplot(x=['Male','Female'],y=[pm,pw],ax=axes[2,0])
# axes[2,0].set_title('Sex'); axes[2,0].set_xlabel(''); axes[2,0].set_ylabel('');
# violin = pd.DataFrame(data[51277])
# violin['aki'] = y_help
# sb.violinplot(ax=axes[2,1],y=51277,x='aki',data=violin,cut=0)
# axes[2,1].set_title('RDW'); axes[2,1].set_xlabel(''); axes[2,1].set_ylabel('%'); axes[2,1].set_ylim(bottom=10, top=21);
# violin = pd.DataFrame(data[51275])
# violin['aki'] = y_help
# sb.violinplot(ax=axes[2,2],y=51275,x='aki',data=violin,cut=0)
# axes[2,2].set_title('PTT'); axes[2,2].set_xlabel(''); axes[2,2].set_ylabel('sec'); axes[2,2].set_ylim(bottom=0, top=199);
# axes[2,3].axis('off')


























###### Violinplots seperately  ###################################################################################








# urea

BIGGER_SIZE = 9
plt.rc('font', size=BIGGER_SIZE)       
plt.rc('axes', titlesize=11)    
plt.rc('axes', labelsize=BIGGER_SIZE)  
plt.rc('xtick', labelsize=BIGGER_SIZE)   
plt.rc('ytick', labelsize=14)   
plt.rc('legend', fontsize=BIGGER_SIZE)   
plt.rc('figure', titlesize=13)  

# fig, axes = plt.subplots(4,4,sharex=False,figsize=(20,11),dpi=400)
# fig.tight_layout()


sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Hst')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Hst')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Hst', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51006])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51006)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51006)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51006, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

# sb.violinplot(ax=axes[0,0],data=violin_z,cut=0, order=order)
# axes[0,0].set_title('Urea'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('mg/dl'); axes[0,0].set_ylim(bottom=0, top=350);

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Urea'); fig.set_xlabel(''); fig.set_ylabel('mg/dl'); fig.set_ylim(bottom=0, top=350);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/Hst.png', bbox_inches="tight")



# Calcium


sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Ca')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Ca')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Ca', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50893])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50893)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50893)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50893, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Calcium'); fig.set_xlabel(''); fig.set_ylabel('mmol/l'); fig.set_ylim(bottom=0, top=5);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/Ca.png', bbox_inches="tight")




# Age

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Age')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Age')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Age', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m['alter'])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc('alter')]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc('alter')]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop('alter', axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Age'); fig.set_xlabel(''); fig.set_ylabel('years');

fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/Alter.png', bbox_inches="tight")





# Hemoglobin

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Hb'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Hb')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Hb')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Hb', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50811])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50811)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50811)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50811, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Hemoglobin'); fig.set_xlabel(''); fig.set_ylabel('g/dl'); fig.set_ylim(top=30);

fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/Hb.png', bbox_inches="tight")




# Glucose

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['BZ'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('BZ')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('BZ')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('BZ', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50809])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50809)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50809)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50809, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Blood sugar'); fig.set_xlabel(''); fig.set_ylabel('mg/dl'); fig.set_ylim(top=500,bottom=0);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/BZ.png', bbox_inches="tight")




# Hematocrit

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['HK'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('HK')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('HK')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('HK', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51221])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51221)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51221)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51221, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Hematocrit'); fig.set_xlabel(''); fig.set_ylabel('%'); #fig.set_ylim(top=250);

fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/HK.png', bbox_inches="tight")



# Ery

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Ery'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Ery')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Ery')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Ery', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51279])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51279)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51279)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51279, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Erythrocytes'); fig.set_xlabel(''); fig.set_ylabel('/pl'); #fig.set_ylim(top=250);

fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/Ery.png', bbox_inches="tight")



# RDW

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['RDW'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('RDW')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('RDW')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('RDW', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51277])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51277)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51277)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51277, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('RDW'); fig.set_xlabel(''); fig.set_ylabel('%'); fig.set_ylim(top=30);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/RDW.png', bbox_inches="tight")



# MCH

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['MCH'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('MCH')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('MCH')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('MCH', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51248])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51248)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51248)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51248, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('MCH'); fig.set_xlabel(''); fig.set_ylabel('pg'); fig.set_ylim(top=50);

fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/MCH.png', bbox_inches="tight")




# MCHC

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['MCHC'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('MCHC')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('MCHC')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('MCHC', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51249])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51249)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51249)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51249, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('MCHC'); fig.set_xlabel(''); fig.set_ylabel('g/dl'); fig.set_ylim(top=50);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/MCHC.png', bbox_inches="tight")



# MCV

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['MCV'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('MCV')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('MCV')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('MCV', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51250])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51250)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51250)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51250, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('MCV'); fig.set_xlabel(''); fig.set_ylabel('fl'); fig.set_ylim(top=125);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/MCV.png', bbox_inches="tight")



# Thromb

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Thromb'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Thromb')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Thromb')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Thromb', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51265])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51265)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51265)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51265, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Thrombocytes'); fig.set_xlabel(''); fig.set_ylabel('K/µL'); fig.set_ylim(top=1000);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/Thromb.png', bbox_inches="tight")




# Leuko

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Leuko'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Leuko')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Leuko')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Leuko', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[51301])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51301)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(51301)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(51301, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Leukocytes'); fig.set_xlabel(''); fig.set_ylabel('K/µL'); fig.set_ylim(top=50, bottom=0);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/Leuko.png', bbox_inches="tight")



# K

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['K'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('K')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('K')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('K', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50971])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50971)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50971)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50971, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Potassium'); fig.set_xlabel(''); fig.set_ylabel('mmol/l'); fig.set_ylim(top=10);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/K.png', bbox_inches="tight")



# Sodium

sum=0
for i,value in enumerate(y_help):
    if value==0:
        sum+=1
        y_help[i]='Non-AKI'
    if value==1:
        y_help[i]='AKI'
        
sum_m=0
for i,value in enumerate(y_help_m):
    if value==0:
        sum_m+=1
        y_help_m[i]='Non-AKI'
    if value==1:
        y_help_m[i]='AKI'        
  
violin = pd.DataFrame(data['Na'])
violin['aki'] = y_help
violin['aki_'] = y_help
violin['naki_'] = y_help

for i in range(violin.shape[0]):
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'Non-AKI':
                violin.iloc[i,violin.columns.get_loc('naki_')] = violin.iloc[i,violin.columns.get_loc('Na')]   
                violin.iloc[i,violin.columns.get_loc('aki_')] = np.nan
        if violin.iloc[i,violin.columns.get_loc('aki')] == 'AKI':
                violin.iloc[i,violin.columns.get_loc('aki_')] = violin.iloc[i,violin.columns.get_loc('Na')]     
                violin.iloc[i,violin.columns.get_loc('naki_')] = np.nan
violin=violin.drop('Na', axis=1)
violin=violin.drop('aki', axis=1)


violin_m = pd.DataFrame(data_m[50983])
violin_m['aki'] = y_help_m
violin_m['aki_m'] = y_help_m
violin_m['naki_m'] = y_help_m

for i in range(violin_m.shape[0]):
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'Non-AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50983)]   
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = np.nan
        if violin_m.iloc[i,violin_m.columns.get_loc('aki')] == 'AKI':
                violin_m.iloc[i,violin_m.columns.get_loc('aki_m')] = violin_m.iloc[i,violin_m.columns.get_loc(50983)]     
                violin_m.iloc[i,violin_m.columns.get_loc('naki_m')] = np.nan
violin_m=violin_m.drop(50983, axis=1)
violin_m=violin_m.drop('aki', axis=1)    
    
#data.rename(columns={'alter':'anchor_age', 'geschlecht':'gender'}, inplace=True)

violin_z=pd.concat([violin,violin_m])

violin_z.rename(columns={'aki_':'AKI', 'aki_m':'AKI Mimic', 'naki_':'NON-AKI', 'naki_m':'NON-AKI Mimic'}, inplace=True)

order=["AKI", "AKI Mimic", "NON-AKI", "NON-AKI Mimic"]

fig=sb.violinplot(data=violin_z,cut=0, order=order)
fig.set_title('Sodium'); fig.set_xlabel(''); fig.set_ylabel('mmol/l'); fig.set_ylim(top=200);


fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/Na.png', bbox_inches="tight")



# Sex


violin = pd.DataFrame(data['Sex'])
y_help = [int(i) in np.array(aki_cases) for i in data.index]
violin['aki'] = y_help
violin_w = violin[violin['Sex']==0]
violin_m = violin[violin['Sex']==1]
pw = violin_w[violin_w['aki']==True].shape[0]/violin_w.shape[0]
pm = violin_m[violin_m['aki']==True].shape[0]/violin_m.shape[0]

violin = pd.DataFrame(data_m['geschlecht'])
y_help = [int(i) in np.array(aki_cases_m) for i in data_m.index]
violin['aki'] = y_help
violin_w = violin[violin['geschlecht']==0]
violin_m = violin[violin['geschlecht']==1]
pw_m = violin_w[violin_w['aki']==True].shape[0]/violin_w.shape[0]
pm_m = violin_m[violin_m['aki']==True].shape[0]/violin_m.shape[0]
fig=sb.barplot(x=['Male','Male Mimic','Female','Female Mimic'],y=[pm,pm_m,pw,pw_m])
fig.set_title('Sex'); fig.set_xlabel(''); fig.set_ylabel('');




fig.figure.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Violinplots_Vergleich/Violinplots_final/Sex.png', bbox_inches="tight")

















