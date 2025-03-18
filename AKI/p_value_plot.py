import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

wilcoxon = pd.read_csv(r'C:\Users\wendland\Documents\GitHub\GTT-Trigger-tool\Lisanne\Tests\wilcoxon_aki_lab_24h_m.csv',sep=';')
ttest = pd.read_csv(r'C:\Users\wendland\Documents\GitHub\GTT-Trigger-tool\Lisanne\Tests\ttest_aki_lab_24h_m.csv',sep=';')

wilcoxon["q-value"].iloc[0:46]=np.log(wilcoxon["q-value"].iloc[0:46])
for i in range(46):
    if ttest[ttest["Labvalue"]==wilcoxon["Labvalue"].iloc[i]]["Statistic"].values[0] > 0:
        wilcoxon["q-value"].iloc[i]=-wilcoxon["q-value"].iloc[i]

fig = plt.figure(dpi=400, figsize=(21,10))
x_labels = ['Creatinine','Urea','GFR','CRP','RDW','INR','Quick','Calcium','Ery','Glucose','PTT','Hb','TropThs','MCHC','HCT','GGT','iCalcium','MCV','FT3','NTpBNp','LDH','Mg','Neutro','UpH','Sodium']
#x=x[:25]
ax = wilcoxon.iloc[0:25].plot.bar(x='Labvalue', y='q-value',label='', rot=45, ax = plt.gca())
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
#ax.set_xticklabels(wilcoxon["Labvalue"].values[:25])
ax.set_xticklabels(x_labels)
plt.legend()
BIGGER_SIZE=20
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
plt.ylabel('Association')
plt.xlabel('Laboratory Value')
ax.xaxis.set_label_coords(x=0.5,y= -0.125)
plt.savefig('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/aki_p_wert_plot.jpg')

fig = plt.figure(dpi=400, figsize=(21,10))
x_labels = ['GOT','Potassium','O2-Sat','PCT','CK-MB','D-Dim','U-Leuc','Platelets','HDL','Leuco','U-Ery','T_Bil','NRBC','MCH','D_Bil','TP','Monoc','TC','Albumin','CHE','p50Tc']
#x=x[:25]
ax = wilcoxon.iloc[25:46].plot.bar(x='Labvalue', y='q-value',label='', rot=45, ax = plt.gca())
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
#ax.set_xticklabels(wilcoxon["Labvalue"].values[25:46])
ax.set_xticklabels(x_labels)
plt.legend()
BIGGER_SIZE=20
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
plt.ylabel('Association')
plt.xlabel('Laboratory Value')
#plt.ylim([-400,400])
ax.xaxis.set_label_coords(x=0.5,y= -0.125)
plt.savefig('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/aki_p_wert_plot2.jpg')
