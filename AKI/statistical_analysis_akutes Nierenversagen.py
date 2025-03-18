import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
import random
import statistical_methods_akutes_Nierenversagen as su
from sklearn.model_selection import train_test_split

list_numbercase = pickle.load(open('/home/lbrueggemann/tests/list_numbercase.pkl','rb'))
lab = pickle.load(open('/home/lbrueggemann/tests/lab.pkl','rb'))
bew = pickle.load(open('/home/lbrueggemann/tests/bew.pkl','rb'))
base = pickle.load(open('/home/lbrueggemann/tests/base.pkl','rb'))

# Determination of cases with acute kidney failure and cases without acute kidney failure.
aki_cases=np.unique(list_numbercase)
non_aki_cases=base['Number case'].unique()[np.in1d(base['Number case'].unique(),aki_cases,invert=True)]

random.seed(5422)
non_aki_cases=random.sample(list(non_aki_cases),5422)
non_aki_cases.sort()
non_aki_cases=np.array(non_aki_cases)

lab=lab[[i in np.concatenate([aki_cases,non_aki_cases]) for i in lab['Number case']]]
base = base[[i in np.concatenate([aki_cases,non_aki_cases]) for i in base['Number case']]]
bew = bew[[i in np.concatenate([aki_cases,non_aki_cases]) for i in bew['Number case']]]

# Bestimmung des Mittelwertes der Laborwerte der ersten 24h
lab_24h, corr_help_24h = su.lab_24h(lab,base['Number case'],bew)

# Wilcoxon-Rangsummentest:
wilcoxon_aki_24=su.wilcoxon_without_BZ(lab_24h, corr_help_24h.columns, aki_cases, non_aki_cases)
wilcoxon_aki_24=su.wilcoxon_BZ(lab_24h, wilcoxon_aki_24, aki_cases, non_aki_cases)
wilcoxon_aki_24 = su.wilcoxon_multiple_test(wilcoxon_aki_24)

wilcoxon_aki_24.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/wilcoxon_aki_24h.csv',sep=';')

# t Test:
ttest_aki_24 = su.ttest_without_BZ(lab_24h, corr_help_24h.columns, aki_cases, non_aki_cases)
ttest_aki_24 = su.ttest_BZ(lab_24h, ttest_aki_24, aki_cases, non_aki_cases)
ttest_aki_24 = su.ttest_multiple_test(ttest_aki_24)

ttest_aki_24.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/ttest_aki_24h.csv',sep=';')

# ############### Dateien vom Server einlesen ################
t1=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_1.csv',sep=';',index_col=0)
t2=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_2.csv',sep=';',index_col=0)
t3=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_3.csv',sep=';',index_col=0)
t4=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_4.csv',sep=';',index_col=0)
t5=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_5.csv',sep=';',index_col=0)
t6=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_6.csv',sep=';',index_col=0)
t7=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_7.csv',sep=';',index_col=0)
t8=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_8.csv',sep=';',index_col=0)
t9=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_9.csv',sep=';',index_col=0)
t10=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_10.csv',sep=';',index_col=0)
t11=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_11.csv',sep=';',index_col=0)
t12=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_12.csv',sep=';',index_col=0)
t13=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_13.csv',sep=';',index_col=0)
t14=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_14.csv',sep=';',index_col=0)
t15=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_15.csv',sep=';',index_col=0)
t16=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_16.csv',sep=';',index_col=0)
t17=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_17.csv',sep=';',index_col=0)
t18=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_18.csv',sep=';',index_col=0)
t19=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_19.csv',sep=';',index_col=0)
t20=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_20.csv',sep=';',index_col=0)
t21=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_21.csv',sep=';',index_col=0)
t22=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_22.csv',sep=';',index_col=0)
t23=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_23.csv',sep=';',index_col=0)
t24=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_24.csv',sep=';',index_col=0)
t25=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_25.csv',sep=';',index_col=0)
t26=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_26.csv',sep=';',index_col=0)
t27=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_27.csv',sep=';',index_col=0)
t28=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_28.csv',sep=';',index_col=0)
t29=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_29.csv',sep=';',index_col=0)
t30=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_30.csv',sep=';',index_col=0)
t31=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_31.csv',sep=';',index_col=0)
t32=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_32.csv',sep=';',index_col=0)
t33=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_33.csv',sep=';',index_col=0)
t34=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_34.csv',sep=';',index_col=0)
t35=pd.read_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_24h_35.csv',sep=';',index_col=0)

ttest_aki_24=pd.concat([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,t32,t33,t34,t35])
ttest_aki_24=ttest_aki_24.drop_duplicates()
ttest_aki_24= ttest_aki_24[:][ttest_aki_24['P-value']!='--']
for i in range(ttest_aki_24.shape[0]):
    ttest_aki_24.iloc[i,ttest_aki_24.columns.get_loc('Statistic')]=float(ttest_aki_24.iloc[i,ttest_aki_24.columns.get_loc('Statistic')])
    ttest_aki_24.iloc[i,ttest_aki_24.columns.get_loc('P-value')]=float(ttest_aki_24.iloc[i,ttest_aki_24.columns.get_loc('P-value')])

ttest_aki_24_m=su.ttest_multiple_test(ttest_aki_24)

ttest_aki_24_m.to_csv('C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/ttest_aki_lab_24h_m.csv',sep=';')

w1=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_1.csv',sep=';',index_col=0)
w2=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_2.csv',sep=';',index_col=0)
w3=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_3.csv',sep=';',index_col=0)
w4=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_4.csv',sep=';',index_col=0)
w5=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_5.csv',sep=';',index_col=0)
w6=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_6.csv',sep=';',index_col=0)
w7=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_7.csv',sep=';',index_col=0)
w8=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_8.csv',sep=';',index_col=0)
w9=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_9.csv',sep=';',index_col=0)
w10=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_10.csv',sep=';',index_col=0)
w11=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_11.csv',sep=';',index_col=0)
w12=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_12.csv',sep=';',index_col=0)
w13=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_13.csv',sep=';',index_col=0)
w14=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_14.csv',sep=';',index_col=0)
w15=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_15.csv',sep=';',index_col=0)
w16=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_16.csv',sep=';',index_col=0)
w17=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_17.csv',sep=';',index_col=0)
w18=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_18.csv',sep=';',index_col=0)
w19=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_19.csv',sep=';',index_col=0)
w20=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_20.csv',sep=';',index_col=0)
w21=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_21.csv',sep=';',index_col=0)
w22=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_22.csv',sep=';',index_col=0)
w23=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_23.csv',sep=';',index_col=0)
w24=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_24.csv',sep=';',index_col=0)
w25=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_25.csv',sep=';',index_col=0)
w26=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_26.csv',sep=';',index_col=0)
w27=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_27.csv',sep=';',index_col=0)
w28=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_28.csv',sep=';',index_col=0)
w29=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_29.csv',sep=';',index_col=0)
w30=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_30.csv',sep=';',index_col=0)
w31=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_31.csv',sep=';',index_col=0)
w32=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_32.csv',sep=';',index_col=0)
w33=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_33.csv',sep=';',index_col=0)
w34=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_34.csv',sep=';',index_col=0)
w35=pd.read_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_24h_35.csv',sep=';',index_col=0)

wilcoxon_aki_24=pd.concat([w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25,w26,w27,w28,w29,w30,w31,w32,w33,w34,w35])
# wilcoxon_aki_24=wilcoxon_aki_24.drop('q-value', axis=1)
# wilcoxon_aki_24=wilcoxon_aki_24.drop('fdr', axis=1)
wilcoxon_aki_24=wilcoxon_aki_24.drop_duplicates()

wilcoxon_aki_24_m = su.wilcoxon_multiple_test(wilcoxon_aki_24)

wilcoxon_aki_24_m.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Tests/Test_Labordaten_richtige_Kreatininabfrage/wilcoxon_aki_lab_24h_m.csv',sep=';')


### Histogram for p-values of the Wilcoxon rank-sum test.
wilcoxon = pd.read_csv(r'C:\Users\wendland\Documents\GitHub\GTT-Trigger-tool\Lisanne\Tests\Test_Labordaten_richtige_Kreatininabfrage\wilcoxon_aki_lab_24h_m.csv',sep=';')
wilcoxon.rename(columns={'q-value': 'adjusted p-value'}, inplace=True)

fig = plt.figure(dpi=400, figsize=(21,10))
ax = wilcoxon.iloc[0:20].plot.bar(x='Labvalue', y='adjusted p-value', rot=45, ax = plt.gca())
BIGGER_SIZE=25
plt.xlabel('Laborwerte')
plt.ylabel('adjustierte p-Werte')
ax.get_legend().remove()
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE)
plt.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/BarplotFeature/wilcoxon_aki_lab_24h_holm_neu.jpg')

#Logarithmizing the p-values.
wilcoxon = pd.read_csv(r'C:\Users\lisan\Documents\GitHub\GTT-Trigger-tool\Lisanne\Tests\Test_Labordaten_richtige_Kreatininabfrage\wilcoxon_aki_lab_24h_m.csv',sep=';')
wilcoxon.rename(columns={'q-value': 'adjusted p-value'}, inplace=True)
wilcoxon['adjusted p-value1']=-np.log(wilcoxon['adjusted p-value'])

fig = plt.figure(dpi=400, figsize=(21,10))
ax = wilcoxon.iloc[0:20].plot.bar(x='Labvalue', y='adjusted p-value', rot=45, ax = plt.gca())
BIGGER_SIZE=25
plt.xlabel('Laborwerte')
plt.ylabel('-log(p-Werte)')
ax.get_legend().remove()
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE)

plt.savefig('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/BarplotFeature/wilcoxon_aki_lab_24h_holm_neu.jpg')


# Sex and Age als Variablen hinzufügen
data=corr_help_24h
base_aki = base[[i in data.index.astype(int) for i in base['Number case']]]
base_aki = base_aki.drop_duplicates(subset=['Number case'])
sex = [int(i) for i in base_aki['Sex']=='M']
sex = pd.Series(sex,name='Sex',index=base_aki['Number case'])
age = base_aki['Age'].astype(float)
age.index = base_aki['Number case']
data = pd.concat([data,sex],axis=1)
data = pd.concat([data,age],axis=1)


#######################################################################################
######################## Logistische Regression #######################################
#######################################################################################

##################### für die 10 Laborwerte ##########################
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)

pickle.dump(log_reg, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/log_reg_lab10.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_pred_lab10.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_test_lab10.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/Xtest_aki_lab10.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_pred_aki_lab10.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/ytest_aki_lab10.csv')

###############Violinplot################

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)        
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE)   
plt.rc('xtick', labelsize=BIGGER_SIZE)   
plt.rc('ytick', labelsize=25)    
plt.rc('legend', fontsize=BIGGER_SIZE)    
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(2,4,sharex=False,figsize=(20,11),dpi=400) 
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
# violin=violin.dropna()
sb.violinplot(ax=axes[0,0],y='Age',x='aki',data=violin,cut=0)
axes[0,0].set_title('Alter'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
# violin=violin.dropna()
sb.violinplot(ax=axes[0,1],y='Hst',x='aki',data=violin,cut=0)
axes[0,1].set_title('Hst'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel(''); 
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
# violin=violin.dropna()
sb.violinplot(ax=axes[0,2],y='Ca',x='aki',data=violin,cut=0)
axes[0,2].set_title('Ca'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('');
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
# violin=violin.dropna()
sb.violinplot(ax=axes[0,3],y='CRP',x='aki',data=violin,cut=0)
axes[0,3].set_title('CRP'); axes[0,3].set_xlabel(''); axes[0,3].set_ylabel('');   
violin = pd.DataFrame(data['Ery'])
violin['aki'] = y_help
# violin=violin.dropna()
sb.violinplot(ax=axes[1,0],y='Ery',x='aki',data=violin,cut=0)
axes[1,0].set_title('Ery'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['PTT'])
violin['aki'] = y_help
# violin=violin.dropna()
sb.violinplot(ax=axes[1,1],y='PTT',x='aki',data=violin,cut=0)
axes[1,1].set_title('PTT'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
violin = pd.DataFrame(data['BZ'])
violin['aki'] = y_help
# violin=violin.dropna()
sb.violinplot(ax=axes[1,2],y='BZ',x='aki',data=violin,cut=0)
axes[1,2].set_title('BZ'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('');
violin = pd.DataFrame(data['Quick'])
violin['aki'] = y_help
# violin=violin.dropna()
sb.violinplot(ax=axes[1,3],y='Quick',x='aki',data=violin,cut=0)
axes[1,3].set_title('Quick'); axes[1,3].set_xlabel(''); axes[1,3].set_ylabel('');

###################### 20 Laboratory values ########################

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)

pickle.dump(log_reg, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/log_reg_lab20.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_pred_lab20.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_test_lab20.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/Xtest_aki_lab20.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_pred_aki_lab20.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/ytest_aki_lab20.csv')


###############Violinplot################

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)       
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE)  
plt.rc('xtick', labelsize=BIGGER_SIZE)   
plt.rc('ytick', labelsize=25)   
plt.rc('legend', fontsize=BIGGER_SIZE)   
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(3,4,sharex=False,figsize=(20,11),dpi=400)
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Age',x='aki',data=violin,cut=0)
axes[0,0].set_title('Alter'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='Hst',x='aki',data=violin,cut=0)
axes[0,1].set_title('Hst'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel(''); 
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,2],y='Ca',x='aki',data=violin,cut=0)
axes[0,2].set_title('Ca'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('');
violin = pd.DataFrame(data['Leuko'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,3],y='Leuko',x='aki',data=violin,cut=0)
axes[0,3].set_title('Leuko'); axes[0,3].set_xlabel(''); axes[0,3].set_ylabel('');   
violin = pd.DataFrame(data['Na'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='Na',x='aki',data=violin,cut=0)
axes[1,0].set_title('Na'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['GOT'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='GOT',x='aki',data=violin,cut=0)
axes[1,1].set_title('GOT'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,2],y='CRP',x='aki',data=violin,cut=0)
axes[1,2].set_title('CRP'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('');
violin = pd.DataFrame(data['Quick'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,3],y='Quick',x='aki',data=violin,cut=0)
axes[1,3].set_title('Quick'); axes[1,3].set_xlabel(''); axes[1,3].set_ylabel('');
violin = pd.DataFrame(data['RDW'])
violin['aki'] = y_help    
sb.violinplot(ax=axes[2,0],y='RDW',x='aki',data=violin,cut=0)
axes[2,0].set_title('RDW'); axes[2,0].set_xlabel(''); axes[2,0].set_ylabel('');
violin = pd.DataFrame(data['MCV'])
violin['aki'] = y_help
sb.violinplot(ax=axes[2,1],y='MCV',x='aki',data=violin,cut=0)
axes[2,1].set_title('MCV'); axes[2,1].set_xlabel(''); axes[2,1].set_ylabel('');
violin = pd.DataFrame(data['MCH'])
violin['aki'] = y_help
sb.violinplot(ax=axes[2,2],y='MCH',x='aki',data=violin,cut=0)
axes[2,2].set_title('MCH'); axes[2,2].set_xlabel(''); axes[2,2].set_ylabel('');
axes[2,3].axis('off')


############################################
# dichotomized variables 

lower_bound = pd.DataFrame(index=data.index,columns=data.columns)
upper_bound = pd.DataFrame(index=data.index,columns=data.columns)
row, col = -1, -1
for i in data.index:
    row = row+1
    print(row)
    col = -1
    for j in data.columns:
        col = col+1
        temp = lab[lab['Number case']==i].drop_duplicates(subset=['Lab_name','lower_ref','upper_ref'])
        if temp[temp['Lab_name']==j]['lower_ref'].size!=0:
            lower_bound.iloc[row,col] = temp[temp['Lab_name']==j]['lower_ref'].iloc[0]
        if temp[temp['Lab_name']==j]['upper_ref'].size!=0:
            upper_bound.iloc[row,col] = temp[temp['Lab_name']==j]['upper_ref'].iloc[0]
upper_bound['GFR'][upper_bound['GFR']=='no upper ref']=np.nan

#10
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)

pickle.dump(log_reg, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/log_reg_lab10_k.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_pred_lab10_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_test_lab10_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/Xtest_aki_lab10_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_pred_aki_lab10_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/ytest_aki_lab10_k.csv')

#20
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors

pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)

pickle.dump(log_reg, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/log_reg_lab20_k.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_pred_lab20_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_test_lab20_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/Xtest_aki_lab20_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/y_pred_aki_lab20_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_LogReg/Ergebnisse_richtige Kreatininabfrage/ytest_aki_lab20_k.csv')


###############Violinplot################

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)         
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE)    
plt.rc('xtick', labelsize=BIGGER_SIZE)   
plt.rc('ytick', labelsize=25)   
plt.rc('legend', fontsize=BIGGER_SIZE)  
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(2,4,sharex=False,figsize=(20,11),dpi=400)
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Hst',x='aki',data=violin,cut=0)
axes[0,0].set_title('Hst'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='CRP',x='aki',data=violin,cut=0)
axes[0,1].set_title('CRP'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel(''); 
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,2],y='Ca',x='aki',data=violin,cut=0)
axes[0,2].set_title('Ca'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('');
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,3],y='Age',x='aki',data=violin,cut=0)
axes[0,3].set_title('Alter'); axes[0,3].set_xlabel(''); axes[0,3].set_ylabel('');   
violin = pd.DataFrame(data['Ery'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='Ery',x='aki',data=violin,cut=0)
axes[1,0].set_title('Ery'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['GFR'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='GFR',x='aki',data=violin,cut=0)
axes[1,1].set_title('GFR'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
violin = pd.DataFrame(data['PTT'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,2],y='PTT',x='aki',data=violin,cut=0)
axes[1,2].set_title('PTT'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('');
axes[1,3].axis('off')


BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)      
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE)  
plt.rc('xtick', labelsize=BIGGER_SIZE)    
plt.rc('ytick', labelsize=25)  
plt.rc('legend', fontsize=BIGGER_SIZE)   
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(3,4,sharex=False,figsize=(20,11),dpi=400)
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Hst',x='aki',data=violin,cut=0)
axes[0,0].set_title('Hst'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='CRP',x='aki',data=violin,cut=0)
axes[0,1].set_title('CRP'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel(''); 
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,2],y='Age',x='aki',data=violin,cut=0)
axes[0,2].set_title('Alter'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('');
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,3],y='Ca',x='aki',data=violin,cut=0)
axes[0,3].set_title('Ca'); axes[0,3].set_xlabel(''); axes[0,3].set_ylabel('');   
violin = pd.DataFrame(data['GFR'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='GFR',x='aki',data=violin,cut=0)
axes[1,0].set_title('GFR'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['Na'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='Na',x='aki',data=violin,cut=0)
axes[1,1].set_title('Na'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
violin = pd.DataFrame(data['K'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,2],y='K',x='aki',data=violin,cut=0)
axes[1,2].set_title('K'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('');
violin = pd.DataFrame(data['Leuko'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,3],y='Leuko',x='aki',data=violin,cut=0)
axes[1,3].set_title('Leuko'); axes[1,3].set_xlabel(''); axes[1,3].set_ylabel('');
violin = pd.DataFrame(data['Thromb'])
violin['aki'] = y_help    
sb.violinplot(ax=axes[2,0],y='Thromb',x='aki',data=violin,cut=0)
axes[2,0].set_title('Thromb'); axes[2,0].set_xlabel(''); axes[2,0].set_ylabel('');
violin = pd.DataFrame(data['PTT'])
violin['aki'] = y_help
sb.violinplot(ax=axes[2,1],y='PTT',x='aki',data=violin,cut=0)
axes[2,1].set_title('PTT'); axes[2,1].set_xlabel(''); axes[2,1].set_ylabel('');
violin = pd.DataFrame(data['GOT'])
violin['aki'] = y_help
sb.violinplot(ax=axes[2,2],y='GOT',x='aki',data=violin,cut=0)
axes[2,2].set_title('GOT'); axes[2,2].set_xlabel(''); axes[2,2].set_ylabel('');
violin = pd.DataFrame(data['HK'])
violin['aki'] = y_help
sb.violinplot(ax=axes[2,3],y='HK',x='aki',data=violin,cut=0)
axes[2,3].set_title('HK'); axes[2,3].set_xlabel(''); axes[2,3].set_ylabel('');


#######################################################################################
############################# Random Forest  ##########################################
#######################################################################################

############### für die top 10 Laborwerte ##################
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result=su.RandomForestFeatSelection_cv(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.RandomForestHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/model_lab10_RF.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_lab10_RF.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_test_lab10_RF.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/Xtest_aki_lab10_RF.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_aki_lab10_RF.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/ytest_aki_lab10_RF.csv')

###############Violinplot################

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)        
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE)   
plt.rc('xtick', labelsize=BIGGER_SIZE)   
plt.rc('ytick', labelsize=25)  
plt.rc('legend', fontsize=BIGGER_SIZE)  
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(2,4,sharex=False,figsize=(20,11),dpi=400) 
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Age',x='aki',data=violin,cut=0)
axes[0,0].set_title('Alter'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='Hst',x='aki',data=violin,cut=0)
axes[0,1].set_title('Hst'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel(''); 
violin = pd.DataFrame(data['GFR'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,2],y='GFR',x='aki',data=violin,cut=0)
axes[0,2].set_title('GFR'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('');
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,3],y='CRP',x='aki',data=violin,cut=0)
axes[0,3].set_title('CRP'); axes[0,3].set_xlabel(''); axes[0,3].set_ylabel('');   
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='Ca',x='aki',data=violin,cut=0)
axes[1,0].set_title('Ca'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['Ery'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='Ery',x='aki',data=violin,cut=0)
axes[1,1].set_title('Ery'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
violin = pd.DataFrame(data['Hb'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,2],y='Hb',x='aki',data=violin,cut=0)
axes[1,2].set_title('Hb'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('');
axes[1,3].axis('off')

############### Top 20 Laboratory values ##################
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result=su.RandomForestFeatSelection_cv(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.RandomForestHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/model_lab20_RF.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_lab20_RF.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_test_lab20_RF.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/Xtest_aki_lab20_RF.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_aki_lab20_RF.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/ytest_aki_lab20_RF.csv')

###############Violinplot################

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)        
plt.rc('axes', titlesize=27)   
plt.rc('axes', labelsize=BIGGER_SIZE)  
plt.rc('xtick', labelsize=BIGGER_SIZE)   
plt.rc('ytick', labelsize=25)    
plt.rc('legend', fontsize=BIGGER_SIZE)  
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(2,4,sharex=False,figsize=(20,11),dpi=400)
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Age',x='aki',data=violin,cut=0)
axes[0,0].set_title('Alter'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='Hst',x='aki',data=violin,cut=0)
axes[0,1].set_title('Hst'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel(''); 
violin = pd.DataFrame(data['GFR'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,2],y='GFR',x='aki',data=violin,cut=0)
axes[0,2].set_title('GFR'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('');
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,3],y='CRP',x='aki',data=violin,cut=0)
axes[0,3].set_title('CRP'); axes[0,3].set_xlabel(''); axes[0,3].set_ylabel('');   
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='Ca',x='aki',data=violin,cut=0)
axes[1,0].set_title('Ca'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['Hb'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='Hb',x='aki',data=violin,cut=0)
axes[1,1].set_title('Hb'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
violin = pd.DataFrame(data['Leuko'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,2],y='Leuko',x='aki',data=violin,cut=0)
axes[1,2].set_title('Leuko'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('');
axes[1,3].axis('off')

############################################
# Dichotomized variables 

lower_bound = pd.DataFrame(index=data.index,columns=data.columns)
upper_bound = pd.DataFrame(index=data.index,columns=data.columns)
row, col = -1, -1
for i in data.index:
    row = row+1
    print(row)
    col = -1
    for j in data.columns:
        col = col+1
        temp = lab[lab['Number case']==i].drop_duplicates(subset=['Lab_name','lower_ref','upper_ref'])
        if temp[temp['Lab_name']==j]['lower_ref'].size!=0:
            lower_bound.iloc[row,col] = temp[temp['Lab_name']==j]['lower_ref'].iloc[0]
        if temp[temp['Lab_name']==j]['upper_ref'].size!=0:
            upper_bound.iloc[row,col] = temp[temp['Lab_name']==j]['upper_ref'].iloc[0]
upper_bound['GFR'][upper_bound['GFR']=='no upper ref']=np.nan

#10
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result=su.RandomForestFeatSelection_cv(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.RandomForestHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/model_lab10_RF_k.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_lab10_RF_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_test_lab10_RF_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/Xtest_aki_lab10_RF_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_aki_lab10_RF_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/ytest_aki_lab10_RF_k.csv')

#20

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors

pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result=su.RandomForestFeatSelection_cv(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.RandomForestHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/model_lab20_RF_k.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_lab20_RF_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_test_lab20_RF_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/Xtest_aki_lab20_RF_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_aki_lab20_RF_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/ytest_aki_lab20_RF_k.csv')

###############Violinplot################

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)         
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE)    
plt.rc('xtick', labelsize=BIGGER_SIZE)  
plt.rc('ytick', labelsize=25)    
plt.rc('legend', fontsize=BIGGER_SIZE)  
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(2,2,sharex=False,figsize=(20,11),dpi=400) 
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Hst',x='aki',data=violin,cut=0)
axes[0,0].set_title('Hst'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['GFR'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='GFR',x='aki',data=violin,cut=0)
axes[0,1].set_title('GFR'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel('');  
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='CRP',x='aki',data=violin,cut=0)
axes[1,0].set_title('CRP'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='Age',x='aki',data=violin,cut=0)
axes[1,1].set_title('Alter'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
  

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)       
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE)  
plt.rc('xtick', labelsize=BIGGER_SIZE)   
plt.rc('ytick', labelsize=25)   
plt.rc('legend', fontsize=BIGGER_SIZE) 
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(3,3,sharex=False,figsize=(20,11),dpi=400) 
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Hst',x='aki',data=violin,cut=0)
axes[0,0].set_title('Hst'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['GFR'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='GFR',x='aki',data=violin,cut=0)
axes[0,1].set_title('GFR'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel(''); 
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,2],y='CRP',x='aki',data=violin,cut=0)
axes[0,2].set_title('CRP'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('');
violin = pd.DataFrame(data['Quick'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='Quick',x='aki',data=violin,cut=0)
axes[1,0].set_title('Quick'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');   
violin = pd.DataFrame(data['RDW'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='RDW',x='aki',data=violin,cut=0)
axes[1,1].set_title('RDW'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
violin = pd.DataFrame(data['GOT'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,2],y='GOT',x='aki',data=violin,cut=0)
axes[1,2].set_title('GOT'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('');
violin = pd.DataFrame(data['Leuko'])
violin['aki'] = y_help
sb.violinplot(ax=axes[2,0],y='Leuko',x='aki',data=violin,cut=0)
axes[2,0].set_title('Leuko'); axes[2,0].set_xlabel(''); axes[2,0].set_ylabel('');
violin = pd.DataFrame(data['Sex'])
y_help = [int(i) in np.array(aki_cases) for i in data.index]
violin['aki'] = y_help
violin_w = violin[violin['Sex']==0]
violin_m = violin[violin['Sex']==1]
pw = violin_w[violin_w['aki']==True].shape[0]/violin_w.shape[0]
pm = violin_m[violin_m['aki']==True].shape[0]/violin_m.shape[0]
sb.barplot(['Männlich','Weiblich'],[pm,pw],ax=axes[2,1])
axes[2,1].set_title('Biologisches Geschlecht'); axes[2,1].set_xlabel(''); axes[2,1].set_ylabel('');
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]
for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[2,2],y='Age',x='aki',data=violin,cut=0)
axes[2,2].set_title('Alter'); axes[2,2].set_xlabel(''); axes[2,2].set_ylabel('');

#######################################################################################
############################# XGBoost  ################################################
#######################################################################################

############## Top 10 laboratory values ###################
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result=su.xgboostFeatSelection_cv(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.xgboostHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/model_lab10_XBG.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_lab10_XGB.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_test_lab10_XGB.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/Xtest_aki_lab10_XGB.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_aki_lab10_XGB.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/ytest_aki_lab10_XGB.csv')


###############Violinplot################

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)         
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE)   
plt.rc('xtick', labelsize=BIGGER_SIZE)  
plt.rc('ytick', labelsize=25)  
plt.rc('legend', fontsize=BIGGER_SIZE)    
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(2,2,sharex=False,figsize=(20,11),dpi=400)
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Age',x='aki',data=violin,cut=0)
axes[0,0].set_title('Alter'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['GFR'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='GFR',x='aki',data=violin,cut=0)
axes[0,1].set_title('GFR'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel('');  
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='Ca',x='aki',data=violin,cut=0)
axes[1,0].set_title('Ca'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='Hst',x='aki',data=violin,cut=0)
axes[1,1].set_title('Hst'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');


############## Top 20 laboratory values ##################

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result=su.xgboostFeatSelection_cv(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.xgboostHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/model_lab20_XBG.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_lab20_XGB.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_test_lab20_XGB.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/Xtest_aki_lab20_XGB.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_aki_lab20_XGB.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/ytest_aki_lab20_XGB.csv')

###############Violinplot################

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)  
plt.rc('axes', titlesize=27)  
plt.rc('axes', labelsize=BIGGER_SIZE)    
plt.rc('xtick', labelsize=BIGGER_SIZE)  
plt.rc('ytick', labelsize=25)  
plt.rc('legend', fontsize=BIGGER_SIZE)  
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(2,3,sharex=False,figsize=(20,11),dpi=400)
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Age',x='aki',data=violin,cut=0)
axes[0,0].set_title('Alter'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['GFR'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='GFR',x='aki',data=violin,cut=0)
axes[0,1].set_title('GFR'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel(''); 
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,2],y='CRP',x='aki',data=violin,cut=0)
axes[0,2].set_title('CRP'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel(''); 
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='Ca',x='aki',data=violin,cut=0)
axes[1,0].set_title('Ca'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['Leuko'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='Leuko',x='aki',data=violin,cut=0)
axes[1,1].set_title('Leuko'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,2],y='Hst',x='aki',data=violin,cut=0)
axes[1,2].set_title('Hst'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('');

############################################
# dichotomized variables

lower_bound = pd.DataFrame(index=data.index,columns=data.columns)
upper_bound = pd.DataFrame(index=data.index,columns=data.columns)
row, col = -1, -1
for i in data.index:
    row = row+1
    print(row)
    col = -1
    for j in data.columns:
        col = col+1
        temp = lab[lab['Number case']==i].drop_duplicates(subset=['Lab_name','lower_ref','upper_ref'])
        if temp[temp['Lab_name']==j]['lower_ref'].size!=0:
            lower_bound.iloc[row,col] = temp[temp['Lab_name']==j]['lower_ref'].iloc[0]
        if temp[temp['Lab_name']==j]['upper_ref'].size!=0:
            upper_bound.iloc[row,col] = temp[temp['Lab_name']==j]['upper_ref'].iloc[0]
upper_bound['GFR'][upper_bound['GFR']=='no upper ref']=np.nan

#10
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result=su.xgboostFeatSelection_cv(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.xgboostHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/model_lab10_XBG_k.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_lab10_XGB_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_test_lab10_XGB_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/Xtest_aki_lab10_XGB_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_aki_lab10_XGB_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/ytest_aki_lab10_XGB_k.csv')


#20

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors


pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result=su.xgboostFeatSelection_cv(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.xgboostHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/model_lab20_XBG.pkl_k','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_lab20_XGB_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_test_lab20_XGB_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/Xtest_aki_lab20_XGB_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_aki_lab20_XGB_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/ytest_aki_lab20_XGB_k.csv')


###############Violinplot################

BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)       
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE)  
plt.rc('xtick', labelsize=BIGGER_SIZE)   
plt.rc('ytick', labelsize=25)  
plt.rc('legend', fontsize=BIGGER_SIZE)   
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(1,3,sharex=False,figsize=(20,11),dpi=400)
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0],y='Hst',x='aki',data=violin,cut=0)
axes[0].set_title('Hst'); axes[0].set_xlabel(''); axes[0].set_ylabel('');
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1],y='CRP',x='aki',data=violin,cut=0)
axes[1].set_title('CRP'); axes[1].set_xlabel(''); axes[1].set_ylabel('');  
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[2],y='Age',x='aki',data=violin,cut=0)
axes[2].set_title('Alter'); axes[2].set_xlabel(''); axes[2].set_ylabel('');


BIGGER_SIZE = 26
plt.rc('font', size=BIGGER_SIZE)        
plt.rc('axes', titlesize=27)    
plt.rc('axes', labelsize=BIGGER_SIZE) 
plt.rc('xtick', labelsize=BIGGER_SIZE)  
plt.rc('ytick', labelsize=25)   
plt.rc('legend', fontsize=BIGGER_SIZE)  
plt.rc('figure', titlesize=25)  

fig, axes = plt.subplots(2,3,sharex=False,figsize=(20,11),dpi=400) 
fig.tight_layout()

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_help = [int(i) for i in y_help]

for i,value in enumerate(y_help):
    if value==0:
        y_help[i]='Nicht-ANS'
    if value==1:
        y_help[i]='ANS'
  
violin = pd.DataFrame(data['Hst'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,0],y='Hst',x='aki',data=violin,cut=0)
axes[0,0].set_title('Hst'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
violin = pd.DataFrame(data['GFR'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,1],y='GFR',x='aki',data=violin,cut=0)
axes[0,1].set_title('GFR'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel(''); 
violin = pd.DataFrame(data['CRP'])
violin['aki'] = y_help
sb.violinplot(ax=axes[0,2],y='CRP',x='aki',data=violin,cut=0)
axes[0,2].set_title('CRP'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel(''); 
violin = pd.DataFrame(data['Ca'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,0],y='Ca',x='aki',data=violin,cut=0)
axes[1,0].set_title('Ca'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
violin = pd.DataFrame(data['Age'])
violin['aki'] = y_help
sb.violinplot(ax=axes[1,1],y='Age',x='aki',data=violin,cut=0)
axes[1,1].set_title('Alter'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
axes[1,2].axis('off')
      
    
########################################################################################################
## Random Forest and XGBoost with the variables from the variable selection of logistic regression. ##
########################################################################################################

# Random Forest with 10 features
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result = su.logReg(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.RandomForestHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/model_lab10_RF_lr.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_lab10_RF_lr.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_test_lab10_RF_lr.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/Xtest_aki_lab10_RF_lr.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_aki_lab10_RF_lr.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/ytest_aki_lab10_RF_lr.csv')


# Random Forest with 20 features
y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result = su.logReg(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.RandomForestHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/model_lab20_RF_lr.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_lab20_RF_lr.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_test_lab20_RF_lr.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/Xtest_aki_lab20_RF_lr.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_aki_lab20_RF_lr.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/ytest_aki_lab20_RF_lr.csv')

# Random Forest with 10 dichotomized features

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result = su.logReg(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.RandomForestHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/model_lab10_RF_lr_k.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_lab10_RF_lr_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_test_lab10_RF_lr_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/Xtest_aki_lab10_RF_lr_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_aki_lab10_RF_lr_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/ytest_aki_lab10_RF_lr_k.csv')

# Random Forest with 20 dichotomized features

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors

pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result = su.logReg(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.RandomForestHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/model_lab20_RF_lr_k.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_lab20_RF_lr_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_test_lab20_RF_lr_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/Xtest_aki_lab20_RF_lr_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/y_pred_aki_lab20_RF_lr_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_RF/ytest_aki_lab20_RF_lr_k.csv')

# XGBoost with 10 features

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result = su.logReg(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.xgboostHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/model_lab10_XBG_lr.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_lab10_XGB_lr.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_test_lab10_XGB_lr.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/Xtest_aki_lab10_XGB_lr.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_aki_lab10_XGB_lr.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/ytest_aki_lab10_XGB_lr.csv')

# XGBoost with 20 features

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','INR','Ca','Quick','Ery','Hb','BZ','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result = su.logReg(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.xgboostHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/model_lab20_XBG_lr.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_lab20_XGB_lr.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_test_lab20_XGB_lr.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/Xtest_aki_lab20_XGB_lr.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_aki_lab20_XGB_lr.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/ytest_aki_lab20_XGB_lr.csv')

# XGBoost with 10 dichotomized features

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors

pred_string=['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result = su.logReg(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.xgboostHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/model_lab10_XBG_lr_k.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_lab10_XGB_lr_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_test_lab10_XGB_lr_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/Xtest_aki_lab10_XGB_lr_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_aki_lab10_XGB_lr_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/ytest_aki_lab10_XGB_lr_k.csv')

# XGBoost with 20 dichotomized features

y_help = [int(i) in np.array(aki_cases) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']] 

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,x_train,lower_bound,upper_bound,ttest_aki_24)
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in x_train.index])])
x_train = predictors

pred_string = ['Sex','Age','Hst','GFR','CRP','Ca','Quick','Ery','Hb','PTT','RDW','HK','MCHC','GOT','MCV','Na','Thromb','K','Leuko','MCH']
predictors = su.categorical_predictors_oneside(pred_string,X_test,lower_bound,upper_bound,ttest_aki_24)
y_test = list(pd.Series(y_test)[np.array([i in predictors.index for i in X_test.index])])
X_test = predictors

result = su.logReg(x_train,y_train)

x_train=x_train[result]
X_test=X_test[result]

model, roc_auc, y_pred, y_test = su.xgboostHyper(x_train,y_train,X_test,y_test)

pickle.dump(model, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/model_lab20_XBG_lr_k.pkl','wb'))
pickle.dump(y_pred, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_lab20_XGB_lr_k.pkl','wb'))
pickle.dump(y_test, open('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_test_lab20_XGB_lr_k.pkl','wb'))

X_test.to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/Xtest_aki_lab20_XGB_lr_k.csv')
pd.DataFrame(y_pred).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/y_pred_aki_lab20_XGB_lr_k.csv')
pd.DataFrame(y_test).to_csv('C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Ergebnisse_XGB/ytest_aki_lab20_XGB_lr_k.csv')

