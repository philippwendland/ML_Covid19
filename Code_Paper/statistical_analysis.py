import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib qt5
import seaborn as sb

#Adding parent directory
import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent.parent))

import utils_repo as u
import statistical_utils as su

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import sklearn


data_link='C:/Users/schmitt4/Desktop/gtt202122/gtt202122.csv'

path = r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\DatensätzeR24'

#parsing data not included in repository
base, diag, dauerdiag, med, lab, bew, vor, vit, ops, drg, goa, ala, pkz, tim, tri, att, medgab = u.parser_gtt(data_link)

# diag.insert(4,'ICD_upper_group',diag['ICD']) 
# diag['ICD_upper_group'][['.' in i for i in diag['ICD_upper_group']]]=[i[:i.index('.')] for i in diag['ICD_upper_group'] if '.' in i]
# diag['ICD_upper_group'].unique()

#Using patients with admission before end 2021
temp = bew[['Aufnahme' in str(i) for i in bew['Bewegungstyp']]]
mask = temp['Date'] < np.datetime64('2021-12-31')
cases = temp[mask]['Number case'].unique()
base2 = base[[i in cases for i in base['Number case']]]
lab2 = lab[[i in cases for i in lab['Number case']]]
bew2 = bew[[i in cases for i in bew['Number case']]]
diag2 = diag[[i in cases for i in diag['Number case']]]
ops2 = ops[[i in cases for i in ops['Number case']]]



#All Covid patients with submission before 2022
covid_cases = diag2.values[[diag2['ICD'].values[i]=='U07.1' for i in range(diag2['ICD'].shape[0])],0]
covid_cases = np.unique(covid_cases)
non_covid_cases = diag2['Number case'].unique()[np.in1d(diag2['Number case'].unique(),covid_cases,invert=True)]


#changing save paths? 
#changing data til end 2021
#ICD upper group removing?

#%%
############################################################################################
############# Wilcoxon Test for lab values Trigger vs. NonTrigger ##########################
############################################################################################

### Covid alive vs. dead
bew_covid = bew2[[i in covid_cases for i in bew2['Number case']]]
covid_dead = bew_covid[bew_covid['Bewegungsart']=='verstorben']
covid_dead = covid_dead['Number case'].unique()
covid_alive = covid_cases[np.in1d(covid_cases, covid_dead, invert=True)]

lab_48h, corr_help_48h = su.lab_48h(lab2,covid_cases,bew2)
wilcoxon_covid_dead = su.wilcoxon_without_BZ(lab_48h, corr_help_48h.columns, covid_dead, covid_alive)
wilcoxon_covid_dead = su.wilcoxon_BZ(lab_48h, wilcoxon_covid_dead, covid_dead, covid_alive)

wilcoxon_covid_dead = su.wilcoxon_multiple_test(wilcoxon_covid_dead)

wilcoxon_covid_dead.to_csv('C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/wilcoxon_covid_dead_48h.csv',sep=';')
#wilcoxon_covid_dead.to_csv('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/wilcoxon_covid_dead_48h.csv',sep=';')

### Covid ICU yes/no
icu_covid = bew_covid[bew_covid['Fachliche_OE']=='3600']
icu_covid = icu_covid['Number case'].unique()
nonicu_covid = covid_cases[np.in1d(covid_cases, icu_covid, invert=True)]

wilcoxon_covid = su.wilcoxon_without_BZ(lab_48h, corr_help_48h.columns, icu_covid, nonicu_covid)
wilcoxon_covid = su.wilcoxon_BZ(lab_48h, wilcoxon_covid, icu_covid, nonicu_covid)

wilcoxon_covid = su.wilcoxon_multiple_test(wilcoxon_covid)

wilcoxon_covid.to_csv('C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/wilcoxon_covid_icu_48h.csv',sep=';')

### Covid vent yes/no
ops_covid = ops2[[i in covid_cases for i in ops2['Number case']]]
vent_covid = ops_covid[['8-71' in str(i) for i in ops_covid['Schlüssel']]]
vent_covid = vent_covid['Number case'].unique()
nonvent_covid = covid_cases[np.in1d(covid_cases, vent_covid, invert=True)]

wilcoxon_covid = su.wilcoxon_without_BZ(lab_48h, corr_help_48h.columns, vent_covid, nonvent_covid)
wilcoxon_covid = su.wilcoxon_BZ(lab_48h, wilcoxon_covid, vent_covid, nonvent_covid)

wilcoxon_covid = su.wilcoxon_multiple_test(wilcoxon_covid)

wilcoxon_covid.to_csv('C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/wilcoxon_covid_vent_48h.csv',sep=';')
#wilcoxon_covid.to_csv('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/wilcoxon_covid_vent_48h.csv',sep=';')


########################################################
##### t-Test for lab values Trigger vs. NonTrigger #####

### Covid alive vs. dead
ttest_covid_dead = su.ttest_without_BZ(lab_48h, corr_help_48h.columns, covid_dead, covid_alive)
ttest_covid_dead = su.ttest_BZ(lab_48h, ttest_covid_dead, covid_dead, covid_alive)

ttest_covid_dead = su.ttest_multiple_test(ttest_covid_dead)

ttest_covid_dead.to_csv('C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/ttest_covid_dead_48h.csv',sep=';')


### Covid icu yes/no

ttest_covid = su.ttest_without_BZ(lab_48h, corr_help_48h.columns, icu_covid, nonicu_covid)
ttest_covid = su.ttest_BZ(lab_48h, ttest_covid, icu_covid, nonicu_covid)

ttest_covid = su.ttest_multiple_test(ttest_covid)

ttest_covid.to_csv('C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/ttest_covid_icu_48h.csv',sep=';')


### Covid vent yes/no
ttest_covid = su.ttest_without_BZ(lab_48h, corr_help_48h.columns, vent_covid, nonvent_covid)
ttest_covid = su.ttest_BZ(lab_48h, ttest_covid, vent_covid, nonvent_covid)

ttest_covid = su.ttest_multiple_test(ttest_covid)

ttest_covid.to_csv('C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/ttest_covid_vent_48h.csv',sep=';')


#%%
###############################################################################
################### Vorhersagemodelle #########################################
###############################################################################

## run only if labvalues of the first 48h are used
lab_48h, corr_help_48h = su.lab_48h(lab2,covid_cases,bew2)
covid_dummy_lab_48h = (corr_help_48h.iloc[np.in1d(corr_help_48h.index,covid_cases),:])
covid_dummy_lab_48h.index = covid_dummy_lab_48h.index.astype(int)
data = covid_dummy_lab_48h

# select Sex and Age
base_covid = base2[[i in data.index.astype(int) for i in base2['Number case']]]
base_covid = base_covid.drop_duplicates(subset=['Number case'])
sex = [int(i) for i in base_covid['Sex']=='M']
sex = pd.Series(sex,name='Sex',index=base_covid['Number case'])
age = base_covid['Age'].astype(float)
age.index = base_covid['Number case']
data = pd.concat([data,sex],axis=1)
data = pd.concat([data,age],axis=1)

### 24h
lab_24h, corr_help_24h = su.lab_24h(lab2,covid_cases,bew2)
covid_dummy_lab_24h = (corr_help_24h.iloc[np.in1d(corr_help_24h.index,covid_cases),:])
covid_dummy_lab_24h.index = covid_dummy_lab_24h.index.astype(int)
data24 = covid_dummy_lab_24h
temp = pd.DataFrame(index=data.index[np.in1d(data.index,data24.index,invert=True)])
data.index[np.in1d(data.index,data24.index,invert=True)]
data24 = pd.concat([data24,temp],axis=0)
data24 = data24.sort_index()
base_covid = base2[[i in data24.index.astype(int) for i in base2['Number case']]]
base_covid = base_covid.drop_duplicates(subset=['Number case'])
sex = [int(i) for i in base_covid['Sex']=='M']
sex = pd.Series(sex,name='Sex',index=base_covid['Number case'])
age = base_covid['Age'].astype(float)
age.index = base_covid['Number case']
data24 = pd.concat([data24,sex],axis=1)
data24 = pd.concat([data24,age],axis=1)

## Output: Dead vs. Alive
bew_covid = bew2[[i in data.index.astype(int) for i in bew2['Number case']]]
deaths_covid = bew_covid[bew_covid['Bewegungsart']=='verstorben']
deaths_covid['Number case']
y_help = [int(i) in np.array(deaths_covid['Number case']) for i in data.index]

## Output: icu ja / nein
bew_covid = bew2[[i in data.index.astype(int) for i in bew2['Number case']]]
icu_covid = bew_covid[bew_covid['Fachliche_OE']=='3600']
icu_covid['Number case'].unique()
y_help = [int(i) in np.array(icu_covid['Number case']) for i in data.index]

## Output: Mech. vent ja / nein
ops_covid = ops2[[i in data.index.astype(int) for i in ops2['Number case']]]
vent_covid = ops_covid[['8-71' in str(i) for i in ops_covid['Schlüssel']]]
vent_covid['Number case']
y_help = [int(i) in np.array(vent_covid['Number case']) for i in data.index]


#######################################################################################
######################## Logistic Regression ##########################################
#######################################################################################

### Covid dead / alive
y_help = [int(i) in np.array(deaths_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','CRP','Hst','GFR','BZ','Krea','RDW','Ca','GOT','PTT','INR']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)

#Creating a sklearn version of logreg for the shapley plots
x_train=x_train[result]
X_test=X_test[result]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
x_train = x_train.dropna()
logreg_sklearn=sklearn.linear_model.LogisticRegression(random_state=4,penalty='none')
logreg_sklearn.fit(x_train,y_train)
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()

y_pred.to_csv(path+'/y_pred_tod48.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_tod48.csv')


#creating predictions of the 24h version
x_train = data24[result]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

log_reg, y_pred, y_test = su.logRegTest24(X_test, y_test, log_reg, result)

y_pred.to_csv(path+'/y_pred_tod24.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_tod24.csv')

x_train = data24[['Sex','Age','CRP','Hst','GFR','BZ','Krea','RDW','Ca','GOT','PTT','INR']]
x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)
log_reg, y_pred, y_test = su.logRegTest24(X_test, y_test, log_reg, result)


### icustation ja / nein
### untere Zeile Spaltung nach covid death ja vs. nein
y_help = [int(i) in np.array(icu_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Ca','CRP','BZ','PTT','GOT']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)

### vent ja / nein
### untere Zeile Spaltung nach covid death ja vs. nein
y_help = [int(i) in np.array(vent_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Ca','CRP','PTT','GOT','BZ']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)

############################################
# Categorical

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
 
###

### Covid dead / alive
y_help = [int(i) in np.array(deaths_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]
ttest = pd.read_csv(r'C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/WilcoxonTtest/ttest_covid_dead_48h.csv',sep=';',header=0)

pred_string = ['Sex','Age','CRP','Hst','GFR','Krea','RDW','Ca','GOT','PTT','Quick']

predictors = su.categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)

x_train= x_train[result]
X_test=X_test[result]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()
x_train = x_train.dropna()
logreg_sklearn=sklearn.linear_model.LogisticRegression(random_state=2,penalty='none')
logreg_sklearn.fit(x_train.dropna()[['Hst','PTT','Age','GOT']],y_train)


pd.DataFrame(y_pred).to_csv(path+'/y_pred_dead_cat48.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_dead_cat48.csv')

#Predictions of the 24 Version
predictors = su.categorical_predictors_oneside(pred_string,result,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

log_reg, y_pred, y_test = su.logRegTest24(X_test, y_test, log_reg, result)

pd.DataFrame(y_pred).to_csv(path+'/y_pred_dead_cat24.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_dead_cat24.csv')



### icustation ja / nein
### untere Zeile Spaltung nach covid death ja vs. nein
y_help = [int(i) in np.array(icu_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]
ttest = pd.read_csv(r'C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/WilcoxonTtest/ttest_covid_icu_48h.csv',sep=';')

pred_string = ['Sex','Age','Ca','CRP','PTT','GOT']

predictors = su.categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)

x_train= x_train[result]
X_test=X_test[result]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()
x_train = x_train.dropna()
logreg_sklearn=sklearn.linear_model.LogisticRegression(random_state=4,penalty='none')
logreg_sklearn.fit(x_train.dropna()[['PTT','Ca','GOT','Sex']],y_train)

pd.DataFrame(y_pred).to_csv(path+'/y_pred_icu_cat48.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_icu_cat48.csv')

predictors = su.categorical_predictors_oneside(result,data24,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)
log_reg, y_pred, y_test = su.logRegTest24(X_test, y_test, log_reg, result)

pd.DataFrame(y_pred).to_csv(path+'/y_pred_icu_cat24.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_icu_cat24.csv')

### vent ja / nein
### untere Zeile Spaltung nach covid death ja vs. nein
y_help = [int(i) in np.array(vent_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]
ttest = pd.read_csv(r'C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/WilcoxonTtest_holm/ttest_covid_vent_48h.csv',sep=';')

pred_string = ['Sex','Age','Ca','CRP','PTT','GOT']

predictors = su.categorical_predictors_oneside(pred_string,data24,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result, log_reg, fpr, tpr = su.logReg_cv2(x_train,y_train)

log_reg, y_pred, y_test = su.logRegTest(x_train, X_test, y_train, y_test, result)


#######################################################################################
############################# Random Forest  ##########################################
#######################################################################################

### Covid dead / alive
y_help = [int(i) in np.array(deaths_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','CRP','Hst','GFR','BZ','Krea','RDW','Ca','GOT','PTT','Quick','INR']]
# after LogReg Selection
x_train = data[['CRP','Age','BZ','Quick']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.RandomForest(x_train,y_train)

su.RandomForestFeatSelection_cv(x_train,y_train)

#Feature Importance function only returns the results as a print
result = x_train.columns
#result = ['a','b']
rf, y_pred, y_test, X_test = su.RandomForestTest(x_train, X_test, y_train, y_test, result)
rf, y_pred, y_test, X_test = su.RandomForestTest24(X_test, y_test, rf, result)

model, roc_auc = su.RandomForestHyper(x_train,y_train)
su.RandomForestHyperTest(x_train, X_test, y_train, y_test, model)

# CATEGORICAL 
ttest = pd.read_csv(r'C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/WilcoxonTtest/ttest_covid_dead_48h.csv',sep=';')
y_train = [int(i) for i in y_help]

pred_string = ['Sex','Age','CRP','Hst','GFR','Krea','RDW','Ca','GOT','PTT','Quick']
# after LogReg Selection
pred_string = ['Hst','PTT','Age','GOT']

predictors = su.categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])


x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.RandomForest(x_train,y_train)
su.RandomForestFeatSelection_cv(x_train,y_train)

result = x_train.columns
#result = ['a','b']
rf, y_pred, y_test, X_test = su.RandomForestTest(x_train, X_test, y_train, y_test, result)

pd.DataFrame(y_pred).to_csv(path+'/y_pred_dead_rf_cat48.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_dead_rf_cat48.csv')

pred_string = ['Hst','PTT','Age','GOT']

predictors = su.categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

rf, y_pred, y_test, X_test = su.RandomForestTest24(X_test, y_test, rf, result)

pd.DataFrame(y_pred).to_csv(path+'/y_pred_dead_rf_cat24.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_dead_rf_cat24.csv')

### icustation ja / nein
y_help = [int(i) in np.array(icu_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]

#   plus sex,age
x_train = data[['Sex','Age','Ca','CRP','BZ','PTT','GOT']]
# nach LogReg Selection
x_train = data[['Ca','BZ','CRP']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.RandomForest(x_train,y_train)
su.RandomForestFeatSelection_cv(x_train,y_train)

result = x_train.columns
rf = su.RandomForestTest(x_train, X_test, y_train, y_test, result)

model, roc_auc = su.RandomForestHyper(x_train,y_train)
su.RandomForestHyperTest(x_train, X_test, y_train, y_test, model)


# CATEGORICAL
ttest = pd.read_csv(r'C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/WilcoxonTtest_holm/ttest_covid_icu_48h.csv',sep=';')
y_train = [int(i) for i in y_help]

pred_string = ['Sex','Age','Ca','CRP','PTT','GOT']
# nach LogReg Selection
pred_string = ['Ca','PTT','GOT','Sex']

predictors = su.categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.RandomForest(x_train,y_train)
su.RandomForestFeatSelection_cv(x_train,y_train)

### vent ja / nein
y_help = [int(i) in np.array(vent_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]

#   plus sex,age
x_train = data[['Sex','Age','Ca','CRP','PTT','GOT','BZ']]
# nach LogReg Selection
x_train = data[['CRP','Ca']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.RandomForest(x_train,y_train)
su.RandomForestFeatSelection_cv(x_train,y_train)

result=['Ca','CRP','BZ']
x_train= x_train[result]
X_test=X_test[result]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()
x_train = x_train.dropna()

rf, y_pred, y_test, X_test = su.RandomForestTest(x_train, X_test, y_train, y_test, result)



model, roc_auc = su.RandomForestHyper(x_train,y_train)
su.RandomForestHyperTest(x_train, X_test, y_train, y_test, model)

pd.DataFrame(y_pred).to_csv(path+'/y_pred_beat48.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_beat48.csv')

x_train = data24[['Sex','Age','Ca','CRP','PTT','GOT','BZ']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

result=['Ca','CRP','BZ']
x_train= x_train[result]
X_test=X_test[result]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()
x_train = x_train.dropna()

rf, y_pred, y_test, X_test = su.RandomForestTest24(X_test, y_test, rf, result)
pd.DataFrame(y_pred).to_csv(path+'/y_pred_beat24.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_beat24.csv')

# CATEGORICAL
ttest = pd.read_csv(r'C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/WilcoxonTtest_holm/ttest_covid_vent_48h.csv',sep=';')
y_train = [int(i) for i in y_help]
pred_string = ['Sex','Age','Ca','CRP','PTT','GOT']
# nach LogReg Selection
pred_string = ['Ca','PTT']

predictors = su.categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.RandomForest(x_train,y_train)
su.RandomForestFeatSelection_cv(x_train,y_train)


x_train = data[x_train.columns[rf.feature_importances_>0.08]]
###
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

X_train = x_train[result]
X_test = X_test[result]
y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])

#smote = SMOTE(random_state=4)
#X_res, y_res = smote.fit_resample(X_train.dropna(), y_train)

rf = RandomForestClassifier(random_state=0)
rf = rf.fit(X_train.dropna(), y_train)
#rf = rf.fit(X_res,y_res)
y_pred = rf.predict_proba(X_test.dropna())[:,1]
fpr, tpr, thresh = roc_curve(y_test,y_pred)
roc_auc = auc(fpr,tpr)
print(roc_auc)

#######################################################################################
############################# XGBoost  ##########################################
#######################################################################################

from xgboost import XGBClassifier

### Covid dead / alive
y_help = [int(i) in np.array(deaths_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]
x_train = data[['Sex','Age','CRP','Hst','GFR','BZ','Krea','RDW','Ca','GOT','PTT','Quick','INR']]
# nach LogReg Selection
x_train = data[['CRP','Age','BZ','Quick']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.xgboost(x_train,y_train)
su.xgboostFeatSelection_cv(x_train,y_train)

result = x_train.columns
su.xgboostTest(x_train, X_test, y_train, y_test, result)

model, roc_auc = su.xgboostHyper(x_train,y_train)
su.xgboostHyperTest(x_train, X_test, y_train, y_test, model)

# CATEGRICAL
ttest = pd.read_csv(r'C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/WilcoxonTtest_holm/ttest_covid_dead_48h.csv',sep=';')
y_train = [int(i) for i in y_help]

pred_string = ['Sex','Age','CRP','Hst','GFR','Krea','RDW','Ca','GOT','PTT','Quick']
# nach LogReg Selection
pred_string = ['Hst','PTT','Age','GOT']

predictors = su.categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.xgboost(x_train,y_train)
su.xgboostFeatSelection_cv(x_train,y_train)

### icustation ja / nein
y_help = [int(i) in np.array(icu_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]
x_train = data[['Sex','Age','Ca','CRP','BZ','PTT','GOT']]
# nach LogReg Selection
x_train = data[['Ca','BZ','CRP']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.xgboost(x_train,y_train)
su.xgboostFeatSelection_cv(x_train,y_train)

result = x_train.columns
xgb, y_pred, y_test = su.xgboostTest(x_train, X_test, y_train, y_test, result)

model, roc_auc = su.xgboostHyper(x_train,y_train)
xgb, y_pred, y_test = su.xgboostHyperTest(x_train, X_test, y_train, y_test, model)


x_train= x_train[['Age', 'Ca', 'CRP', 'BZ', 'PTT']]
X_test=X_test[['Age', 'Ca', 'CRP', 'BZ', 'PTT']]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()
x_train = x_train.dropna()

x_train = data24[['Age', 'Ca', 'CRP', 'BZ', 'PTT']]

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

x_train= x_train[['Age', 'Ca', 'CRP', 'BZ', 'PTT']]
X_test=X_test[['Age', 'Ca', 'CRP', 'BZ', 'PTT']]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()
x_train = x_train.dropna()

pd.DataFrame(y_pred).to_csv(path+'/y_pred_icu48.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_icu48.csv')

x_train = data[['Age', 'Ca', 'CRP', 'BZ', 'PTT']]
x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

x_train= x_train[['Age', 'Ca', 'CRP', 'BZ', 'PTT']]
X_test=X_test[['Age', 'Ca', 'CRP', 'BZ', 'PTT']]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()
x_train = x_train.dropna()


xgb, y_pred, y_test = su.xgboostTest24(X_test, y_test, xgb, result)

pd.DataFrame(y_pred).to_csv(path+'/y_pred_icu24.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_icu24.csv')

# CATEGORICAL
ttest = pd.read_csv(r'C:/Users/schmitt4/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/WilcoxonTtest_holm/ttest_covid_icu_48h.csv',sep=';')
y_train = [int(i) for i in y_help]
pred_string = ['Sex','Age','Ca','CRP','PTT','GOT']
# nach LogReg Selection
pred_string = ['Ca','PTT','Sex','GOT']

predictors = su.categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.xgboost(x_train,y_train)
su.xgboostFeatSelection_cv(x_train,y_train)

### vent ja / nein
y_help = [int(i) in np.array(vent_covid['Number case']) for i in data.index]
y_train = [int(i) for i in y_help]

x_train = data[['Sex','Age','Ca','CRP','PTT','GOT','BZ']]
# nach LogReg Selection
x_train = data[['CRP','Ca']]
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=2)

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.xgboost(x_train,y_train)
su.xgboostFeatSelection_cv(x_train,y_train)

result = x_train.columns
xgb, y_pred, y_test = su.xgboostTest(x_train, X_test, y_train, y_test, result)
xgb, y_pred, y_test = su.xgboostTest24(X_test, y_test, xgb, result)

model, roc_auc = su.xgboostHyper(x_train,y_train)
su.xgboostHyperTest(x_train, X_test, y_train, y_test, model)

# CATEGORICAL 
ttest = pd.read_csv(r'C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/WilcoxonTtest/ttest_covid_vent_48h.csv',sep=';')
y_train = [int(i) for i in y_help]
pred_string = ['Sex','Age','Ca','CRP','PTT','GOT']
# nach LogReg Selection
pred_string = ['Ca','PTT']

predictors = su.categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

su.xgboost(x_train,y_train)
su.xgboostFeatSelection_cv(x_train,y_train)

x_train= x_train[['Ca','CRP','PTT','GOT']]
X_test=X_test[['Ca','CRP','PTT','GOT']]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()
x_train = x_train.dropna()

pd.DataFrame(y_pred).to_csv(path+'/y_pred_beat_cat48.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_beat_cat48.csv')

#test 
pred_string = ['Sex','Age','Ca','CRP','PTT','GOT']

predictors = su.categorical_predictors_oneside(pred_string,data24,lower_bound,upper_bound,ttest)
x_train = predictors
y_train = list(pd.Series(y_train)[np.array([i in predictors.index for i in data.index])])

x_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)


x_train= x_train[['Ca','CRP','PTT','GOT']]
X_test=X_test[['Ca','CRP','PTT','GOT']]
y_train = list(pd.Series(y_train)[np.array(x_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
X_test = X_test.dropna()
x_train = x_train.dropna()

pd.DataFrame(y_pred).to_csv(path+'/y_pred_beat_cat24.csv')
pd.DataFrame(y_test).to_csv(path+'/ytest_beat_cat24.csv')

###
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

X_train = x_train[result]
X_test = X_test[result]
y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])

#smote = SMOTE(random_state=4)
#X_res, y_res = smote.fit_resample(X_train.dropna(), y_train) 

xgb = XGBClassifier(random_state=0)
xgb.fit(X_train.dropna(), y_train)
#xgb.fit(X_res, y_res)
y_pred = xgb.predict_proba(X_test.dropna())[:,1]
fpr, tpr, thresh = roc_curve(y_test,y_pred)
roc_auc = auc(fpr,tpr)
print(roc_auc)

### 
from sklearn.feature_selection import SelectFromModel

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=4)

y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])

sel = SelectFromModel(XGBClassifier(random_state=0))
sel.fit(X_train.dropna(), y_train)
selected_feat= X_train.columns[(sel.get_support())]
print(selected_feat)

xgb = XGBClassifier(random_state=0)
xgb = xgb.fit(X_train.dropna()[selected_feat], y_train)
#rf = rf.fit(X_res,y_res)
y_pred = xgb.predict_proba(X_test.dropna()[selected_feat])[:,1]
fpr, tpr, thresh = roc_curve(y_test,y_pred)
roc_auc = auc(fpr,tpr)
print(roc_auc)

######################### Violin Plots ###############
import statistical_utils as su

## mit Funktionen
su.violinplot_dead_logreg_alternativ(deaths_covid,data)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/Violinenplots_20_11/dead_logreg_alternativ.jpg')
su.violinplot_dead_logreg_alternativ_cat(deaths_covid,data)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/Violinenplots_20_11/dead_logreg_cat_alternativ.jpg')
su.violinplot_vent_xgb_cat(vent_covid,data)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/Violinenplots_20_11/beat_xgb_cat.jpg')
su.violinplot_icu_logreg_cat(icu_covid,data)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/Violinenplots_20_11/icu_logreg_cat.jpg')
su.violinplot_icu_xgb(icu_covid,data)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/Violinenplots_20_11/icu_xgb_bsp.jpg')
su.violinplot_vent_rf(vent_covid,data)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/Violinenplots_20_11/beat_rf.jpg')

################# Histogramm für Feature selection #######################

wilcoxon = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\wilcoxon_covid_vent_48h.csv',sep=',')
ttest = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\ttest_covid_vent_48h.csv',sep=',')

wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30]=np.log(wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30])
for i in range(20):
    if ttest[ttest["Labvalue"]==wilcoxon["Labvalue"].iloc[i]]["Statistic"].values[0] > 0:
        wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]=-wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]

fig = plt.figure(dpi=400, figsize=(21,10))
x = ['Calcium','CRP','Glucose','LDH','Lymph','Neutro','PTT','GOT','Eos','CK-NAC','iCa','FCOHb','Mg','Free T3','T_Bilirubin','Lactate','Basophils','Monoc','Potassium','Urea','MCV','Amyl','Seg','U-Leuc','Lip','pCO2','Alb','ALP','GPT','MCHC']
x=x[:20]
ax = wilcoxon.iloc[0:20].plot.bar(x='Labvalue', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
#plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=25
#plt.xlabel('Laboratory value')
plt.ylabel('Association')
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
ax.xaxis.set_label_coords(x=0.5,y= -0.2)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/wilcoxon_vent_holm_log2.jpg')

wilcoxon = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\wilcoxon_covid_icu_48h.csv',sep=',')
ttest = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\ttest_covid_icu_48h.csv',sep=',')

wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30]=np.log(wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30])
for i in range(30):
    if ttest[ttest["Labvalue"]==wilcoxon["Labvalue"].iloc[i]]["Statistic"].values[0] > 0:
        wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]=-wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]


fig = plt.figure(dpi=400, figsize=(21,10))
x = ['Glucose','Calcium','CRP','LDH','Lymph','Neutro','PTT','GOT','Eos','iCalcium','FCOHb','Monoc','CK-NAC','Seg','Mg','Albumin','pCO2','pH','Baso','GPT','T_Bilirubin','Urea','Ferritin','Free T3','U-SpeWe','D-Bil','Myelo','PCT','Lact','CK-MB']
x=x[:25]
ax = wilcoxon.iloc[0:25].plot.bar(x='Labvalue', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=20
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
plt.ylabel('Association')
plt.xlabel('Laboratory Value')
ax.xaxis.set_label_coords(x=0.5,y= -0.125)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/wilcoxon_icu_holm_log.jpg')

wilcoxon = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\wilcoxon_covid_dead_48h.csv',sep=',')
ttest = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\ttest_covid_dead_48h.csv',sep=',')


wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30]=np.log(wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30])
for i in range(23):
    if ttest[ttest["Labvalue"]==wilcoxon["Labvalue"].iloc[i]]["Statistic"].values[0] > 0:
        wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]=-wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]

fig = plt.figure(dpi=400, figsize=(21,10))
x = ['Urea','Creatinine','GFR','CRP','Glucose','Quick','INR','PTT','O2-Sat','TropThs','RDW','LDH','GOT','PCT','Calcium','Lymph','NTpBNp','MCV','Leuko','MCHC','FO2Hb','Platelets','Monoc','pO2','D-Dimer','Neutro','iCalcium','Urine-pH','pH','Sodium']
x=x[:23]
ax = wilcoxon.iloc[0:23].plot.bar(x='Labvalue', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=20
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
plt.ylabel('Association')
plt.xlabel('Laboratory value')
ax.xaxis.set_label_coords(x=0.5,y= -0.135)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/wilcoxon_dead_holm_log.jpg')


wilcoxon = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\wilcoxon_covid_vent_48h.csv',sep=',')
ttest = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\ttest_covid_vent_48h.csv',sep=',')

# wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30]=np.log(wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30])
# for i in range(20):
#     if ttest[ttest["Labvalue"]==wilcoxon["Labvalue"].iloc[i]]["Statistic"].values[0] > 0:
#         wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]=-wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]

fig = plt.figure(dpi=400, figsize=(21,10))
x = ['Calcium','CRP','Glucose','LDH','Lymph','Neutro','PTT','GOT','Eos','CK-NAC','iCa','FCOHb','Mg','Free T3','T_Bilirubin','Lactate','Basophils','Monoc','Potassium','Urea','MCV','Amyl','Seg','U-Leuc','Lip','pCO2','Alb','ALP','GPT','MCHC']
x=x[:20]
ax = wilcoxon.iloc[0:20].plot.bar(x='Labvalue', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
#plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
#plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=20
plt.xlabel('Laboratory value')
plt.ylabel("Bonferroni-Holm adj. P-values")
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
ax.xaxis.set_label_coords(x=0.5,y= -0.125)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/wilcoxon_vent_holm_notlog.jpg')

wilcoxon = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\wilcoxon_covid_icu_48h.csv',sep=',')
ttest = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\ttest_covid_icu_48h.csv',sep=',')

# wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30]=np.log(wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30])
# for i in range(30):
#     if ttest[ttest["Labvalue"]==wilcoxon["Labvalue"].iloc[i]]["Statistic"].values[0] > 0:
#         wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]=-wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]


fig = plt.figure(dpi=400, figsize=(21,10))
x = ['Glucose','Calcium','CRP','LDH','Lymph','Neutro','PTT','GOT','Eos','iCalcium','FCOHb','Monoc','CK-NAC','Seg','Mg','Albumin','pCO2','pH','Baso','GPT','T_Bilirubin','Urea','Ferritin','Free T3','U-SpeWe','D-Bil','Myelo','PCT','Lact','CK-MB']
x=x[:25]
ax = wilcoxon.iloc[0:25].plot.bar(x='Labvalue', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
#plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
#plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=20
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
plt.ylabel("Bonferroni-Holm adj. P-values")
plt.xlabel('Laboratory Value')
ax.xaxis.set_label_coords(x=0.5,y= -0.125)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/wilcoxon_icu_holm_notlog.jpg')

wilcoxon = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\wilcoxon_covid_dead_48h.csv',sep=',')
ttest = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\WilcoxonTtest_holm\Paper\ttest_covid_dead_48h.csv',sep=',')


# wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30]=np.log(wilcoxon["Bonferroni-Holm adj. P-values"].iloc[0:30])
# for i in range(23):
#     if ttest[ttest["Labvalue"]==wilcoxon["Labvalue"].iloc[i]]["Statistic"].values[0] > 0:
#         wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]=-wilcoxon["Bonferroni-Holm adj. P-values"].iloc[i]

fig = plt.figure(dpi=400, figsize=(21,10))
x = ['Urea','Creatinine','GFR','CRP','Glucose','Quick','INR','PTT','O2-Sat','TropThs','RDW','LDH','GOT','PCT','Calcium','Lymph','NTpBNp','MCV','Leuko','MCHC','FO2Hb','Platelets','Monoc','pO2','D-Dimer','Neutro','iCalcium','Urine-pH','pH','Sodium']
x=x[:25]
ax = wilcoxon.iloc[0:25].plot.bar(x='Labvalue', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
#plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
#plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=20
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
plt.ylabel("Bonferroni-Holm adj. P-values")
plt.xlabel('Laboratory value')
ax.xaxis.set_label_coords(x=0.5,y= -0.135)
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/wilcoxon_dead_holm_notlog.jpg')

###### Diagnosis tests


fisher = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\Fisher_holm_paper\fisher_covid_dead_alldiag_overOR.csv',sep=',')
fisher['Bonferroni-Holm adj. P-values']=-np.log(fisher['Bonferroni-Holm adj. P-values'].astype(float))

fig = plt.figure(dpi=400, figsize=(21,10))
#x = ['Z51.5','J12.8','I10.00','I50.14','N18.4','J96.00','F05.1','N17.93','I48.9','J80.03','E87.0','E87.5','N39.0','D64.9','I25.13','N17.91','J80.02','R40.0','A49.9','K72.0','Z95.1']
ax = fisher.iloc[:20].plot.bar(x='Diagnose', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
#plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
#plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
#ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=25
plt.xlabel('Diagnosis')
plt.ylabel('Association')
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
#ax.xaxis.set_label_coords(x=0.5,y= -0.2)
fig.tight_layout()

plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/fisher_dead_log.jpg')



fisher = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\Fisher_holm_paper\fisher_covid_icu_alldiag_overOR.csv',sep=',')
fisher['Bonferroni-Holm adj. P-values']=-np.log(fisher['Bonferroni-Holm adj. P-values'].astype(float))

fig = plt.figure(dpi=400, figsize=(21,10))
#x = ['Z51.5','J12.8','I10.00','I50.14','N18.4','J96.00','F05.1','N17.93','I48.9','J80.03','E87.0','E87.5','N39.0','D64.9','I25.13','N17.91','J80.02','R40.0','A49.9','K72.0','Z95.1']
ax = fisher.iloc[:15].plot.bar(x='Diagnose', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
#plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
#plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
#ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=25
plt.xlabel('Diagnosis')
plt.ylabel('Association')
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
fig.tight_layout()
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/fisher_icu_log.jpg')


fisher = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\Fisher_holm_paper\fisher_covid_vent_alldiag_overOR.csv',sep=',')
fisher['Bonferroni-Holm adj. P-values']=-np.log(fisher['Bonferroni-Holm adj. P-values'].astype(float))

fig = plt.figure(dpi=400, figsize=(21,10))
#x = ['Z51.5','J12.8','I10.00','I50.14','N18.4','J96.00','F05.1','N17.93','I48.9','J80.03','E87.0','E87.5','N39.0','D64.9','I25.13','N17.91','J80.02','R40.0','A49.9','K72.0','Z95.1']
ax = fisher.iloc[:6].plot.bar(x='Diagnose', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
#plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
#plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
#ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=25
plt.xlabel('Diagnosis')
plt.ylabel('Association')
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
fig.tight_layout()
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/fisher_vent_log.jpg')



fisher = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\Fisher_holm_paper\fisher_covid_dead_overOR.csv',sep=',')
fisher['Bonferroni-Holm adj. P-values']=-np.log(fisher['Bonferroni-Holm adj. P-values'].astype(float))

fig = plt.figure(dpi=400, figsize=(21,10))
#x = ['Z51.5','J12.8','I10.00','I50.14','N18.4','J96.00','F05.1','N17.93','I48.9','J80.03','E87.0','E87.5','N39.0','D64.9','I25.13','N17.91','J80.02','R40.0','A49.9','K72.0','Z95.1']
ax = fisher.iloc[:20].plot.bar(x='Diagnose', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
#plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
#plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
#ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=25
plt.xlabel('Diagnosis')
plt.ylabel('Association')
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
#ax.xaxis.set_label_coords(x=0.5,y= -0.2)
fig.tight_layout()
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/fisher_dead_group_log.jpg')


fisher = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\Fisher_holm_paper\fisher_covid_icu_overOR.csv',sep=',')
fisher['Bonferroni-Holm adj. P-values']=-np.log(fisher['Bonferroni-Holm adj. P-values'].astype(float))

fig = plt.figure(dpi=400, figsize=(21,10))
#x = ['Z51.5','J12.8','I10.00','I50.14','N18.4','J96.00','F05.1','N17.93','I48.9','J80.03','E87.0','E87.5','N39.0','D64.9','I25.13','N17.91','J80.02','R40.0','A49.9','K72.0','Z95.1']
ax = fisher.iloc[:10].plot.bar(x='Diagnose', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
#plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
#plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
#ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=25
plt.xlabel('Diagnosis')
plt.ylabel('Association')
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
#ax.xaxis.set_label_coords(x=0.5,y= -0.2)
fig.tight_layout()
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/fisher_icu_group_log.jpg')


fisher = pd.read_csv(r'C:\Users\wendl\Documents\GitHub\GTT-Trigger-tool\Preprocessing_data_cleaning\Version13b\Fisher_holm_paper\fisher_covid_vent_overOR.csv',sep=',')
fisher['Bonferroni-Holm adj. P-values']=-np.log(fisher['Bonferroni-Holm adj. P-values'].astype(float))

fig = plt.figure(dpi=400, figsize=(21,10))
#x = ['Z51.5','J12.8','I10.00','I50.14','N18.4','J96.00','F05.1','N17.93','I48.9','J80.03','E87.0','E87.5','N39.0','D64.9','I25.13','N17.91','J80.02','R40.0','A49.9','K72.0','Z95.1']
ax = fisher.iloc[:7].plot.bar(x='Diagnose', y='Bonferroni-Holm adj. P-values',label='', rot=45, ax = plt.gca())
#plt.axhline(y=0.05,color='red',label='Significance level = 0.05')
plt.axhline(y=2.995732273553991,color='red',label='Significance level = 0.05')
#plt.axhline(y=-2.995732273553991,color='red')
plt.axhline(y=0,color='black')
#ax.set_xticklabels(x)
plt.legend()
BIGGER_SIZE=25
plt.xlabel('Diagnosis')
plt.ylabel('Association')
plt.rc('xtick', labelsize=BIGGER_SIZE);plt.rc('ytick', labelsize=BIGGER_SIZE)   
plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
#plt.title('Bonferroni-Holm adjusted p-values of t-test for death')
#ax.xaxis.set_label_coords(x=0.5,y= -0.2)
fig.tight_layout()
plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/fisher_vent_group_log.jpg')

#Venn-diagramm

death_group = 0
icu_group = 0
death_icu_group = 0
vent_group = 0
death_vent_group = 0
icu_vent_group = 0
allgroup = 0



for i in covid_cases:
    if i in deaths_covid["Number case"].values and i not in icu_covid["Number case"].values and i not in vent_covid["Number case"].values:
        death_group = death_group + 1
    elif i not in deaths_covid["Number case"].values and i in icu_covid["Number case"].values and i not in vent_covid["Number case"].values:
        icu_group = icu_group + 1
    elif i in deaths_covid["Number case"].values and i in icu_covid["Number case"].values and i not in vent_covid["Number case"].values:
        death_icu_group = death_icu_group + 1
    elif i not in deaths_covid["Number case"].values and i not in icu_covid["Number case"].values and i in vent_covid["Number case"].values:
        vent_group = vent_group + 1
    elif i in deaths_covid["Number case"].values and i not in icu_covid["Number case"].values and i in vent_covid["Number case"].values:
        death_vent_group = death_vent_group + 1
    elif i not in deaths_covid["Number case"].values and i in icu_covid["Number case"].values and i in vent_covid["Number case"].values:
        icu_vent_group =icu_vent_group + 1
    elif i in deaths_covid["Number case"].values and i in icu_covid["Number case"].values and i in vent_covid["Number case"].values:
        allgroup = allgroup + 1

from matplotlib_venn import venn3

fig = plt.figure(dpi=300, figsize=(10,10))

venn = venn3(subsets = (death_group, icu_group, death_icu_group, vent_group, death_vent_group, icu_vent_group, allgroup), set_labels = ('In-hospital mortality', 'Transfer to ICU', 'Mechanical ventilation'), alpha = 0.5);
lbl = venn.get_label_by_id('A')
x, y = lbl.get_position()
lbl.set_position((x+0.55, y+0.05)) 

lbl = venn.get_label_by_id('B')
x, y = lbl.get_position()
lbl.set_position((x-0.3, y+0.05)) 

lbl = venn.get_label_by_id('C')
x, y = lbl.get_position()
lbl.set_position((x-0.25, y)) 

for text in venn.set_labels:
    text.set_fontsize(30)
for text in venn.subset_labels:
    text.set_fontsize(30)

plt.savefig('C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/BarplotFeature/venn.jpg')


