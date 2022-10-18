import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
import matplotlib.pyplot as plt
from statsmodels.stats import multitest
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from itertools import chain, combinations
from collections import Counter
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from skopt import BayesSearchCV
import seaborn as sb
    
def data_numeric_labvalues_BZ_48h(lab):
    ''' dataset which includes all numerical labvalues and BZ (all time points togehter) '''
    labvalues = lab.iloc[:,0:6]
    labvalues['Lab_name'][['Glucose' in str(labvalues['Lab_name_long'].values[i]) for i in range(labvalues['Lab_name'].shape[0])]]='BZ'
    labvalues = labvalues['Lab_name'].unique()
    temp = []
    for i in lab['Lab_name'].unique():
        print(i)
        if pd.notna(lab['Einheit'][lab['Lab_name']==i].unique()[0]) and sum(pd.notna(lab['Value'][lab['Lab_name']==i]))>0:# and 'BZ' not in str(i):
             temp.append(i)
    for i in np.array(['INR','LDLHDL','pH','P-Sp.G','SpeGew','TFS','U-E/Kr','UpH','U-pH','USpG','Ct-Wer']):
        temp.append(i)
    temp = np.array(temp)
    labvalues = np.intersect1d(temp,labvalues)
    labvalues
    # Merge all BZ to one variable 
    BZ = lab.iloc[:,0:6]
    BZ['Lab_name'][['Glucose' in str(BZ['Lab_name_long'].values[i]) for i in range(BZ['Lab_name'].shape[0])]]='BZ'
    BZ = BZ[:][['Urin-Glucose qual.' != str(i) for i in BZ['Lab_name_long']]]
    BZ = BZ[:][BZ['Lab_name']=='BZ']
    BZ
    # Dataframe for Labvalues without BZ
    nonBZ = lab.iloc[:,0:6]
    nonBZ = nonBZ[:][[str(i) in labvalues for i in nonBZ['Lab_name']]]
    nonBZ
    # Merge both dataframes
    data = pd.concat([BZ,nonBZ])
    

    return(data) 

def lab_48h(lab,covid_cases,bew):
    ''' Create Dataframe with averaged labvalues for every patient of the first
        48 hours'''
    lab_48h = pd.DataFrame(columns=lab.columns)
    for case in covid_cases:
        #print(case)
        #case = 1159392
        bew_case=bew[bew['Number case']==case]
        aufnahme = bew[bew['Number case']==case][['Aufnahme' in i for i in bew_case['Bewegungstyp']]]    
        lab_case = lab[lab['Number case'] == case] 
        #lab_case['Date'] - aufnahme['Date'] > np.timedelta64(0,'D')
        #lab_case['Date'] - aufnahme['Date'] < np.timedelta64(48,'h')
        #sum([(i - aufnahme['Date'].values[0]) > np.timedelta64(0,'D') and (i - aufnahme['Date'].values[0]) < np.timedelta64(48,'h') for i in lab_case['Date']])
        mask = []
        for i in lab_case['Date']:       
            if pd.notna(i):
                value = (i - aufnahme['Date'].values[0]) > np.timedelta64(0,'D') and (i - aufnahme['Date'].values[0]) < np.timedelta64(48,'h')
                mask.append(value)
            else:
                mask.append(False)
        lab_case_48h = lab_case[mask]
        lab_48h = pd.concat([lab_48h,lab_case_48h])
    print(lab_48h)
    
    data = data_numeric_labvalues_BZ_48h(lab_48h)
    
    corr_help_48h = pd.DataFrame(index=data['Number case'].unique(),columns=data['Lab_name'].unique())
    for i in range(data['Lab_name'].unique().shape[0]):
        temp = data[:][data['Lab_name']==data['Lab_name'].unique()[i]]
        temp2 = pd.to_numeric(temp['Value'],errors='coerce')
        temp=temp[temp2.notna()]
        lab = []
        for case in data['Number case'].unique():
            lab.append((temp['Value'][temp['Number case']==case]).astype(float,errors='ignore').mean(skipna=True))
        lab = np.array(lab)
        corr_help_48h.iloc[:,i] = lab
        print(i)
    
    return lab_48h, corr_help_48h


def lab_24h(lab,covid_cases,bew):
    ''' Create Dataframe with averaged labvalues for every patient of the first
        24 hours'''
    lab_24h = pd.DataFrame(columns=lab.columns)
    for case in covid_cases:
        #print(case)
        #case = 1159392
        bew_case=bew[bew['Number case']==case]
        aufnahme = bew[bew['Number case']==case][['Aufnahme' in i for i in bew_case['Bewegungstyp']]]    
        lab_case = lab[lab['Number case'] == case] 
        #lab_case['Date'] - aufnahme['Date'] > np.timedelta64(0,'D')
        #lab_case['Date'] - aufnahme['Date'] < np.timedelta64(48,'h')
        #sum([(i - aufnahme['Date'].values[0]) > np.timedelta64(0,'D') and (i - aufnahme['Date'].values[0]) < np.timedelta64(48,'h') for i in lab_case['Date']])
        mask = []
        for i in lab_case['Date']:       
            if pd.notna(i):
                value = (i - aufnahme['Date'].values[0]) > np.timedelta64(0,'D') and (i - aufnahme['Date'].values[0]) < np.timedelta64(24,'h')
                mask.append(value)
            else:
                mask.append(False)
        lab_case_24h = lab_case[mask]
        lab_24h = pd.concat([lab_24h,lab_case_24h])
    print(lab_24h)
    
    data = data_numeric_labvalues_BZ_48h(lab_24h)
    
    corr_help_24h = pd.DataFrame(index=data['Number case'].unique(),columns=data['Lab_name'].unique())
    for i in range(data['Lab_name'].unique().shape[0]):
        temp = data[:][data['Lab_name']==data['Lab_name'].unique()[i]]
        temp2 = pd.to_numeric(temp['Value'],errors='coerce')
        temp=temp[temp2.notna()]
        lab = []
        for case in data['Number case'].unique():
            lab.append((temp['Value'][temp['Number case']==case]).astype(float,errors='ignore').mean(skipna=True))
        lab = np.array(lab)
        corr_help_24h.iloc[:,i] = lab
        print(i)
    
    return lab_24h, corr_help_24h

def wilcoxon_without_BZ(lab, labvalues, trigger_cases, nontrigger_cases):
    ''' labvalues: output of function labvalues_numeric
        wilcoxon tests for each labvalue which is numeric (without BZ) BZ=Blutzucker=Blood sugar '''
    wilcoxon = []
    i=1
    for labvalue in labvalues:
        print(i, labvalue)
        i=i+1
        temp = lab[:][lab['Lab_name']==labvalue]
        temp['Value'] = temp['Value'][['<' not in str(i) for i in temp['Value']]]
        temp['Value'] = temp['Value'][['>' not in str(i) for i in temp['Value']]]
        temp['Value'] = temp['Value'][['massenh' not in str(i) for i in temp['Value']]]
        temp['Value'] = temp['Value'][['negativ' not in str(i) for i in temp['Value']]]
        temp2 = pd.to_numeric(temp['Value'],errors='coerce')
        temp=temp[temp2.notna()]
        yes = []
        for case in trigger_cases:
            yes.append((temp['Value'][lab['Number case']==case]).astype(float).mean(skipna=True))
        yes = np.array(yes)
        no = []
        for case in nontrigger_cases:
            no.append((temp['Value'][lab['Number case']==case]).astype(float).mean(skipna=True))
        no = np.array(no)
        if not (np.unique(yes[~np.isnan(yes)]).shape[0]==1 and np.unique(no[~np.isnan(no)]).shape[0]==1 and np.unique(yes[~np.isnan(yes)]) == np.unique(no[~np.isnan(no)])[0]) and np.unique(yes[~np.isnan(yes)]).shape[0]>0 and np.unique(no[~np.isnan(no)]).shape[0]>0:
            W,p = mannwhitneyu(yes[~np.isnan(yes)],no[~np.isnan(no)])
        else:
            W=np.nan
            p=np.nan
        notnan_trigger = len(yes)-np.sum(np.isnan(yes))
        notnan_nontrigger = len(no)-np.sum(np.isnan(no))
        if notnan_trigger==0 or notnan_nontrigger==0:
            W=np.nan
            p=np.nan
        if not np.isnan(W):
            wilcoxon.append([labvalue,W,p,notnan_trigger,notnan_nontrigger])
    wilcoxon = pd.DataFrame(wilcoxon, columns = ['Labvalue','Statistic','P-value','notnan Trigger','notnan Nontrigger'])
    return(wilcoxon)

def wilcoxon_BZ(lab, wilcoxon, trigger_cases, nontrigger_cases):
    ''' wilcoxon: output of function wilcoxon_without_BZ
        wilcoxon test for labvalue BZ (all time points) BZ=Blutzucker=Blood sugar '''
    labvalues = lab.iloc[:,0:6]
    labvalues['Lab_name'][['Glucose' in str(labvalues['Lab_name_long'].values[i]) for i in range(labvalues['Lab_name'].shape[0])]]='BZ'
    labvalues = labvalues[:][['Urin-Glucose qual.' != str(i) for i in labvalues['Lab_name_long']]]
    labvalues = labvalues[:][labvalues['Lab_name']=='BZ']
    labvalues
    temp2 = pd.to_numeric(labvalues['Value'],errors='coerce')
    labvalues=labvalues[temp2.notna()]
    yes = []
    for case in trigger_cases:
        yes.append((labvalues['Value'][labvalues['Number case']==case]).astype(float).mean(skipna=True))
    yes = np.array(yes)
    no = []
    for case in nontrigger_cases:
        no.append((labvalues['Value'][labvalues['Number case']==case]).astype(float).mean(skipna=True))
    no = np.array(no)
    notnan_trigger = len(yes)-np.sum(np.isnan(yes))
    notnan_nontrigger = len(no)-np.sum(np.isnan(no)) 
    if not (np.unique(yes[~np.isnan(yes)]).shape[0]==1 and np.unique(no[~np.isnan(no)]).shape[0]==1 and np.unique(yes[~np.isnan(yes)]) == np.unique(no[~np.isnan(no)])[0]) and np.unique(yes[~np.isnan(yes)]).shape[0]>0 and np.unique(no[~np.isnan(no)]).shape[0]>0:
        W,p = mannwhitneyu(yes[~np.isnan(yes)],no[~np.isnan(no)])
    else:
        W=np.nan
        p=np.nan
    notnan_trigger = len(yes)-np.sum(np.isnan(yes))
    notnan_nontrigger = len(no)-np.sum(np.isnan(no))
    if notnan_trigger==0 or notnan_nontrigger==0:
        W=np.nan
        p=np.nan
    else:
        wilcoxon = wilcoxon.append({'Labvalue' : 'BZ' , 'Statistic' : W, 'P-value' : p, 'notnan Trigger' : notnan_trigger, 'notnan Nontrigger' : notnan_nontrigger} , ignore_index=True)
    return(wilcoxon)

def wilcoxon_multiple_test(wilcoxon):
    reject, qvalue, alphaS, alphaB = multitest.multipletests(wilcoxon['P-value'],method='holm-sidak')
    wilcoxon.insert(3,'Bonferroni-Holm adj. P-values',qvalue) 
    reject, qvalue, alphaS, alphaB = multitest.multipletests(wilcoxon['P-value'],method='fdr_bh')
    wilcoxon.insert(4,'Benjamini-Hochberg adj. P-values',qvalue)
    wilcoxon = wilcoxon.sort_values(by=['P-value','Labvalue'])
    return(wilcoxon)

def ttest_without_BZ(lab,labvalues,trigger_cases,nontrigger_cases):
    ''' labvalues: output of function labvalues_numeric
        ttest tests for each labvalue which is numeric (without BZ) '''
    ttest = []
    i=1
    for labvalue in labvalues:
        print(i, labvalue)
        i=i+1
        temp = lab[:][lab['Lab_name']==labvalue]
        temp['Value'] = temp['Value'][['<' not in str(i) for i in temp['Value']]]
        temp['Value'] = temp['Value'][['>' not in str(i) for i in temp['Value']]]
        temp['Value'] = temp['Value'][['massenh' not in str(i) for i in temp['Value']]]
        temp['Value'] = temp['Value'][['negativ' not in str(i) for i in temp['Value']]]
        temp2 = pd.to_numeric(temp['Value'],errors='coerce')
        temp=temp[temp2.notna()]
        yes = []
        for case in trigger_cases:
            yes.append((temp['Value'][lab['Number case']==case]).astype(float).mean(skipna=True))
        yes = np.array(yes)
        no = []
        for case in nontrigger_cases:
            no.append((temp['Value'][lab['Number case']==case]).astype(float).mean(skipna=True))
        no = np.array(no)
        T,p = ttest_ind(yes,no,equal_var=False,nan_policy='omit')
        notnan_trigger = len(yes)-np.sum(np.isnan(yes))
        notnan_nontrigger = len(no)-np.sum(np.isnan(no))
        ttest.append([labvalue,T,p,notnan_trigger,notnan_nontrigger])
    ttest = pd.DataFrame(ttest, columns = ['Labvalue','Statistic','P-value','notnan Trigger','notnan Nontrigger'])
    return(ttest)



def ttest_BZ(lab,ttest,trigger_cases,nontrigger_cases):
    ''' ttest: output of function test_without_BZ
        ttest test for labvalue BZ (all time points) '''
    # Glucose separat berechnen
    labvalues = lab.iloc[:,0:6]
    labvalues['Lab_name'][['Glucose' in str(labvalues['Lab_name_long'].values[i]) for i in range(labvalues['Lab_name'].shape[0])]]='BZ'
    labvalues = labvalues[:][['Urin-Glucose qual.' != str(i) for i in labvalues['Lab_name_long']]]
    labvalues = labvalues[:][labvalues['Lab_name']=='BZ']
    labvalues
    temp2 = pd.to_numeric(labvalues['Value'],errors='coerce')
    labvalues=labvalues[temp2.notna()]
    yes = []
    for case in trigger_cases:
        yes.append((labvalues['Value'][labvalues['Number case']==case]).astype(float).mean(skipna=True))
    yes = np.array(yes)
    no = []
    for case in nontrigger_cases:
        no.append((labvalues['Value'][labvalues['Number case']==case]).astype(float).mean(skipna=True))
    no = np.array(no)
    notnan_trigger = len(yes)-np.sum(np.isnan(yes))
    notnan_nontrigger = len(no)-np.sum(np.isnan(no)) 
    T,p = ttest_ind(yes,no,equal_var=False,nan_policy='omit')
    ttest = ttest.append({'Labvalue' : 'BZ' , 'Statistic' : T, 'P-value' : p, 'notnan Trigger' : notnan_trigger, 'notnan Nontrigger' : notnan_nontrigger} , ignore_index=True)
    return(ttest)

def ttest_multiple_test(ttest):
    reject, qvalue, alphaS, alphaB = multitest.multipletests(ttest['P-value'],method='holm-sidak')
    ttest.insert(3,'Bonferroni-Holm adj. P-values',qvalue) 
    reject, qvalue, alphaS, alphaB = multitest.multipletests(ttest['P-value'],method='fdr_bh')
    ttest.insert(4,'Benjamini-Hochberg adj. P-values',qvalue)
    ttest = ttest[:][ttest['P-value']!='--']
    ttest = ttest.sort_values(by=['P-value','Labvalue'])
    return(ttest)

### Logistic Regression
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.05, 
                       verbose=True):
    #https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-a-forward-selection-stepwise-regression-algorithm
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included+[new_column]])),missing='drop').fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included])),missing='drop').fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def powerset(iterable):
#https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def all_combinations(list_variables,x_train,y_train):
    all_combinations=list(powerset(list_variables))
    aic_list = np.zeros(shape=len(all_combinations))
    aic_list[:] = np.nan
    bic_list = np.zeros(shape=len(all_combinations))
    bic_list[:] = np.nan
    for i in range(len(all_combinations)):
        log_reg = sm.Logit(y_train,sm.add_constant(x_train[list(all_combinations[i])]),missing='drop').fit()
        aic_list[i]=log_reg.aic
        bic_list[i]=log_reg.bic
    min_aic = np.min(aic_list)
    min_aic_variables = all_combinations[np.argmin(aic_list)]
    min_bic = np.min(bic_list)
    min_bic_variables = all_combinations[np.argmin(bic_list)]
    return min_aic, min_aic_variables, min_bic, min_bic_variables
    

def logReg_cv2(x_train,y_train,all_combis=False,n_splits=5,random_state=4):
    kf = KFold(n_splits, shuffle=True,random_state=random_state)
    
    x_train_list=[]
    x_test_list=[]
    y_train_list=[]
    y_test_list=[]
    for train_index, test_index in kf.split(x_train):
        x_train1, x_test1 = x_train.iloc[train_index,:], x_train.iloc[test_index,:]
        x_train_list.append(x_train1)
        x_test_list.append(x_test1)
        y_train1, y_test1 = np.array(y_train)[train_index], np.array(y_train)[test_index]
        y_train_list.append(y_train1)
        y_test_list.append(y_test1)
        
    predict_list=[]
    if all_combis:
        for i in range(n_splits):
            min_aic, min_aic_variables, min_bic, min_bic_variables = all_combinations(x_train_list[i].columns, x_train_list[i], y_train_list[i])
            result = list(min_bic_variables)
            predict_list.extend(result)
        Count=Counter(predict_list)
        variables=list({x: count for x, count in Count.items() if count >= n_splits/2}.keys())
            
    else:
        for i in range(n_splits):
            result = stepwise_selection(x_train_list[i], y_train_list[i])
            print(result)
            predict_list.extend(result)
        Count=Counter(predict_list)
        variables=list({x: count for x, count in Count.items() if count >= n_splits/2}.keys())
        print(predict_list)

    result=variables
    print('resulting features:')
    print(result)
    
    log_reg_list=[]    
    y_pred_list=[]
    for i in range(n_splits):
        log_reg = sm.Logit(y_train_list[i],sm.add_constant(x_train_list[i][result]),missing='drop').fit()
        print(log_reg.summary())
        log_reg_list.append(log_reg)
        y_pred = log_reg.predict(sm.add_constant(x_test_list[i][result]))
        y_pred_list.append(y_pred)
    
    # ROC - AUC
    
    roc_point_list = []
    pr_point_list = []
    for i in range(n_splits):
        roc_point = []
        pr_point = []
        thresholds = list(np.array(list(range(0,101,1)))/100)
        for threshold in thresholds:
            tp=0; fp=0; fn=0; tn=0
            y_test = list(pd.Series(y_test_list[i])[np.invert(np.array(y_pred_list[i].isna()))])
            y_pred = y_pred_list[i].dropna()
            for k in range(len(y_test)):
                if y_pred.iloc[k] >= threshold:
                    prediction = 1
                else: 
                    prediction = 0
                if prediction == 1 and y_test[k] == 1:
                    tp = tp+1
                elif prediction == 0 and y_test[k] == 1:
                    fn = fn+1
                elif prediction == 1 and y_test[k] == 0:
                    fp = fp+1
                elif prediction == 0 and y_test[k] == 0:
                    tn = tn+1
            tpr = tp/(tp+fn)
            fpr = fp/(tn+fp)
            if (fp+tp)==0:
                prec = 0
            else:
                prec = tp/(fp+tp)
            rec = tp/(tp+fn)
            roc_point.append([tpr,fpr]) 
            pr_point.append([rec,prec])
        roc_point_list.append(roc_point)
        pr_point_list.append(pr_point)
        p=pd.DataFrame(roc_point_list[i],columns=['tpr','fpr'])
        pr=pd.DataFrame(pr_point_list[i],columns=['rec','prec'])
        print(auc(p.fpr,p.tpr))
    
    return result, log_reg, fpr, tpr

def logRegTest(x_train, X_test, y_train, y_test, result):
    X_train = x_train        
    log_reg = sm.Logit(y_train,sm.add_constant(X_train[result]),missing='drop').fit()
    print(log_reg.summary())
    y_pred = log_reg.predict(sm.add_constant(X_test[result]))
    
    # ROC - AUC    
    y_test = list(pd.Series(y_test)[np.invert(np.array(y_pred.isna()))])
    y_pred = y_pred.dropna()
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    return log_reg, y_pred, y_test

def logRegTest24(X_test, y_test, log_reg, result):
    y_pred = log_reg.predict(sm.add_constant(X_test[result]))
    
    # ROC - AUC    
    y_test = list(pd.Series(y_test)[np.invert(np.array(y_pred.isna()))])
    y_pred = y_pred.dropna()
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    return log_reg, y_pred, y_test

def RandomForest(x_train,y_train,random_state=4):
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)
    
    y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
    
    rf = RandomForestClassifier(random_state=0)
    rf = rf.fit(X_train.dropna(), y_train)
    y_pred = rf.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    
    
    
def RandomForestFeatSelection_cv(x_train,y_train,n_splits=5):
    kf = KFold(n_splits, shuffle=True,random_state=4)
    
    x_train_list=[]
    x_test_list=[]
    y_train_list=[]
    y_test_list=[]
    for train_index, test_index in kf.split(x_train):
        x_train1, x_test1 = x_train.iloc[train_index,:], x_train.iloc[test_index,:]
        x_train_list.append(x_train1)
        x_test_list.append(x_test1)
        y_train1, y_test1 = np.array(y_train)[train_index], np.array(y_train)[test_index]
        y_train_list.append(y_train1)
        y_test_list.append(y_test1)
        
    predict_list=[]
    for i in range(n_splits):
        y_train1 = list(pd.Series(y_train_list[i])[np.array(x_train_list[i].isna().sum(axis=1)==0)])
        y_test1 = list(pd.Series(y_test_list[i])[np.array(x_test_list[i].isna().sum(axis=1)==0)])
        sel = SelectFromModel(RandomForestClassifier(random_state=0))
        sel.fit(x_train_list[i].dropna(), y_train1)
        selected_feat= x_train_list[i].columns[(sel.get_support())]
        #print(selected_feat)
        predict_list.extend(selected_feat)
    Count=Counter(predict_list)
    variables=list({x: count for x, count in Count.items() if count >= n_splits/2}.keys())
    #print(predict_list)
        
    result=variables
    print('resulting features:')
    print(result)
      
    auc_list=[]
    for i in range(n_splits):
        y_train1 = list(pd.Series(y_train_list[i])[np.array(x_train_list[i].isna().sum(axis=1)==0)])
        y_test1 = list(pd.Series(y_test_list[i])[np.array(x_test_list[i].isna().sum(axis=1)==0)])
        rf = RandomForestClassifier(random_state=0)
        rf = rf.fit(x_train_list[i].dropna()[result], y_train1)
        y_pred = rf.predict_proba(x_test_list[i].dropna()[result])[:,1]
        fpr, tpr, thresh = roc_curve(y_test1,y_pred)
        roc_auc = auc(fpr,tpr)
        auc_list.append(roc_auc)
        print(roc_auc)
    print(np.mean(auc_list))

def RandomForestHyper(x_train,y_train,random_state=4):
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)

    y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
        
    rf = RandomForestClassifier(random_state=0)
    rf = rf.fit(X_train.dropna(), y_train)
    y_pred = rf.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    
    rf = RandomForestClassifier(random_state=0)
    distributions = {'n_estimators':[50,75,100,200,300,400,500],
                     'max_depth':[2,4,6,8,10],
                     'min_samples_leaf':[1,2,4],
                     'min_samples_split':[2,4,6,8,10]}
    opt = BayesSearchCV(rf,distributions,random_state=4,
                        scoring='roc_auc',cv=5,n_iter=100)
    opt.fit(X_train.dropna(),y_train)
    #opt.fit(X_res,y_res)
    print(opt.best_estimator_)
    print(opt.best_score_)
    
    # Anwendung auf Testdaten
    rf2 = opt.best_estimator_.fit(X_train.dropna(), y_train)
    #rf2 = opt.best_estimator_.fit(X_res,y_res)
    y_pred = rf2.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    return rf2, roc_auc


def RandomForestHyperTest(x_train, X_test, y_train, y_test, rf):
    X_train = x_train
    y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
        
    rf = rf.fit(X_train.dropna(), y_train)
    y_pred = rf.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    
def RandomForestTest(x_train, X_test, y_train, y_test, result):
    X_train = x_train[result]
    X_test = X_test[result]
    y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
        
    rf = RandomForestClassifier(random_state=0)
    rf = rf.fit(X_train.dropna(), y_train)
    y_pred = rf.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    return rf, y_pred, y_test, X_test

def RandomForestTest24(X_test, y_test, rf, result):
    X_test = X_test[result]
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
        
    y_pred = rf.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    return rf, y_pred, y_test, X_test

    
def xgboost(x_train,y_train,random_state=4):
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)

    y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
    
    xgb = XGBClassifier(random_state=0)
    xgb.fit(X_train.dropna(), y_train)
    y_pred = xgb.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    

def xgboostHyper(x_train,y_train,random_state=4):
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=random_state)

    y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
        
    xgb = XGBClassifier(random_state=0)
    xgb.fit(X_train.dropna(), y_train)
    y_pred = xgb.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    
    xgb = XGBClassifier(random_state=0)
    distributions = {'n_estimators': [100,200,300,400,500],
                  'learning_rate': [0.1],
                  'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                  'min_child_weight': [1, 2, 3, 4, 5, 6],
                  'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
                  'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                  }
    opt = BayesSearchCV(xgb,distributions,random_state=4,
                        scoring='roc_auc',cv=5,n_iter=100)
    opt.fit(X_train.dropna(),y_train)
    #opt.fit(X_res,y_res)
    print(opt.best_estimator_)
    print(opt.best_score_)
    
    # Anwendung auf Testdaten
    xgb2 = opt.best_estimator_.fit(X_train.dropna(), y_train)
    #rf2 = opt.best_estimator_.fit(X_res,y_res)
    y_pred = xgb2.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    return xgb2, roc_auc

def xgboostHyperTest(x_train, X_test, y_train, y_test, xgb):
    X_train = x_train
    y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
        
    xgb = xgb.fit(X_train.dropna(), y_train)
    y_pred = xgb.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)  
    return xgb, y_pred, y_test
    
def xgboostTest(x_train, X_test, y_train, y_test, result):
    X_train = x_train[result]
    X_test = X_test[result]
    y_train = list(pd.Series(y_train)[np.array(X_train.isna().sum(axis=1)==0)])
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
    
    xgb = XGBClassifier(random_state=0)
    xgb.fit(X_train.dropna(), y_train)
    y_pred = xgb.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    return xgb, y_pred, y_test

def xgboostTest24(X_test, y_test, xgb, result):
    X_test = X_test[result]
    y_test = list(pd.Series(y_test)[np.array(X_test.isna().sum(axis=1)==0)])
    
    y_pred = xgb.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    return xgb, y_pred, y_test
    
def xgboostFeatSelection_cv(x_train,y_train,n_splits=5):
    kf = KFold(n_splits, shuffle=True,random_state=4)
    
    x_train_list=[]
    x_test_list=[]
    y_train_list=[]
    y_test_list=[]
    for train_index, test_index in kf.split(x_train):
        x_train1, x_test1 = x_train.iloc[train_index,:], x_train.iloc[test_index,:]
        x_train_list.append(x_train1)
        x_test_list.append(x_test1)
        y_train1, y_test1 = np.array(y_train)[train_index], np.array(y_train)[test_index]
        y_train_list.append(y_train1)
        y_test_list.append(y_test1)
        
    predict_list=[]
    for i in range(n_splits):
        y_train1 = list(pd.Series(y_train_list[i])[np.array(x_train_list[i].isna().sum(axis=1)==0)])
        y_test1 = list(pd.Series(y_test_list[i])[np.array(x_test_list[i].isna().sum(axis=1)==0)])
        sel = SelectFromModel(XGBClassifier(random_state=0))
        sel.fit(x_train_list[i].dropna(), y_train1)
        selected_feat= x_train_list[i].columns[(sel.get_support())]
        #print(selected_feat)
        predict_list.extend(selected_feat)
    Count=Counter(predict_list)
    variables=list({x: count for x, count in Count.items() if count >= n_splits/2}.keys())
    #print(predict_list)
        
    result=variables
    print('resulting features:')
    print(result)
      
    auc_list=[]
    for i in range(n_splits):
        y_train1 = list(pd.Series(y_train_list[i])[np.array(x_train_list[i].isna().sum(axis=1)==0)])
        y_test1 = list(pd.Series(y_test_list[i])[np.array(x_test_list[i].isna().sum(axis=1)==0)])
        xgb = XGBClassifier(random_state=0)
        xgb = xgb.fit(x_train_list[i].dropna()[result], y_train1)
        y_pred = xgb.predict_proba(x_test_list[i].dropna()[result])[:,1]
        fpr, tpr, thresh = roc_curve(y_test1,y_pred)
        roc_auc = auc(fpr,tpr)
        auc_list.append(roc_auc)
        print(roc_auc)
    print(np.mean(auc_list))
    

def violinplot_dead_logreg_alternativ(deaths_covid,data):
    BIGGER_SIZE = 40
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=40)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=37)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=25)  
    
    fig, axes = plt.subplots(2,2,sharex=False,figsize=(20,11),dpi=400) #altes Format: 14,14
    #fig.suptitle('Violinplot of predictors for death')
    plt.subplots_adjust(hspace=0.3)
    y_help = [int(i) in np.array(deaths_covid['Number case']) for i in data.index]
    violin = pd.DataFrame(data['CRP'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,0],y='CRP',x='Beatmung',data=violin,cut=0)
    axes[0,0].set_title('CRP'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
    axes[0,0].set_xticklabels(['Survived','Deceased'])
    violin = pd.DataFrame(data['Age'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,1],y='Age',x='Beatmung',data=violin,cut=0)
    axes[0,1].set_title('Age'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel('');
    axes[0,1].set_xticklabels(['Survived','Deceased'])
    violin = pd.DataFrame(data['Hst'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,0],y='Hst',x='Beatmung',data=violin,cut=0)
    axes[1,0].set_title('Urea'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
    axes[1,0].set_xticklabels(['Survived','Deceased'])
    violin = pd.DataFrame(data['BZ'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,1],y='BZ',x='Beatmung',data=violin,cut=0)
    axes[1,1].set_title('Glucose'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
    axes[1,1].set_xticklabels(['Survived','Deceased'])
    
def violinplot_dead_logreg_alternativ_cat(deaths_covid,data):
    BIGGER_SIZE = 40
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=40)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=37)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=25)  
    
    fig, axes = plt.subplots(2,2,sharex=False,figsize=(20,11),dpi=400) #altes Format: 14,14
    #fig.suptitle('Violinplot of predictors for death')
    plt.subplots_adjust(hspace=0.3)
    y_help = [int(i) in np.array(deaths_covid['Number case']) for i in data.index]
    violin = pd.DataFrame(data['Hst'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,0],y='Hst',x='Beatmung',data=violin,cut=0)
    axes[0,0].set_title('Urea'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
    axes[0,0].set_xticklabels(['Survived','Deceased'])
    violin = pd.DataFrame(data['PTT'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,1],y='PTT',x='Beatmung',data=violin,cut=0)
    axes[0,1].set_title('PTT'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel('');
    axes[0,1].set_xticklabels(['Survived','Deceased'])
    violin = pd.DataFrame(data['GOT'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,0],y='GOT',x='Beatmung',data=violin,cut=0)
    axes[1,0].set_title('GOT'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
    axes[1,0].set_xticklabels(['Survived','Deceased'])
    violin = pd.DataFrame(data['Age'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,1],y='Age',x='Beatmung',data=violin,cut=0)
    axes[1,1].set_title('Age'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
    axes[1,1].set_xticklabels(['Survived','Deceased'])

def violinplot_vent_xgb_cat(beatmung_covid,data):    
    BIGGER_SIZE = 40
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=40)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=37)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=25)  
    
    fig, axes = plt.subplots(2,2,sharex=False,figsize=(20,11),dpi=400) #altes Format: 14,14
    #fig.suptitle('Violinplot of predictors for death')
    plt.subplots_adjust(hspace=0.3)
    y_help = [int(i) in np.array(beatmung_covid['Number case']) for i in data.index]
    violin = pd.DataFrame(data['Ca'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,0],y='Ca',x='Beatmung',data=violin,cut=0)
    axes[0,0].set_title('Calcium'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
    axes[0,0].set_xticklabels(['Natural', 'Mechanical'])
    violin = pd.DataFrame(data['PTT'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,1],y='PTT',x='Beatmung',data=violin,cut=0)
    axes[0,1].set_title('PTT'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel('');
    axes[0,1].set_xticklabels(['Natural', 'Mechanical'])
    violin = pd.DataFrame(data['GOT'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,0],y='GOT',x='Beatmung',data=violin,cut=0)
    axes[1,0].set_title('GOT'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
    axes[1,0].set_xticklabels(['Natural', 'Mechanical'])
    violin = pd.DataFrame(data['CRP'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,1],y='CRP',x='Beatmung',data=violin,cut=0)
    axes[1,1].set_title('CRP'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
    axes[1,1].set_xticklabels(['Natural', 'Mechanical'])
    
def violinplot_icu_logreg_cat(intensiv_covid,data):
    BIGGER_SIZE = 40
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=40)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=37)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=40)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=25)  
    
    fig, axes = plt.subplots(2,2,sharex=False,figsize=(20,11),dpi=400) #altes Format: 14,14
    #fig.suptitle('Violinplot of predictors for death')
    plt.subplots_adjust(hspace=0.3)
    y_help = [int(i) in np.array(intensiv_covid['Number case']) for i in data.index]
    violin = pd.DataFrame(data['Ca'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,0],y='Ca',x='Beatmung',data=violin,cut=0)
    axes[0,0].set_title('Calcium'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
    axes[0,0].set_xticklabels(['Not ICU','ICU'])
    violin = pd.DataFrame(data['PTT'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,1],y='PTT',x='Beatmung',data=violin,cut=0)
    axes[0,1].set_title('PTT'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel('');
    axes[0,1].set_xticklabels(['Not ICU','ICU'])
    violin = pd.DataFrame(data['GOT'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,0],y='GOT',x='Beatmung',data=violin,cut=0)
    axes[1,0].set_title('GOT'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
    axes[1,0].set_xticklabels(['Not ICU','ICU'])
    violin = pd.DataFrame(data['Sex'])
    violin['Beatmung'] = y_help
    violin_w = violin[violin['Sex']==0]
    violin_m = violin[violin['Sex']==1]
    pw = violin_w[violin_w['Beatmung']==True].shape[0]/violin_w.shape[0]
    pm = violin_m[violin_m['Beatmung']==True].shape[0]/violin_m.shape[0]
    sb.barplot(['Male','Female'],[pm,pw],ax=axes[1,1])
    axes[1,1].set_title('Biological Sex'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
    #axes[1,1].set_xticklabels(['Portion of m','Portion of f'])
    axes[1,1].set_xticklabels([r'$\dfrac{| \mathrm{Male} \  \mathrm{ICU}|}{|\mathrm{Male}|}$','$\dfrac{|\mathrm{Female} \ \mathrm{ICU}|}{|\mathrm{Female}|}$'])
    #plt.xticks(fontsize = 22)
    #plt.rc('ytick', labelsize=25)
     
def violinplot_dead_logreg(deaths_covid,data):
    BIGGER_SIZE = 22
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=25)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=25)  
    
    fig, axes = plt.subplots(2,3,sharex=False,figsize=(20,11),dpi=400) #altes Format: 14,14
    #fig.suptitle('Violinplot of predictors for death')
    y_help = [int(i) in np.array(deaths_covid['Number case']) for i in data.index]
    violin = pd.DataFrame(data['Ca'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,0],y='Ca',x='Beatmung',data=violin,cut=0)
    axes[0,0].set_title('Ca'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
    violin = pd.DataFrame(data['CRP'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,1],y='CRP',x='Beatmung',data=violin,cut=0)
    axes[0,1].set_title('CRP'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel('');
    violin = pd.DataFrame(data['Age'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,2],y='Age',x='Beatmung',data=violin,cut=0)
    axes[0,2].set_title('Age'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('');
    violin = pd.DataFrame(data['Thromb'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,0],y='Thromb',x='Beatmung',data=violin,cut=0)
    axes[1,0].set_title('PLT'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
    violin = pd.DataFrame(data['Na'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,1],y='Na',x='Beatmung',data=violin,cut=0)
    axes[1,1].set_title('So'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
    violin = pd.DataFrame(data['BZ'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,2],y='BZ',x='Beatmung',data=violin,cut=0)
    axes[1,2].set_title('Gluc'); axes[1,2].set_xlabel(''); axes[1,2].set_ylabel('');
    
def violinplot_icu_xgb(intensiv_covid,data):
    BIGGER_SIZE = 35
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=35)     # fontsize of the axes title
    plt.rc('axes', labelsize=35)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=35)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=33)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=25)  
    
    fig, axes = plt.subplots(2,3,sharex=False,figsize=(20,11),dpi=400) #altes Format: 14,14
    #fig.suptitle('Violinplot of predictors for transfer to ICU')
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    y_help = [int(i) in np.array(intensiv_covid['Number case']) for i in data.index]
    violin = pd.DataFrame(data['Age'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,0],y='Age',x='Beatmung',data=violin,cut=0)
    axes[0,0].set_title('Age'); axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('');
    axes[0,0].set_xticklabels(['Not ICU','ICU'])
    violin = pd.DataFrame(data['Ca'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,1],y='Ca',x='Beatmung',data=violin,cut=0)
    axes[0,1].set_title('Calcium'); axes[0,1].set_xlabel(''); axes[0,1].set_ylabel('');
    axes[0,1].set_xticklabels(['Not ICU','ICU'])
    violin = pd.DataFrame(data['CRP'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0,2],y='CRP',x='Beatmung',data=violin,cut=0)
    axes[0,2].set_title('CRP'); axes[0,2].set_xlabel(''); axes[0,2].set_ylabel('');
    axes[0,2].set_xticklabels(['Not ICU','ICU'])
    violin = pd.DataFrame(data['BZ'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,0],y='BZ',x='Beatmung',data=violin,cut=0)
    axes[1,0].set_title('Glucose'); axes[1,0].set_xlabel(''); axes[1,0].set_ylabel('');
    axes[1,0].set_xticklabels(['Not ICU','ICU'])
    violin = pd.DataFrame(data['GOT'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1,1],y='GOT',x='Beatmung',data=violin,cut=0)
    axes[1,1].set_title('GOT'); axes[1,1].set_xlabel(''); axes[1,1].set_ylabel('');
    axes[1,1].set_xticklabels(['Not ICU','ICU'])
    axes[1,2].axis('off')

def violinplot_vent_rf(beatmung_covid,data):
    BIGGER_SIZE = 40
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=40)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=32)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=25)  
    
    
    fig, axes = plt.subplots(1,3,sharex=False,figsize=(20,11),dpi=400) #altes Format: 14,14
    #fig.suptitle('Violinplot of predictors for mechanical ventilation')
    plt.subplots_adjust(wspace=0.3)
    y_help = [int(i) in np.array(beatmung_covid['Number case']) for i in data.index]
    violin = pd.DataFrame(data['Ca'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[0],y='Ca',x='Beatmung',data=violin,cut=0)
    axes[0].set_title('Calcium'); axes[0].set_xlabel(''); axes[0].set_ylabel('');
    axes[0].set_xticklabels(['Natural', 'Mechanical'])
    violin = pd.DataFrame(data['CRP'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[1],y='CRP',x='Beatmung',data=violin,cut=0)
    axes[1].set_title('CRP'); axes[1].set_xlabel(''); axes[1].set_ylabel('');
    axes[1].set_xticklabels(['Natural', 'Mechanical'])
    violin = pd.DataFrame(data['BZ'])
    violin['Beatmung'] = y_help
    sb.violinplot(ax=axes[2],y='BZ',x='Beatmung',data=violin,cut=0)
    axes[2].set_title('Glucose'); axes[2].set_xlabel(''); axes[2].set_ylabel('');
    axes[2].set_xticklabels(['Natural', 'Mechanical'])
    