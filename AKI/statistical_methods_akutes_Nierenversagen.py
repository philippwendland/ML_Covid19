import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats import multitest
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from itertools import chain, combinations
from collections import Counter
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from skopt import BayesSearchCV

### Wilcoxon and T-test
def wilcoxon_without_BZ(lab, labvalues, trigger_cases, nontrigger_cases):
    ''' labvalues: output of function labvalues_numeric
        wilcoxon tests for each labvalue which is numeric (without BZ) '''
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
        wilcoxon test for labvalue BZ (all time points) '''
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
    reject, qvalue, alphaS, alphaB = multitest.multipletests(wilcoxon['P-value'],method='holm')
    wilcoxon.insert(3,'q-value',qvalue) 
    reject, qvalue, alphaS, alphaB = multitest.multipletests(wilcoxon['P-value'],method='fdr_bh')
    wilcoxon.insert(4,'fdr',qvalue)
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
    # Computing glucose
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
    reject, qvalue, alphaS, alphaB = multitest.multipletests(ttest['P-value'],method='holm')
    ttest.insert(3,'q-value',qvalue) 
    reject, qvalue, alphaS, alphaB = multitest.multipletests(ttest['P-value'],method='fdr_bh')
    ttest.insert(4,'fdr',qvalue)
    ttest = ttest[:][ttest['P-value']!='--']
    ttest = ttest.sort_values(by=['P-value','Labvalue'])
    return(ttest)

def data_numeric_labvalues_BZ(lab):
    ''' dataset which includes all numerical labvalues and BZ (all time points togehter) '''
    labvalues = lab.iloc[:,0:6]
    labvalues['Lab_name'][['Glucose' in str(labvalues['Lab_name_long'].values[i]) for i in range(labvalues['Lab_name'].shape[0])]]='BZ'
    labvalues = labvalues['Lab_name'].unique()
    temp = []
    for i in lab['Lab_name'].unique():
        print(i)
        if pd.notna(lab['Einheit'][lab['Lab_name']==i].unique()[0]) and sum(pd.notna(lab['Value'][lab['Lab_name']==i]))>0:
             temp.append(i)
    for i in np.array(['INR','LDLHDL','pH','P-Sp.G','SpeGew','TFS','U-E/Kr','UpH','U-pH','USpG','Ct-Wer']):
        temp.append(i)
    temp = np.array(temp)
    labvalues = np.intersect1d(temp,labvalues)
    labvalues 
    BZ = lab.iloc[:,0:6]
    BZ['Lab_name'][['Glucose' in str(BZ['Lab_name_long'].values[i]) for i in range(BZ['Lab_name'].shape[0])]]='BZ'
    BZ = BZ[:][['Urin-Glucose qual.' != str(i) for i in BZ['Lab_name_long']]]
    BZ = BZ[:][BZ['Lab_name']=='BZ']
    BZ
    nonBZ = lab.iloc[:,0:6]
    nonBZ = nonBZ[:][[str(i) in labvalues for i in nonBZ['Lab_name']]]
    nonBZ
    data = pd.concat([BZ,nonBZ])
    
    return(data) 

def lab_24h(lab,covid_cases,bew):
    ''' Generates a dataset that assigns each patient an average (MW) of each laboratory value, but only laboratory values from the first 24 hours are used.'''
    lab_24h = pd.DataFrame(columns=lab.columns)
    for case in covid_cases:
        #print(case)
        #case = 1159392
        bew_case=bew[bew['Number case']==case]
        aufnahme = bew[bew['Number case']==case][['Aufnahme' in i for i in bew_case['Bewegungstyp']]]    
        lab_case = lab[lab['Number case'] == case] 
        mask = []
        if len(aufnahme)!=0 and pd.notna(aufnahme['Date'].values[0]):
            for i in lab_case['Date']:       
                if pd.notna(i):
                    value = (i - aufnahme['Date'].values[0]) > np.timedelta64(0,'D') and (i - aufnahme['Date'].values[0]) < np.timedelta64(24,'h')
                    mask.append(value)
                else:
                    mask.append(False)
            lab_case_24h = lab_case[mask]
            lab_24h = pd.concat([lab_24h,lab_case_24h])
    print(lab_24h)
    
    data = data_numeric_labvalues_BZ(lab_24h)
    
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


####################################################################
######               Logistic Regression                  #######
####################################################################

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
        #print(log_reg.aic)
    min_aic = np.min(aic_list)
    min_aic_variables = all_combinations[np.argmin(aic_list)]
    min_bic = np.min(bic_list)
    min_bic_variables = all_combinations[np.argmin(bic_list)]
    return min_aic, min_aic_variables, min_bic, min_bic_variables


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
            # a=X.corr()
            # print(a)
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
        worst_pval = pvalues.max() # null if p-values is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

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
    fig, ax = plt.subplots(figsize=(24,10))
    fig2, ax2 = plt.subplots(figsize=(24,10))
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    ax2.plot([0, 1], [0, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    
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
        ax.plot(p.fpr,p.tpr,alpha=0.3,label='ROC fold %0.0f (AUC = %0.2f)' % (i,auc(p.fpr,p.tpr)))
        ax2.plot(pr.rec,pr.prec,alpha=0.3,label='ROC fold %0.0f (AUC = %0.2f)' % (i, auc(pr.rec,pr.prec)))
        print(auc(p.fpr,p.tpr))
    
    tprs = []
    fprs = []
    precs = []
    recs = [] 
    p1=pd.DataFrame(roc_point_list[0],columns=['tpr','fpr'])
    p2=pd.DataFrame(roc_point_list[1],columns=['tpr','fpr'])
    p3=pd.DataFrame(roc_point_list[2],columns=['tpr','fpr'])
    p4=pd.DataFrame(roc_point_list[3],columns=['tpr','fpr'])
    p5=pd.DataFrame(roc_point_list[4],columns=['tpr','fpr'])
    pr1=pd.DataFrame(pr_point_list[0],columns=['rec','prec'])
    pr2=pd.DataFrame(pr_point_list[1],columns=['rec','prec'])
    pr3=pd.DataFrame(pr_point_list[2],columns=['rec','prec'])
    pr4=pd.DataFrame(pr_point_list[3],columns=['rec','prec'])
    pr5=pd.DataFrame(pr_point_list[4],columns=['rec','prec'])
    tprs.append(p1.tpr); tprs.append(p2.tpr); tprs.append(p3.tpr); tprs.append(p4.tpr); tprs.append(p5.tpr)
    fprs.append(p1.fpr); fprs.append(p2.fpr); fprs.append(p3.fpr); fprs.append(p4.fpr); fprs.append(p5.fpr)
    precs.append(pr1.prec); precs.append(pr2.prec); precs.append(pr3.prec); precs.append(pr4.prec); precs.append(pr5.prec)
    recs.append(pr1.rec); recs.append(pr2.rec); recs.append(pr3.rec); recs.append(pr4.rec); recs.append(pr5.rec) 
    mean_fpr = np.mean(fprs,axis=0)
    mean_tpr = np.mean(tprs,axis=0)
    mean_prec = np.mean(precs,axis=0)
    mean_rec = np.mean(recs,axis=0)
    print(len(mean_tpr))
    std_tpr = np.std(tprs,axis=0)
    std_prec = np.std(precs,axis=0)
    ax.plot(mean_fpr,mean_tpr,color='b',label='Mean ROC (AUC = %0.2f)' % auc(mean_fpr,mean_tpr))
    ax2.plot(mean_rec,mean_prec,color='b',label='Mean ROC (AUC = %0.2f)' % auc(mean_rec,mean_prec))
    
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = np.maximum(mean_prec - std_prec, 0)
    ax.fill_between(mean_fpr,tprs_lower,tprs_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.") 
    ax2.fill_between(mean_rec,prec_lower,prec_upper,color="grey",alpha=0.2,label=r"$\pm$ 1 std. dev.") 
    
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)          
    plt.rc('axes', titlesize=BIGGER_SIZE)   
    plt.rc('axes', labelsize=BIGGER_SIZE)   
    plt.rc('xtick', labelsize=BIGGER_SIZE)    
    plt.rc('ytick', labelsize=BIGGER_SIZE) 
    plt.rc('legend', fontsize=BIGGER_SIZE)  
    plt.rc('figure', titlesize=BIGGER_SIZE)  
    
    ax.set(ylabel='true positive rate',xlabel='false positive rate',title="ROC Kurve")
    ax.legend(loc="lower right")
    plt.show()
    ax2.set(ylabel='recall',xlabel='precision',title="Precision-Recall-Curve")
    ax2.legend(loc="upper right")
    plt.show()
        
    return result, log_reg, fpr, tpr

def logReg(x_train,y_train,all_combis=False,n_splits=5,random_state=4):
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
    
    return result

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
    
    # #AUC der Trainingsdaten
    # y_pred = log_reg.predict(sm.add_constant(X_train[result]))
    # y_train = list(pd.Series(y_train)[np.invert(np.array(y_pred.isna()))])
    # y_pred = y_pred.dropna()
    # fpr, tpr, thresh = roc_curve(y_train,y_pred)
    # roc_auc = auc(fpr,tpr)
    # print('AUC der Trainingsdaten:')
    # print(roc_auc)    
    
    return log_reg, y_pred, y_test

#######################################################################################
############################# Random Forest  ##########################################
#######################################################################################

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
    
    return result

def RandomForestHyper(x_train,y_train,X_test,y_test):
    X_train = x_train
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
    print(opt.best_estimator_)
    print(opt.best_score_)
    
    # Anwendung auf Testdaten
    rf2 = opt.best_estimator_.fit(X_train.dropna(), y_train)
    y_pred = rf2.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    
    # # AUC der Trainingsdaten
    # y_pred = rf2.predict_proba(X_train.dropna())[:,1]
    # fpr, tpr, thresh = roc_curve(y_train,y_pred)
    # roc_auc = auc(fpr,tpr)
    # print('AUC der Trainingsdaten:')
    # print(roc_auc)
    
    return rf2, roc_auc, y_pred, y_test 

#######################################################################################
############################# XGBoost  ################################################
#######################################################################################

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
    
    return result
    
def xgboostHyper(x_train,y_train,X_test,y_test):
    X_train = x_train
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
    print(opt.best_estimator_)
    print(opt.best_score_)
    
    # test data
    xgb2 = opt.best_estimator_.fit(X_train.dropna(), y_train)
    y_pred = xgb2.predict_proba(X_test.dropna())[:,1]
    fpr, tpr, thresh = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr,tpr)
    print(roc_auc)
    
    # # training data
    # y_pred = xgb2.predict_proba(X_train.dropna())[:,1]
    # fpr, tpr, thresh = roc_curve(y_train,y_pred)
    # roc_auc = auc(fpr,tpr)
    # print('AUC der Trainingsdaten:')
    # print(roc_auc)
    
    return xgb2, roc_auc, y_pred, y_test

#######################################################################################
##############################  Dichotomized features  ###############################
#######################################################################################

def dummy_labvalue_oneside(labvalue,data,lower_bound,upper_bound,ttest):
    dummy = []
    for i in range(data[labvalue].shape[0]):
        if lower_bound[labvalue].iloc[i] is not np.nan and (ttest[ttest['Labvalue']==labvalue]['Statistic']<0).iloc[0]:
            dummy.append(data[labvalue].iloc[i] < lower_bound[labvalue].astype(float).iloc[i])
        elif upper_bound[labvalue].iloc[i] is not np.nan and (ttest[ttest['Labvalue']==labvalue]['Statistic']>0).iloc[0]:
            dummy.append(data[labvalue].iloc[i] > upper_bound[labvalue].astype(float).iloc[i])
        else: 
            dummy.append(False)
    dummy = [int(i) for i in dummy]
    return(dummy)

def categorical_predictors_oneside(pred_string,data,lower_bound,upper_bound,ttest):
    subset_na = []
    for i in pred_string:
        if not(lower_bound[i].notna().sum()==0 and upper_bound[i].notna().sum()==0):
            subset_na.append(i)
    print(subset_na)
    temp = data.dropna(subset=subset_na)
    lower_bound_temp = lower_bound[[i in temp.index for i in lower_bound.index]]
    upper_bound_temp = upper_bound[[i in temp.index for i in upper_bound.index]]
    predictors = pd.DataFrame(index=temp.index)
    for i in pred_string:
        if i!='Sex' and i!='Age':
            predictors[i] = dummy_labvalue_oneside(i,temp,lower_bound_temp,upper_bound_temp,ttest)
    mask = np.in1d(pred_string,subset_na,invert=True)
    pred_numeric = np.array(pred_string)[mask]
    # pred_numeric.dtype=str
    # pred_numeric.astype(str)
    pred_numeric=pred_numeric.tolist()
    for i in pred_numeric:
        predictors[i] = temp[i]
    if 'Age' in pred_string:
        predictors['Age'] = (temp['Age']>60).astype(int)
    return(predictors)