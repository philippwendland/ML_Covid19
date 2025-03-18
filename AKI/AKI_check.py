import numpy as np

# Creatinine query with an increase of 0.3 within 48 hours or an increase to 1.5 times the baseline value within 7 days or an absolute value of 4.
lab1=lab
list_numbercase=[]
diag1=diag

removable_patients=[]
    
for i in lab1['Number case'].unique():
    a=lab1[lab1['Number case']==i]
    #adding diagnosis
    d=diag1[diag1['Number case']==i]
    helpvar=False
    greater4=False
    aa=a[a['Lab_name']=='Krea']     
    if aa.empty==False:
            
        aa=aa.sort_values(by='Date',ascending=True)
        aa['Value']=aa['Value'].astype(float)
        for k in range(aa.shape[0]):
            for j in range(k+1,aa.shape[0]):
                time2=aa.iloc[j,aa.columns.get_loc('Date')]
                time1=aa.iloc[k,aa.columns.get_loc('Date')]
                time_diff=time2-time1
                time_diff=time_diff.total_seconds()
                # print(time_diff)
                diff=aa.iloc[j,aa.columns.get_loc('Value')]-aa.iloc[k,aa.columns.get_loc('Value')]
                if (diff>=0.3 and time_diff<=172800) or (aa.iloc[j,aa.columns.get_loc('Value')] >= 1.5*aa.iloc[k,aa.columns.get_loc('Value')] and time_diff<=604800):
                    list_numbercase.append(i)
                if aa.iloc[k,aa.columns.get_loc('Value')] >= 4:
                    
                    if np.abs(diff)<0.3 and sum(['N18' in i for i in d['ICD']])!=0:
                        removable_patients.append(i)
                    
                    list_numbercase.append(i)                    

helpvar1 = ops[ops['Schlüssel']=='8-854.2']['Number case'].unique()
helpvar2 = ops[ops['Schlüssel']=='8-854.3']['Number case'].unique()
removable_patients.extend(list(helpvar1))
removable_patients.extend(list(helpvar2))

list_numbercase = [i for i in list_numbercase if i not in removable_patients]
aki_cases=np.unique(list_numbercase)
removable_cases=np.unique(removable_patients)

lab=lab[[i not in removable_patients for i in lab['Number case']]]
base=base[[i not in removable_patients for i in base['Number case']]]
bew=bew[[i not in removable_patients for i in bew['Number case']]]



