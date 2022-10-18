import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

import pandas as pd
import numpy as np

def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def parser_gtt(data_link,number_columns,lab_complete=False,vit_complete=False):
    #this class implements the parser of the data of the gtt
    #data link is a variable with the link to the data
    #number_columns describes the number of columns
    #lab_complete =True does the complete data preprocessing of the lav-values. Needs a lot of time
    #version7 indicates whether to use the new data version or not
    
    #Dummy columns
    colnames=["A","B","C","D","E","F","G","H","I","J","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","1","2",'3','4','5','6','7','8','9']
    number_columns=number_columns+1
    colnames=colnames[:number_columns]
    
    #maybe setting dtypes manually https://www.roelpeters.be/solved-dtypewarning-columns-have-mixed-types-specify-dtype-option-on-import-or-set-low-memory-in-pandas/
    data = pd.read_csv(data_link, names=colnames,sep=';',header=None, encoding = "ISO-8859-1")
    
    data['A']=data['A'].astype(str)
    data=data.drop_duplicates()
    
    base=data[data['B']=='001PAT']
    if not version9:
        base.columns=['Number case','Datatype','Patientenstamm','Number patient','Birth','Age','Sex','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    else:
        base.columns=['Number case','Datatype','Patientenstamm','Number patient','Birth','Age','Sex','Erfassungsdatum','Datum der letzten Aenderung','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    #datatype and patientenstamm is not interesting
    base=base[['Number case','Number patient','Birth','Age','Sex']]
    base['Number case']=base['Number case'].astype('int')
    base['Number patient']=base['Number patient'].astype('int')
    base['Age'][pd.notnull(base['Age'])]=base['Age'][pd.notnull(base['Age'])].astype('int')
    base['Birth']=pd.to_datetime(base['Birth'],dayfirst=True)
    
    #Remove empty space at the end of sex
    base['Sex']=base['Sex'].astype('string')
    base['Sex'] = [str(i)[0] for i in base['Sex']]
    base['Sex']=base['Sex'].astype('string')
    base=base.drop_duplicates()
    base=base.sort_index().sort_values(by='Number case', kind='mergesort')
    
    diag=data[data['B']=='002ICD']
    
    diag.columns=['Number case','Datatype','Diagnosen','Date','Uhrzeit','ICD_Version','ICD','Klartext','Einweisungsdiagnose','Aufnahmediagnose','Entlassdiagnose','Fachrichtungs-Hauptdiagnose','Krankenhaushauptdiagnose','Nebendiagnose','Erfassungsdatum','Datum der letzten Aenderung','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    diag=diag[['Number case','Date','Uhrzeit','ICD_Version','ICD','Klartext','Einweisungsdiagnose','Aufnahmediagnose','Entlassdiagnose','Fachrichtungs-Hauptdiagnose','Krankenhaushauptdiagnose','Nebendiagnose','Erfassungsdatum','Datum der letzten Aenderung']]
     #Datatype and Diagnosen is not interesting, so we can remove it
    diag['Date']=pd.to_datetime(diag['Date'].values + ' ' + diag['Uhrzeit'].values,dayfirst=True)
    diag=diag.drop('Uhrzeit',1)

    diag['Number case']=diag['Number case'].astype('int')
    diag['ICD_Version']=diag['ICD_Version'].astype('string')
    diag['ICD']=diag['ICD'].astype('string')
    diag['Klartext']=diag['Klartext'].astype('string')
    diag['Einweisungsdiagnose']=diag['Einweisungsdiagnose'].astype('string')
    diag['Aufnahmediagnose']=diag['Aufnahmediagnose'].astype('string')
    diag['Entlassdiagnose']=diag['Entlassdiagnose'].astype('string')
    diag['Fachrichtungs-Hauptdiagnose']=diag['Fachrichtungs-Hauptdiagnose'].astype('string')
    diag['Krankenhaushauptdiagnose']=diag['Krankenhaushauptdiagnose'].astype('string')
    diag['Nebendiagnose']=diag['Nebendiagnose'].astype('string')
    
    if not version8 and not version9 and not version10 and not version11:
        diag['Nebendiagnose'][[' ' in i for i in diag['Nebendiagnose']]]=[i[:i.index(' ')] for i in diag['Nebendiagnose'] if ' ' in i]
        diag['Nebendiagnose']=diag['Nebendiagnose'].astype('string')
    
    else:
        diag['Erfassungsdatum']=pd.to_datetime(diag['Erfassungsdatum'],dayfirst=True)
        diag['Datum der letzten Aenderung']=diag['Datum der letzten Aenderung'].astype('string')
        diag['Datum der letzten Aenderung'][['  ' in i for i in diag['Datum der letzten Aenderung']]]=[i[:i.index('  ')] for i in diag['Datum der letzten Aenderung'] if '  ' in i]
        diag['Datum der letzten Aenderung']=pd.to_datetime(diag['Datum der letzten Aenderung'],dayfirst=True)

    diag=diag.sort_index().sort_values(by='Number case', kind='mergesort')
    diag=diag.drop_duplicates()
    #diag2=diag[['Number case','Date','ICD_Version','ICD','Klartext','Einweisungsdiagnose','Aufnahmediagnose','Entlassdiagnose','Fachrichtungs-Hauptdiagnose','Krankenhaushauptdiagnose','Nebendiagnose']]
    
    dauerdiag=data[data['B']=='003DID']
    
    dauerdiag.columns=['Number case','Datatype','Diagnosen','Date','Uhrzeit','ICD_Version','ICD','Klartext','Einweisungsdiagnose','Aufnahmediagnose','Entlassdiagnose','Fachrichtungs-Hauptdiagnose','Krankenhaushauptdiagnose','Nebendiagnose','Erfassungsdatum','Datum der letzten Aenderung','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    dauerdiag=dauerdiag[['Number case','Date','Uhrzeit','ICD_Version','ICD','Klartext','Einweisungsdiagnose','Aufnahmediagnose','Entlassdiagnose','Fachrichtungs-Hauptdiagnose','Krankenhaushauptdiagnose','Nebendiagnose','Erfassungsdatum','Datum der letzten Aenderung']]
    
    dauerdiag['Date']=pd.to_datetime(dauerdiag['Date'].values + ' ' + dauerdiag['Uhrzeit'].values,dayfirst=True)
    dauerdiag=dauerdiag.drop('Uhrzeit',1)

    dauerdiag['Number case']=dauerdiag['Number case'].astype('int')
    dauerdiag['ICD_Version']=dauerdiag['ICD_Version'].astype('string')
    dauerdiag['ICD']=dauerdiag['ICD'].astype('string')
    dauerdiag['Klartext']=dauerdiag['Klartext'].astype('string')
    dauerdiag['Einweisungsdiagnose']=dauerdiag['Einweisungsdiagnose'].astype('string')
    dauerdiag['Aufnahmediagnose']=dauerdiag['Aufnahmediagnose'].astype('string')
    dauerdiag['Entlassdiagnose']=dauerdiag['Entlassdiagnose'].astype('string')
    dauerdiag['Fachrichtungs-Hauptdiagnose']=dauerdiag['Fachrichtungs-Hauptdiagnose'].astype('string')
    dauerdiag['Krankenhaushauptdiagnose']=dauerdiag['Krankenhaushauptdiagnose'].astype('string')
    dauerdiag['Nebendiagnose']=dauerdiag['Nebendiagnose'].astype('string')

    dauerdiag['Erfassungsdatum']=pd.to_datetime(dauerdiag['Erfassungsdatum'],dayfirst=True)
    dauerdiag['Datum der letzten Aenderung']=dauerdiag['Datum der letzten Aenderung'].astype('string')
    dauerdiag['Datum der letzten Aenderung'][['  ' in str(i) for i in dauerdiag['Datum der letzten Aenderung']]]=[i[:i.index('  ')] for i in dauerdiag['Datum der letzten Aenderung'] if '  ' in i]
    dauerdiag['Datum der letzten Aenderung']=pd.to_datetime(dauerdiag['Datum der letzten Aenderung'],dayfirst=True)

    dauerdiag=dauerdiag.sort_index().sort_values(by='Number case', kind='mergesort')
    dauerdiag=dauerdiag.drop_duplicates()
    
    
    medaid=data[data['B']=='004AID']
    medaid.columns=['Number case','Datatype','Medikation','ATC','Name_med','Dosierung','Katalog','Dauermedikation','Medikation bei Bedarf','Bedarfsmedikation','Vergabeart','Relevant fuer die Entlassung','Datum Vergabe','Startdatum','Enddatum','Status','Medikationskontext','Gabe Vormittags','Gabe Nachmittags','Gabe Abends','Gabe Nachts','nan','nan','nan','nan','nan','nan','nan']
    medaid=medaid[['Number case','ATC','Name_med','Dosierung','Katalog','Dauermedikation','Medikation bei Bedarf','Bedarfsmedikation','Vergabeart','Relevant fuer die Entlassung','Datum Vergabe','Startdatum','Enddatum','Status','Medikationskontext','Gabe Vormittags','Gabe Nachmittags','Gabe Abends','Gabe Nachts']]

    medaid['Ingredient'] = np.zeros((medaid['Name_med'].shape[0],1))
    medaid['Ingredient'][['[' in i for i in medaid['Name_med']]] = [i[i.index('[')+1:-1] for i in medaid['Name_med'][['[' in i for i in medaid['Name_med']]]]
    
    #This is a list comprehension. it takes the i-th values of the Column with the names of the medications and then takes the ingredient out of it
    medaid['Name_med'][['[' in i for i in medaid['Name_med']]] = [i[:i.index('[')] for i in medaid['Name_med'][['[' in i for i in medaid['Name_med']]]]

    #if not version8 and not version9:
    #    a=medaid[['Steigerung' in str(i) for i in medaid['Katalog']]]
    #    a.index[0]
    #    med=med.drop(a.index[0])
    
    # if not version8 and not version9:    
    #     for i in range(3):
            
    #         a=medaid[['2AB' in str(i) for i in medaid['Katalog']]]
    #         a.index[0]
    #         med=med.drop(a.index[0])
        
    # if not version7 and not version8 and not version9:
    
    #     for i in range(3):
            
    #         a=medaid[['7BB27' in str(i) for i in medaid['Katalog']]]
    #         a.index[0]
    #         med=med.drop(a.index[0])
    
    #medaid['Vergabeart']['oral' == medaid['Vergabeart']]='peroral'
    
    medaid['Number case']=medaid['Number case'].astype('int')
    medaid['ATC']=medaid['ATC'].astype('string')
    medaid['Name_med']=medaid['Name_med'].astype('string')
    medaid['Dosierung']=medaid['Dosierung'].astype('string')
    medaid['Katalog']=medaid['Katalog'].astype('string')
    medaid['Dauermedikation']=medaid['Dauermedikation'].astype('string')
    medaid['Medikation bei Bedarf']=medaid['Medikation bei Bedarf'].astype('string')
    medaid['Bedarfsmedikation']=medaid['Bedarfsmedikation'].astype('string')
    medaid['Vergabeart']=medaid['Vergabeart'].astype('string')
    medaid['Relevant fuer die Entlassung']=medaid['Relevant fuer die Entlassung'].astype('string')
    medaid['Datum Vergabe']=pd.to_datetime(medaid['Datum Vergabe'],dayfirst=True,errors='coerce')
    medaid['Startdatum']=pd.to_datetime(medaid['Startdatum'],dayfirst=True)
    medaid['Enddatum'][[str(99) in str(i) for i in medaid["Enddatum"]]]="31.12.2099"
    medaid['Enddatum']=pd.to_datetime(medaid['Enddatum'],dayfirst=True)
    medaid['Status']=medaid['Status'].astype('string')
    medaid['Medikationskontext']=medaid['Medikationskontext'].astype('string')
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='05']='0.5'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='075']='0.75'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='033']='0.33'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='025']='0.25'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='03']='0.3'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='04']='0.4'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='999']='9.99'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='1004']='10.04'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='2375']='23.75'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='501']='50'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='375']='3.75'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='225']='2.25'
    medaid['Gabe Vormittags'][medaid['Gabe Vormittags']=='475']='47.5'
    #medaid[medaid['Gabe Vormittags']][[125 == medaid["Gabe Vormittags"].values[i] and "1,25" in medaid['Dosierung'].values[i] for i in range(medaid.shape[0])]]=='1.25'
    medaid['Gabe Vormittags'] = medaid['Gabe Vormittags'].astype('float')
    medaid['Gabe Nachmittags'] = medaid['Gabe Nachmittags'].astype('float')
    medaid['Gabe Nachmittags'][medaid['Gabe Nachmittags']=='105']='10.5'
    medaid['Gabe Abends'] = medaid['Gabe Abends'].astype('float')
    medaid['Gabe Abends'][medaid['Gabe Abends']=='999']='9.99'
    #medaid[medaid['Gabe Abends']][[125 == medaid["Gabe Abends"].values[i] and "1,25" in medaid['Dosierung'].values[i] for i in range(medaid.shape[0])]]=='1.25'
    medaid['Gabe Abends'][medaid['Gabe Abends']=='225']='22.5'
    medaid['Gabe Abends'][medaid['Gabe Abends']=='1004']='10.04'
    medaid['Gabe Abends'][medaid['Gabe Abends']=='105']='1.5'
    medaid['Gabe Abends'][medaid['Gabe Abends']=='475']='47.5'
    medaid['Gabe Nachts'] = medaid['Gabe Nachts'].astype('float')
    medaid['Gabe Nachts'][medaid['Gabe Nachts']=='375']='3.75'
    medaid['Ingredient']=medaid['Ingredient'].astype('string')
    medaid['Vergabeart'][medaid['Vergabeart']=='oral'] = 'peroral'
    medaid=medaid.sort_index().sort_values(by='Number case', kind='mergesort')
    medaid=medaid.drop_duplicates()
#Values und referenzen haben keinen Typen zugeordnet, da dies Variablenabhängig ist

    lab.columns=['Number case','Datatype','Labor','Date','Lab_name','Lab_name_long','Value','Einheit','lower_ref','upper_ref','Loinc','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    lab=lab[['Number case','Date','Lab_name','Lab_name_long','Value','Einheit','lower_ref','upper_ref','Loinc']]

    lab['Loinc'][lab['Loinc'].notnull()]=[i[:i.index(' ')] for i in lab['Loinc'][lab['Loinc'].notnull()]]
    
    #einheitliche Labels für booleans
    lab['Value'][lab['Value']=='neg']='negativ'
    lab['Value'][lab['Value']=='neg.']='negativ'
    lab['Value']['pos' == lab['Value']]='positiv'
    lab.values[['!' in str(i) for i in lab['Value']],4]=np.nan
    lab['Value']['NEG' == lab['Value']]='negativ'
    lab['Value']['POST' == lab['Value']]='positiv'
    lab['Value']['POS.' == lab['Value']]='positiv'
    lab['Value']['POS' == lab['Value']]='positiv'
    #lab['Value']['+' == lab['Value']]='positiv'
    lab['Value']['ni.ber.' == lab['Value']]=np.nan
    lab.values[['>' in str(i) for i in lab['lower_ref']],7]='no upper ref'
    lab.values[['>' in str(i) for i in lab['lower_ref']],6]=[str(i[1:]) for i in lab['lower_ref'] if '>' in str(i)]
    #lab.values[['<' in str(i) for i in lab['upper_ref']],6]='no lower ref'
    lab.values[['<' in str(i) for i in lab['upper_ref']],6]=0
    lab.values[['<' in str(i) for i in lab['upper_ref']],7]=[str(i[1:]) for i in lab['upper_ref'] if '<' in str(i)]
    lab.values[[not i for i in lab['upper_ref']],7]=np.nan
        
    #print(datetime.now())
    
    if lab_complete==True:
        
        #remove < and > an erster Stelle
        lab.values[['<=' == str(lab['Value'].values[i])[:2] or '>=' == str(lab['Value'].values[i])[:2] for i in range(lab.shape[0])],4] = [str(i)[2:] for i in lab['Value'].values[['<=' == str(lab['Value'].values[i])[:2] or '>=' == str(lab['Value'].values[i])[:2] for i in range(lab.shape[0])]]]
        lab.values[['<' == str(lab['Value'].values[i])[0] or '>' == str(lab['Value'].values[i])[0] for i in range(lab.shape[0])],4] = [str(i)[1:] for i in lab['Value'].values[['<' == str(lab['Value'].values[i])[0] or '>' == str(lab['Value'].values[i])[0] for i in range(lab.shape[0])]]]
        
        # for i in lab['Lab_name'].unique():
        #     #take all Einheiten for every Variable and test whether there exist differences between them
        #     if pd.notna(data['Einheit'][data[group]==i].unique()[0]) and sum(pd.notna(data['Value'][data[group]==i]))>0:# and 'BZ' not in str(i):
            
        #     if sum(pd.notna(data['Value'][data[group]==i]))>0 and i in set(['INR','LDLHDL','pH','P-Sp.G','SpeGew','TFS','U-E/Kr','UpH','U-pH','USpG','Ct-Wer'])

        
        #Zusammenfassung mehrerer Labels
        lab.values[['Erythrozyten (+)' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='L-Ery' for i in range(lab['Value'].shape[0])],4] = '(+)'
        lab.values[['Erythrozyten +' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='L-Ery' for i in range(lab['Value'].shape[0])],4] = '+'
        lab.values[['0' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='L-Ery' for i in range(lab['Value'].shape[0])],4] = 'negativ'
        lab.values[['sehr vereinz' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='L-Ery' for i in range(lab['Value'].shape[0])],4] = 'negativ'
        lab.values[['keine' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='L-Ery' for i in range(lab['Value'].shape[0])],4] = 'negativ'
        lab.values[['Wärmeautoantiköper' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='akid' for i in range(lab['Value'].shape[0])],4] = 'Wärmeautoantikörper'
        lab.values[['nichtfadenziehend' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Visk' for i in range(lab['Value'].shape[0])],4] = 'nicht fadenziehend'
        #lab.values[['anal' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='SonLok' for i in range(lab['Value'].shape[0])],4] = 'Anal'
        lab['Value']['anal' == lab['Value']]='Anal'
        #lab.values[['Anus' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='SonLok' for i in range(lab['Value'].shape[0])],4] = 'Anal'
        lab['Value']['Anus' == lab['Value']]='Anal'
        lab['Value']['Analabstrich' == lab['Value']]='Anal'
        #lab.values[['Rectal' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='SonLok' for i in range(lab['Value'].shape[0])],4] = 'Rektal'
        lab['Value']['Rectal' == lab['Value']]='Rektal'
        lab['Value']['rectal' == lab['Value']]='Rektal'
        #lab.values[['Analabstrich' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='SonLok' for i in range(lab['Value'].shape[0])],4] = 'Anal'
        #lab.values[['re Unerschenkel' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='Wu Lok' for i in range(lab['Value'].shape[0])],4] = 'Unterschenkel rechts'
        lab['Value']['re Unerschenkel' == lab['Value']]='Unterschenkel rechts'
        #lab.values[['Unterschenkel li' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='Wu Lok' for i in range(lab['Value'].shape[0])],4] = 'Unterschenkel links'
        lab['Value']['Unterschenkel li' == lab['Value']]='Unterschenkel links'
        #lab.values[['Linker Unterschenkel' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='Wu Lok' for i in range(lab['Value'].shape[0])],4] = 'Unterschenkel links'
        lab['Value']['Linker Unterschenkel' == lab['Value']]='Unterschenkel links'
        #lab.values[['li U Schenkel' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='Wu Lok' for i in range(lab['Value'].shape[0])],4] = 'Unterschenkel links'
        lab['Value']['li U Schenkel' == lab['Value']]='Unterschenkel links'
        #lab.values[['li US' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='Wu Lok' for i in range(lab['Value'].shape[0])],4] = 'Unterschenkel links'
        lab['Value']['li US' == lab['Value']]='Unterschenkel links'
        lab['Value']['US Stumf li' == lab['Value']]='Unterschenkel Stumpf links'
        lab['Value']['US' == lab['Value']]='Unterschenkel'
        lab['Value']['li. Unterschenkel' == lab['Value']]='Unterschenkel links'
        lab['Value']['li US unten' == lab['Value']]='Unterschenkel links unten'
        lab['Value']['li US oben' == lab['Value']]='Unterschenkel links oben'
        #lab.values[['Geßäs' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='Wu Lok' for i in range(lab['Value'].shape[0])],4] = 'Gesäß'
        lab['Value']['Geßäs' == lab['Value']]='Gesäß'
        #lab.values[['re. Fuss' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='Wu Lok' for i in range(lab['Value'].shape[0])],4] = 're.Fuß'
        lab['Value']['re. Fuss' == lab['Value']]='Fuß rechts'
        lab['Value']['Re. Fuß' == lab['Value']]='Fuß rechts'
        lab['Value']['li.Fuss' == lab['Value']]='Fuß links'
        lab['Value']['li Auge' == lab['Value']]='Auge links'
        lab.values[['Fersen  re' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='Wu Lok' for i in range(lab['Value'].shape[0])],4] = 're Ferse'
        lab['Value']['Fersen  re' == lab['Value']]='re Ferse'
        lab['Value']['unterschenkel' == lab['Value']]='Unterschenkel'
        lab['Value']['re. Unterschenkel/ amputation' == lab['Value']]='Unterschenkel rechts, Amputation'
        lab['Value']['zehn' == lab['Value']]='Zehen'
        lab['Value']['re.Unterschenkel' == lab['Value']]='Unterschenkel rechts'
        lab['Value']['re. US' == lab['Value']]='Unterschenkel rechts'
        lab['Value']['steiß' == lab['Value']]='Steiß'
        lab['Value']['Trochenter re' == lab['Value']]='Trochanter rechts'
        lab['Value']['Trochanter re' == lab['Value']]='Trochanter rechts'
        lab['Value']['rechter Unterschenkel hinten' == lab['Value']]='Unterschenkel rechts hinten'
        lab['Value']['li UA' == lab['Value']]='li. Unterarm'
        lab['Value']['re US' == lab['Value']]='Unterschenkel rechts'
        lab['Value']['US re' == lab['Value']]='Unterschenkel rechts'
        lab['Value']['Bein re.' == lab['Value']]='Bein rechts'
        lab['Value']['Bein li' == lab['Value']]='Bein links'
        lab['Value']['bein links' == lab['Value']]='Bein links'
        lab['Value']['Vorfuss links' == lab['Value']]='Vorfuß links'
        lab['Value']['re Vorfuß' == lab['Value']]='Vorfuß rechts'
        lab['Value']['re. Großzeh' == lab['Value']]='Großzeh rechts'
        lab['Value']['rechte Fußsohle' == lab['Value']]='Fußsohle rechts'
        lab['Value']['re Fußsohle' == lab['Value']]='Fußsohle rechts'
        lab['Value']['U-Schenkel rechts' == lab['Value']]='Unterschenkel rechts'
        lab['Value']['re Unterschenkel' == lab['Value']]='Unterschenkel rechts'
        lab['Value']['li. US' == lab['Value']]='Unterschenkel links'
        lab['Value']['li Ferse' == lab['Value']]='Ferse links'
        lab['Value']['Li.Ferße' == lab['Value']]='Ferse links'
        lab['Value']['li Bein' == lab['Value']]='Bein links'
        lab['Value']['Schienbein re' == lab['Value']]='Schienbein rechts'
        lab['Value']['li. Trochanter' == lab['Value']]='Trochanter links'
        lab['Value']['linke Wade' == lab['Value']]='Wade links'
        lab['Value']['rechte Wade' == lab['Value']]='Wade rechts'
        lab['Value']['re. Unterschenkel' == lab['Value']]='Unterschenkel rechts'
        lab['Value']['Unterschenkel rechts' == lab['Value']]='Unterschenkel rechts'
        lab['Value']['li Ohr' == lab['Value']]='Ohr links'
        lab['Value']['Kleinzeh li' == lab['Value']]='Kleinzeh links'
        lab['Value']['rechte Leiste' == lab['Value']]='Leiste rechts'
        lab['Value']['re Leiste' == lab['Value']]='Leiste rechts'
        lab['Value']['li. Leiste' == lab['Value']]='Leiste links'
        lab['Value']['li Außenknöchel' == lab['Value']]='Außenknöchel links'
        lab['Value']['li. Ballen außen' == lab['Value']]='Ballen links außen'
        lab['Value']['li. Arm' == lab['Value']]='Arm links'
        lab['Value']['re.Fuß' == lab['Value']]='Fuß rechts'
        lab['Value']['re Ferse' == lab['Value']]='Ferse rechts'
        lab['Value']['linker Fuß' == lab['Value']]='Fuß links'
        lab['Value']['li Fuß' == lab['Value']]='Fuß links'
        lab['Value']['Zehen re' == lab['Value']]='Zehen rechts'
        lab['Value']['Trochanter re' == lab['Value']]='Trochanter rechts'
        lab['Value']['li. Schienbein' == lab['Value']]='Schienbein links'
        lab['Value']['li. Unterarm' == lab['Value']]='Unterarm links'
        lab['Value']['Stumpf re.' == lab['Value']]='Stumpf rechts'
        lab['Value']['Li. US' == lab['Value']]='Unterschenkel links'
        lab['Value']['li Arm' == lab['Value']]='Arm links'
        lab['Value']['re Arm' == lab['Value']]='Arm rechts'
        lab['Value']['re. Fuß' == lab['Value']]='Fuß rechts'
        lab['Value']['Vorfuß re' == lab['Value']]='Vorfuß rechts'
        lab['Value']['Vorfuss re' == lab['Value']]='Vorfuß rechts'
        lab['Value']['Vorfuss Links' == lab['Value']]='Vorfuß links'
        lab['Value']['Fuß re.' == lab['Value']]='Fuß rechts'
        lab['Value']['fuß' == lab['Value']]='Fuß'
        lab['Value']['Zehe' == lab['Value']]='Zeh'
        lab['Value']['rechte Flanke' == lab['Value']]='Flanke rechts'
        
        #lab.values[['Erythrozyten' in str(lab['Value'].values[i]) or 'Erythrocyten' in str(lab['Value'].values[i]) or 'Erys' in str(lab['Value'].values[i]) or 'Erythroccyten' in str(lab['Value'].values[i]) or 'Ery.' in str(lab['Value'].values[i]) or 'Ery:' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4]
        lab.values[['Erythrozyten ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Erythrozyten ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Erythrozyten ' in str(lab['Value'].values[i])]
        lab.values[['Erythozyten ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Erythozyten ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Erythozyten ' in str(lab['Value'].values[i])]
        lab.values[['Erythrozyten: ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Erythrozyten: ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Erythrozyten: ' in str(lab['Value'].values[i])]
        lab.values[['Erythrozyten' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Erythrozyten', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Erythrozyten' in str(lab['Value'].values[i])]
        lab.values[['Erythroccyten: ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Erythroccyten: ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Erythroccyten: ' in str(lab['Value'].values[i])]
        lab.values[['Erythrocyten ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Erythrocyten ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Erythrocyten ' in str(lab['Value'].values[i])]
        lab.values[['Ery.: ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Ery.: ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Ery.: ' in str(lab['Value'].values[i])]
        lab.values[['Ery: ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Ery: ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Ery: ' in str(lab['Value'].values[i])]
        lab.values[['Ery:' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Ery:', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Ery:' in str(lab['Value'].values[i])]
        lab.values[['Ery. ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Ery. ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Ery. ' in str(lab['Value'].values[i])]
        lab.values[['Ery.' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Ery.', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Ery.' in str(lab['Value'].values[i])]
        lab.values[['Erys ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Erys ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Erys ' in str(lab['Value'].values[i])]
        lab.values[['Ery\'s ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Ery\'s ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Ery\'s ' in str(lab['Value'].values[i])]
        lab.values[['Erys' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Erys', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Erys' in str(lab['Value'].values[i])]
        lab.values[['Ery ' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Ery ', 'Ery') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Ery ' in str(lab['Value'].values[i])]
        #lab.values[['Ery(+),' == str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],4] = [str(lab['Value'].values[i]).replace('Ery(+),', 'Ery(+)') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Bem1' and 'Ery(+),' == str(lab['Value'].values[i])]    
        lab['Value']['Ery(+),' == lab['Value']]='Ery(+)'
        lab['Lab_name']['U-Bili' == lab['Lab_name']] = 'UBil'
        lab['Lab_name_long']['Urinbilirubin'==lab['Lab_name_long']] = 'Urin Bilirubin'
        lab['Lab_name']['UEryHb' == lab['Lab_name']] = 'UEry'
        lab['Lab_name']['U-Glu' == lab['Lab_name']] = 'UGlu'
        lab['Lab_name']['U-Nitr' == lab['Lab_name']] = 'UNitr'
        lab['Lab_name']['UStLeu' == lab['Lab_name']] = 'ULeu'
        lab['Lab_name']['U-Ubg' == lab['Lab_name']] = 'UUbg'
        lab['Lab_name']['-Glu' == lab['Lab_name']] = 'Gluc'
        lab['Lab_name']['-O2Sät' == lab['Lab_name']] = 'O2-Sät'
        lab.values[[str(i)[0] =='-' for i in lab['Lab_name']],2] = [str(i[1:]) for i in lab['Lab_name'] if str(i)[0]=='-']
        lab.values[[str(i)[0] =='-' for i in lab['Lab_name_long']],3] = [str(i[1:]) for i in lab['Lab_name_long'] if str(i)[0]=='-']
        lab.values[[str(i)[0:2] =='ct' for i in lab['Lab_name_long']],3] = [str(i[2:]) for i in lab['Lab_name_long'] if str(i)[0:2]=='ct']  
        
        #lab['Value'].loc['Elliptozyten.Teardropzellen' == lab['Value']] = 'Elliptozyten, Teardropzellen'
        a=lab[['Elliptozyten.Teardropzellen' in str(i) for i in lab['Value']]]
        if len(a)>0:
            a.index[0]
            lab=lab.drop(a.index[0])
            a['Value']= 'Elliptozyten, Teardropzellen'
            lab=lab.append(a)
        lab['Value']['vereinzelt Fragmentozyten/Geldrollenbildung' == lab['Value']] = 'vereinzelt Fragmentozyten, Geldrollenbildung'
        lab['Value']['Fragmentozyten Geldrollenbildung' == lab['Value']] = 'Fragmentozyten, Geldrollenbildung'
        lab['Value']['Leukozytentrauben Zellreste' == lab['Value']] = 'Leukozytentrauben, Zellreste'
        lab['Value']['V.a. Lymphom?hämatologische Abklärung empfohlen' == lab['Value']] = 'Lymphom, hämatolog.Abklärung'
        lab['Value']['Teardroperythrozyten. Geldrollen' == lab['Value']] = 'Teardroperythrozyten, Geldrollen'
        lab['Value']['Riesenthrombozyten. Thrombozytenaggregate' == lab['Value']] = 'Riesenthrombozyten, Thrombozytenaggregate'
        temp = lab[:][lab['Einheit']=='Elliptozyten,Teardropzellen,Fragmentozyten']
        lab = lab.append(temp, ignore_index=True)
        lab['Einheit'][np.sort(lab.index)[-1]] = np.nan
        lab['lower_ref'][np.sort(lab.index)[-1]] = np.nan
        lab['upper_ref'][np.sort(lab.index)[-1]] = np.nan
        lab['Loinc'][np.sort(lab.index)[-1]] = ''
        lab['Value'][np.sort(lab.index)[-1]] = 'Elliptozyten,Teardropzellen,Fragmentozyten'     
        lab.values[['Elliptozyten,Teardropzellen,Fragmentozyten' in str(lab['Einheit'].values[i]) and lab['Lab_name_long'].values[i]=='BB-Bemerkungen' for i in range(lab['Einheit'].shape[0])],6]=np.nan
        lab.values[['Elliptozyten,Teardropzellen,Fragmentozyten' in str(lab['Einheit'].values[i]) and lab['Lab_name_long'].values[i]=='BB-Bemerkungen' for i in range(lab['Einheit'].shape[0])],7]=np.nan
        lab.values[['Elliptozyten,Teardropzellen,Fragmentozyten' in str(lab['Einheit'].values[i]) and lab['Lab_name_long'].values[i]=='BB-Bemerkungen' for i in range(lab['Einheit'].shape[0])],8]=''
        lab.values[['Elliptozyten,Teardropzellen,Fragmentozyten' in str(lab['Einheit'].values[i]) and lab['Lab_name_long'].values[i]=='BB-Bemerkungen' for i in range(lab['Einheit'].shape[0])],5]=np.nan   
        lab['Value']['Leukotrauben Zellreste' == lab['Value']]='Leukozytentrauben, Zellreste'
       
        
        #Make multiple einträge for multiple values of Bemerkungen
        for j in range(3):
            lab_append=pd.DataFrame(columns=lab.columns)
            for i in range(lab['Value'].shape[0]):
                if ',' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3' or lab['Lab_name'].values[i]=='P-Bem1' or lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3' or lab['Lab_name'].values[i]=='akid'):
                    app=lab.values[i]
                    val=app[4]
                    newval = str(val)[val.index(',')+1:]
                    if newval:
                        if newval[0]==' ':
                            app[4]=newval[1:]
                        else:
                            app[4]=newval
                        lab_append.loc[len(lab_append)]=app
                    
                    if str(val)!=',' and not str(val)[:val.index(',')]:
                        lab.values[i,4]=str(val)[val.index(' ')+1:]
                    elif str(val)!=',' and str(val)[:val.index(',')][-1]!=' ':
                        lab.values[i,4]=str(val)[:val.index(',')]
                    else:
                        lab.values[i,4]=str(val)[:val.index(',')-1]
                if 'und' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3'):
                    app=lab.values[i]
                    val=app[4]
                    newval = str(val)[val.index('und')+3:]
                    if newval[0]==' ':
                        app[4]=newval[1:]
                    else:
                        app[4]=newval
                    lab_append.loc[len(lab_append)]=app
                    if str(val)[:val.index('und')][-1]!=' ':
                        lab.values[i,4]=str(val)[:val.index('und')]
                    else:
                        lab.values[i,4]=str(val)[:val.index('und')-1]
            #Verhindere Multipler Index
                if '+' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') and '+'!= str(lab['Value'].values[i]):
                    app=lab.values[i]
                    val=app[4]
                    newval = str(val)[val.index('+')+1:]
                    if newval:
                        if newval[0]==' ':
                            app[4]=newval[1:]
                        elif newval[-1] == ' ':
                            app[4]=newval[:-2]
                        else:
                            app[4]=newval
                        lab_append.loc[len(lab_append)]=app
                    if str(val)[:val.index('+')][-1]!=' ':
                        lab.values[i,4]=str(val)[:val.index('+')]
                    else:
                        lab.values[i,4]=str(val)[:val.index('+')-1]
            lab=pd.concat((lab,lab_append),axis=0,ignore_index=True)
            
        lab.values[['ery' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],2]='P-Ery_PBem'
        lab.values[['Ery' in str(lab['Value'].values[i]) and lab['Lab_name'].values[i]=='P-Bem1' for i in range(lab['Value'].shape[0])],2]='P-Ery_PBem'
        lab['Value']['P-Ery_PBem' == lab['Lab_name']] = [str(lab['Value'].values[i])[3:] for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='P-Ery_PBem']
        lab.values[['+++' in str(lab['Einheit'].values[i]) and lab['Lab_name'].values[i]=='P-Ery_PBem' for i in range(lab['Einheit'].shape[0])],4]='+++'
        lab.values[['+++' in str(lab['Einheit'].values[i]) and lab['Lab_name'].values[i]=='P-Ery_PBem' for i in range(lab['Einheit'].shape[0])],6]=np.nan
        lab.values[['+++' in str(lab['Einheit'].values[i]) and lab['Lab_name'].values[i]=='P-Ery_PBem' for i in range(lab['Einheit'].shape[0])],7]=np.nan
        lab.values[['+++' in str(lab['Einheit'].values[i]) and lab['Lab_name'].values[i]=='P-Ery_PBem' for i in range(lab['Einheit'].shape[0])],8]=''
        lab.values[['+++' in str(lab['Einheit'].values[i]) and lab['Lab_name'].values[i]=='P-Ery_PBem' for i in range(lab['Einheit'].shape[0])],5]=np.nan   
        lab.values[[str(lab['Lab_name'].values[i]) == 'P-Ery_PBem' and str(lab['Value'].values[i]) == "' +++" for i in range(lab['Lab_name'].shape[0])],4] = '+++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'P-Ery_PBem' and str(lab['Value'].values[i]) == ':++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'P-Ery_PBem' and str(lab['Value'].values[i]) == 'erythrozyten: (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        
        #setting 'negativ' in numerical variables to zero
        temp = lab[:][lab['Lab_name']=='M.Albu']
        temp['Value'][temp['Value']=='negativ']='0'
        lab['Value'][temp.index] = temp['Value']
        
        #setting 'massenh' in numerical variables to 1000
        temp = lab[:][lab['Lab_name']=='U-Ery']
        temp['Value'][temp['Value']=='massenh']='1000'
        lab['Value'][temp.index] = temp['Value']
        
        lab['Value']['trüb,blutig'==lab['Value']]='blutig, trüb'
        lab['Value']['rötlichgelb'==lab['Value']]='rötlich gelb' 
    
        lab['Value']['Thromboaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozyteaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombotytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrmbozytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['THrombozytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Trombozytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozytemaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombzytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Theombozytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozytenaggreate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thromobzytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozxtenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozytenagreggate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozytnaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thromozytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombpzytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozyytenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozytebaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozytenaggregat' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombozytenagggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['Thrombocytenaggregate' == lab['Value']]='Thrombozytenaggregate'     
        lab['Value']['Thrombozyzenaggregate' == lab['Value']]='Thrombozytenaggregate'
        lab['Value']['vereinzelt thrombozytenaggregate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab['Value']['vereinzelz Thrombocytenaggregate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab['Value']['vereinzelt Trhombozytenaggregate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab['Value']['ver.Thromboagglutinate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab['Value']['ver. Thromboaggregate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab['Value']['vereinz. Thromvozytenaggregate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab['Value']['vereinzelt Thromboaggregate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab['Value']['vereinzelt Thrombotytenaggregate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab['Value']['vereinz. Thromvbozytenaggregate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab['Value']['vereinz. Thrombozytenaggregate' == lab['Value']]='vereinzelt Thrombozytenaggregate'
        lab.values[['Thrombozytenaggregate' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2] ='Thrombozytenaggregate_BBem'
        lab['Value'][lab['Value'] == 'Thrombozytenaggregate'] = 'positiv'
        lab['Value'][lab['Value'] == 'vereinzelt Thrombozytenaggregate']= 'vereinzelt'
        
        lab.values[['Ery' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Ery_BBem'
        lab['Value'][lab['Value'] == 'vereinz. pincered Erys'] = 'vereinzelt'
        
        #lab.values[[i == 'Thrombozytenaggregate_BBem' for i in lab['Lab_name']],4] = [str(lab['Value'].values[i]).replace('Thrombozytenaggregate','') for i in range(lab['Value'].shape[0]) if lab['Lab_name'].values[i]=='Thrombozytenaggregate_BBem']
        
        lab.values[['Thrombozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Thrombozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Thrombozytenaggregate_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Thrombozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        
        lab['Value']['ver. Mikrozyten' == lab['Value']]='vereinzelt Mikrozyten'
        lab['Value']['vereinzelt mikrozyten' == lab['Value']]='vereinzelt Mikrozyten'
        lab['Value']['mikrozyten' == lab['Value']]='Mikrozyten'
        lab['Value']['Miktozyten' == lab['Value']]='Mikrozyten'
        lab.values[['Mikrozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Mikrozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Mikrozyten_BBem' and str(lab['Value'].values[i]) == 'Mikrozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Mikrozyten_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Mikrozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab.values[['Megalozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Megalozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Megalozyten_BBem' and str(lab['Value'].values[i]) == 'Megalozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Megalozyten_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Megalozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab['Value']['Riesenthro' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesenthrombos' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesnthrombozyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesenthromozyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesentrombozyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesenthromobozyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesenthromozyte' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesenthrombozythen' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Risenthrombozyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesenthzombozyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesentzhombozyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['z.T.Riesenthrombozyten' == lab['Value']]='z.T. Riesenthrombozyten'
        lab['Value']['z.T. Riesenthromboz.' == lab['Value']]='z.T. Riesenthrombozyten'
        lab['Value']['z.T.Riesenthrombocyten' == lab['Value']]='z.T. Riesenthrombozyten'
        lab['Value']['ver. Riesenthrombos' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombozythen' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombozythen' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthtombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombozyen' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzeltRiesenthrombos' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzeltRiesenthrombos' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombozythen' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombozythen' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['Vereinzelt Riesenthrombocyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombocyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombozythen' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombozytm' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthro mbozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombos' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthromobozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesethrombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesemtrhombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt riesenthrombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt riesenthrombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Rieenthrombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesentrombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinz.Riesenthrombos' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzeltRiesenthrombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinz.. Riesenthrombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthromozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinzelt Riesenthrombozten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinz. Riesenthro' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['verinz. Riesenthro' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['ver.Riesenthrombos' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['vereinz. Riesenthrombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['Riesemthrombozyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesenthrombocyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesenthrombzyten' == lab['Value']]='Riesenthrombozyten' 
        lab['Value']['riesenthrombozyten' == lab['Value']]='Riesenthrombozyten' 
        lab['Value']['Riesenthromobzyten' == lab['Value']]='Riesenthrombozyten' 
        lab['Value']['Riesenthomobzyten' == lab['Value']]='Riesenthrombozyten' 
        lab['Value']['Riesenthrombozyteb' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesentzrombozyten' == lab['Value']]='Riesenthrombozyten' 
        lab['Value']['Riesenthromobzyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['Riesethrombozyten' == lab['Value']]='Riesenthrombozyten'
        lab['Value']['einige Riesenthrombos' == lab['Value']]='einige Riesenthrombozyten'
        lab['Value']['einige Riesentrombozyten' == lab['Value']]='einige Riesenthrombozyten'
        lab['Value']['Riesenthrombozyten vereinzelt' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab['Value']['Vereinzelt Riesenthrombozyten' == lab['Value']]='vereinzelt Riesenthrombozyten'
        lab.values[['Riesenthrombozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Riesenthrombozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Riesenthrombozyten_BBem' and str(lab['Value'].values[i]) == 'Riesenthrombozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Riesenthrombozyten_BBem' and str(lab['Value'].values[i]) == 'z.T. Riesenthrombozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'zum Teil'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Riesenthrombozyten_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Riesenthrombozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Riesenthrombozyten_BBem' and str(lab['Value'].values[i]) == 'einige Riesenthrombozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'einige'
        
        lab['Value']['Fragmantozyten' == lab['Value']]='Fragmentozyten'
        lab['Value']['Fragmentpzyten' == lab['Value']]='Fragmentozyten'
        lab['Value']['Fragmentozytern' == lab['Value']]='Fragmentozyten'
        lab['Value']['Fragmentozythen' == lab['Value']]='Fragmentozyten'
        lab['Value']['Fragmentozyen' == lab['Value']]='Fragmentozyten'
        lab['Value']['Fragmetozyten' == lab['Value']]='Fragmentozyten'
        lab['Value']['Fragmeozyten' == lab['Value']]='Fragmentozyten'
        lab['Value']['Fragmentocyten' == lab['Value']]='Fragmentozyten'
        lab['Value']['Fragmentotyten' == lab['Value']]='Fragmentozyten'
        lab['Value']['vereinz.Fragmentocyten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinzelt Fragmenzozyten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinzelt Fragmentocyten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinz. Fragmentoyzten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinzelt Fragmerntozyten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinzelt Fragementozyten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinz. Fragmentozyten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinz. Fragmentocyten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinz. Framentozyten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinzelt Fragmento' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinz.Fragmentozytenb' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['ver.Fragmentozyten' == lab['Value']]='vereinzelt Fragmentozyten'
        lab['Value']['vereinz. Fragmentozytzen' == lab['Value']]='vereinzelt Fragmentozyten'       
        lab['Value']['g.ver. Fragmentozyten' == lab['Value']]='ganz vereinzelt Fragmentozyten'
        lab['Value']['g.ver. Fragmentocyten' == lab['Value']]='ganz vereinzelt Fragmentozyten'
        lab.values[['Fragmentozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Fragmentozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Fragmentozyten_BBem' and str(lab['Value'].values[i]) == 'Fragmentozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Fragmentozyten_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Fragmentozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Fragmentozyten_BBem' and str(lab['Value'].values[i]) == 'einige Fragmentozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'einige'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Fragmentozyten_BBem' and str(lab['Value'].values[i]) == 'ganz vereinzelt Fragmentozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'ganz vereinzelt'
        
        lab['Value']['Geldrollenbidung' == lab['Value']]='Geldrollenbildung'
        lab['Value']['Geldollenbildung' == lab['Value']]='Geldrollenbildung'
        lab['Value']['Geldrollrnbildung' == lab['Value']]='Geldrollenbildung'
        lab['Value']['Geldrolle' == lab['Value']]='Geldrollen'
        lab['Value']['Gedldrollen' == lab['Value']]='Geldrollen'
        lab['Value']['Gedlrollen' == lab['Value']]='Geldrollen'
        lab['Value']['Geldrollnen' == lab['Value']]='Geldrollen'
        lab['Value']['leicht Geldrollenbildung' == lab['Value']]='leichte Geldrollenbildung'
        lab['Value']['leichte Gelrollenbildung' == lab['Value']]='leichte Geldrollenbildung'
        lab['Value']['leichte Geldrollenbbildung' == lab['Value']]='leichte Geldrollenbildung'
        lab['Value']['vereinz. Geldrollen' == lab['Value']]='vereinzelt Geldrollen'
        lab['Value']['z.T.Geldrollenbildung' == lab['Value']]='vereinzelt Geldrollenbildung'
        lab.values[['Geldrolle' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Geldrolle_BBem'
        
        lab['Value']['vereinzelt Akantozythen' == lab['Value']]='vereinzelt Akanthozyten'
        lab['Value']['vereinz. Akanthozyten' == lab['Value']]='vereinzelt Akanthozyten'
        lab['Value']['vereinzelt akanthozyten' == lab['Value']]='vereinzelt Akanthozyten'
        lab['Value']['vereinzelt Akantozyten' == lab['Value']]='vereinzelt Akanthozyten'
        lab['Value']['vereinzelt akantozyten' == lab['Value']]='vereinzelt Akanthozyten'
        lab['Value']['-Akanto' == lab['Value']]='Akanthozyten'
        lab['Value']['Akanthocyten' == lab['Value']]='Akanthozyten'
        lab['Value']['Akanthozythen' == lab['Value']]='Akanthozyten'
        lab['Value']['Akantzythen' == lab['Value']]='Akanthozyten'
        lab['Value']['Akantozythen' == lab['Value']]='Akanthozyten'
        lab['Value']['Akhantozyten' == lab['Value']]='Akanthozyten'
        lab['Value']['Akantozythen' == lab['Value']]='Akanthozyten'
        lab['Value']['Akanthocythen' == lab['Value']]='Akanthozyten'
        lab['Value']['Akantozyten' == lab['Value']]='Akanthozyten'
        lab['Value']['Akanthozytnen' == lab['Value']]='Akanthozyten'
        lab['Value']['Akanthozyteb' == lab['Value']]='Akanthozyten'
        lab.values[['Akanthozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Akanthozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Akanthozyten_BBem' and str(lab['Value'].values[i]) == 'Akanthozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Akanthozyten_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Akanthozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab.values[['Stamatozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Stamatozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Stamatozyten_BBem' and str(lab['Value'].values[i]) == 'Stamatozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab['Value']['-Elliptozyten' == lab['Value']]='Elliptozyten'
        lab['Value']['Ellitozyten' == lab['Value']]='Elliptozyten'
        lab['Value']['elliptozyten' == lab['Value']]='Elliptozyten'
        lab['Value']['Elloptozyten' == lab['Value']]='Elliptozyten'
        lab['Value']['Elliptozyzten' == lab['Value']]='Elliptozyten'
        lab['Value']['Ellipthozyten' == lab['Value']]='Elliptozyten'
        lab['Value']['Eliptozyten' == lab['Value']]='Elliptozyten'
        lab['Value']['vereinz. Elliptozyten' == lab['Value']]='vereinzelt Elliptozyten'
        lab['Value']['vereinzelt Elliptocyten' == lab['Value']]='vereinzelt Elliptozyten'
        lab['Value']['vereinzelt elliptozyten' == lab['Value']]='vereinzelt Elliptozyten'
        lab.values[['Elliptozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Elliptozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Elliptozyten_BBem' and str(lab['Value'].values[i]) == 'Elliptozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Elliptozyten_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Elliptozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab['Value']['ver. Ovalozyten' == lab['Value']]='vereinzelt Ovalozyten'
        lab['Value']['vereinzelt Ovalozyen' == lab['Value']]='vereinzelt Ovalozyten'
        lab.values[['Ovalozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Ovalozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Ovalozyten_BBem' and str(lab['Value'].values[i]) == 'Ovalozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Ovalozyten_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Ovalozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab['Value']['Teardroperythozyten' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['teardroperythrozyten' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Teardroperythrozyt' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Terdroperythrozyten' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Teardroperthrozyten' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Teardropzellen' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Tear drops' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Teartrops' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Teardropformen' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['tear drops' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Tear Drops' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Tear drop' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Tear Trops' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Teardrops' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Teardroperythrozyten.' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['Tear-Drops' == lab['Value']]='Teardroperythrozyten'
        lab['Value']['vereinzelt Teardropzellen' == lab['Value']]='vereinzelt Teardroperythrozyten'
        lab['Value']['vereinzelt Tear-Drops' == lab['Value']]='vereinzelt Teardroperythrozyten'
        lab['Value']['vereinz. tear drops' == lab['Value']]='vereinzelt Teardroperythrozyten'
        lab['Value']['vereinzelt Taerdroperythrozyten' == lab['Value']]='vereinzelt Teardroperythrozyten'
        lab['Value']['vereinz. Teardrops.' == lab['Value']]='vereinzelt Teardroperythrozyten'    
        lab['Value']['verein. Teardrops' == lab['Value']]='vereinzelt Teardroperythrozyten'    
        lab['Value']['vereinz. tear drops' == lab['Value']]='vereinzelt Teardroperythrozyten' 
        lab['Value']['vereinz. teardrops' == lab['Value']]='vereinzelt Teardroperythrozyten' 
        lab['Value']['vereinz.tear drops' == lab['Value']]='vereinzelt Teardroperythrozyten'    
        lab['Value']['vereinzelt teardroperythrozyten' == lab['Value']]='vereinzelt Teardroperythrozyten'    
        lab['Value']['vereinz. Teardrops' == lab['Value']]='vereinzelt Teardroperythrozyten'    
        lab['Value']['vereinz. Tear drops' == lab['Value']]='vereinzelt Teardroperythrozyten'    
        lab.values[['Teardroperythrozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Teardroperythrozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Teardroperythrozyten_BBem' and str(lab['Value'].values[i]) == 'Teardroperythrozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Teardroperythrozyten_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Teardroperythrozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'

        lab.values[['Schistozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Schistozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Schistozyten_BBem' and str(lab['Value'].values[i]) == 'Schistozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'


        lab['Value']['Anulocyten' == lab['Value']]='Anulozyten'  
        lab['Value']['Anulozynte' == lab['Value']]='Anulozyten'        
        lab['Value']['vereinzelt Anulocyten' == lab['Value']]='vereinzelt Anulozyten' 
        lab.values[['Anulozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Anulozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Anulozyten_BBem' and str(lab['Value'].values[i]) == 'Anulozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Anulozyten_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Anulozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab['Value']['Basophile tüpfelung' == lab['Value']]='basophile Tüpfelung'
        lab['Value']['basophile tüpfelung' == lab['Value']]='basophile Tüpfelung'
        lab['Value']['vereinz. basophile Tüpfelung' == lab['Value']]='vereinzelt basophile Tüpfelung'
        lab.values[['basophile Tüpfelung' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Basophile_Tuepfelung_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Basophile_Tuepfelung_BBem' and str(lab['Value'].values[i]) == 'basophile Tüpfelung' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Basophile Tüpfelung_BBem' and str(lab['Value'].values[i]) == 'vereinzelt basophile Tüpfelung' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab['Value']['Echinozyten' == lab['Value']]='Stechapfelzellen'
        lab['Value']['Echinocyten' == lab['Value']]='Stechapfelzellen'
        lab['Value']['Stechapfelzelle' == lab['Value']]='Stechapfelzellen'
        lab['Value']['Stechapfelform' == lab['Value']]='Stechapfelzellen'
        lab['Value']['Stechapfel' == lab['Value']]='Stechapfelzellen'
        lab['Value']['Stechapfelzyten' == lab['Value']]='Stechapfelzellen'
        lab['Value']['vereinzelt Echinozyten' == lab['Value']]='vereinzelt Stechapfelzellen'
        lab['Value']['Stechapfelerythrozyten' == lab['Value']]='Stechapfelzellen'
        lab['Value']['Stechapfelerys' == lab['Value']]='Stechapfelzellen'
        lab.values[['Stechapfelzellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Stechapfelzellen_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Stechapfelzellen_BBem' and str(lab['Value'].values[i]) == 'Stechapfelzellen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Stechapfelzellen_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Stechapfelzellen' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab['Value']['vereinz. Kugelzellen' == lab['Value']]='vereinzelt Kugelzellen'
        lab['Value']['ver. Kugelzellen' == lab['Value']]='vereinzelt Kugelzellen'
        lab['Value']['ver.Kugelzellen' == lab['Value']]='vereinzelt Kugelzellen'
        lab.values[['Kugelzellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Kugelzellen_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kugelzellen_BBem' and str(lab['Value'].values[i]) == 'Kugelzellen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kugelzellen_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Kugelzellen' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab.values[['Stomatozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Stomatozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Stomatozyten_BBem' and str(lab['Value'].values[i]) == 'Stomatozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab['Value']['Targetzellen' == lab['Value']]='Target-Zellen'
        lab['Value']['Targetzelle' == lab['Value']]='Target-Zellen'
        lab['Value']['Target Zellen' == lab['Value']]='Target-Zellen'
        lab['Value']['vereinzelt Tragetzellen' == lab['Value']]='vereinzelt Target-Zellen'
        lab['Value']['ver. Targetzellen' == lab['Value']]='vereinzelt Target-Zellen'
        lab['Value']['vereinzelt Tragetzellen' == lab['Value']]='vereinzelt Target-Zellen'
        lab['Value']['verz.Targetzellen' == lab['Value']]='vereinzelt Target-Zellen'
        lab['Value']['vereinz. Targetzellen' == lab['Value']]='vereinzelt Target-Zellen'
        lab.values[['Target-Zellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Target-Zellen_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Target-Zellen_BBem' and str(lab['Value'].values[i]) == 'Target-Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Target-Zellen_BBem' and str(lab['Value'].values[i]) == 'vereinzelt Target-Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab['Value']['EliptozytenJolly Körperchen' == lab['Value']]='Jolly-Körperchen'
        lab['Value']['Jolly Körperchen' == lab['Value']]='Jolly-Körperchen'
        lab['Value']['Jolly-Körper' == lab['Value']]='Jolly-Körperchen'
        lab['Value']['Jolly Körper' == lab['Value']]='Jolly-Körperchen'
        lab['Value']['Jolly- Körper' == lab['Value']]='Jolly-Körperchen'
        lab['Value']['Joly-Körperchen' == lab['Value']]='Jolly-Körperchen'
        lab.values[['Jolly-Körperchen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Jolly-Koerperchen_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Jolly-Koerperchen_BBem' and str(lab['Value'].values[i]) == 'Jolly-Körperchen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab.values[['Makrozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Makrozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Makrozyten_BBem' and str(lab['Value'].values[i]) == 'Makrozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab['Value']['Doehlekörperchen' == lab['Value']]='Doehle Körperchen'
        lab.values[['Doehle Körperchen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Doehle Körperchen_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Doehle Körperchen_BBem' and str(lab['Value'].values[i]) == 'Doehle Körperchen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        
        lab.values[['Kernschatten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Kernschatten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kernschatten_BBem' and str(lab['Value'].values[i]) == 'Kernschatten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kernschatten_BBem' and str(lab['Value'].values[i]) == 'einige Kernschatten' for i in range(lab['Lab_name'].shape[0])],4] = 'einige'
        
        lab['Value']['toxische Granulation' == lab['Value']]='toxische_Granulation'
        lab.values[['toxische_Granulation' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='toxische_Granulation_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'toxische_Granulation_BBem' and str(lab['Value'].values[i]) == 'toxische_Granulation' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab.values[['Tubuluszellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Tubuluszellen_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Tubuluszellen_UBem' and str(lab['Value'].values[i]) == 'Tubuluszellen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Tubuluszellen_UBem' and str(lab['Value'].values[i]) == 'V.a.Tubuluszellen' for i in range(lab['Lab_name'].shape[0])],4] = 'vor allem'
        
        lab['Value']['viele kaputte' == lab['Value']]='viele defekte Zellen'
        lab.values[['defekte Zellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='defekte_Zellen_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'defekte_Zellen_UBem' and str(lab['Value'].values[i]) == 'defekte Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'defekte_Zellen_UBem' and str(lab['Value'].values[i]) == 'viele defekte Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'vielen'
        
        lab.values[['Tyrosin' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Tyrosin_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Tyrosin_UBem' and str(lab['Value'].values[i]) == 'Tyrosin' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        
        lab.values[['Ammoniumurate' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Ammoniumurate_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Ammoniumurate_UBem' and str(lab['Value'].values[i]) == 'Ammoniumurate' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Ammoniumurate_UBem' and str(lab['Value'].values[i]) == 'vereinzelt Ammoniumurate' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        
        lab['Value']['Hefepilze' == lab['Value']]='Hefezellen'
        lab['Value']['Hefe' == lab['Value']]='Hefezellen'
        lab['Value']['Hefen' == lab['Value']]='Hefezellen'
        lab['Value']['Hezellen' == lab['Value']]='Hefezellen'
        lab['Value']['Heezellen' == lab['Value']]='Hefezellen'
        lab['Value']['hefezellen' == lab['Value']]='Hefezellen'
        lab['Value']['Hefetellen' == lab['Value']]='Hefezellen'
        lab['Value']['Heezellen+' == lab['Value']]='Hefezellen +'
        lab['Value']['Hefepilze: +' == lab['Value']]='Hefezellen +'
        lab['Value']['Hefezellen+' == lab['Value']]='Hefezellen +'
        lab['Value']['Hefezellen(+)' == lab['Value']]='Hefezellen (+)'
        lab['Value']['Hefen+' == lab['Value']]='Hefezellen +'
        lab['Value']['Hefen +' == lab['Value']]='Hefezellen +'
        lab['Value']['Hefe +' == lab['Value']]='Hefezellen +'
        lab['Value']['Hefe (+)' == lab['Value']]='Hefezellen (+)'
        lab['Value']['Hefe((+))' == lab['Value']]='Hefezellen ((+))'
        lab['Value']['Hefe(+)' == lab['Value']]='Hefezellen (+)'
        lab['Value']['Hefen(+)' == lab['Value']]='Hefezellen (+)'
        lab['Value']['ganz ver. Hefezellen' == lab['Value']]='ganz vereinzelt Hefezellen'
        lab['Value']['vereinz. Hefezellen' == lab['Value']]='vereinzelt Hefezellen'
        lab['Value']['Hefetellen++' == lab['Value']]='Hefezellen ++'
        lab['Value']['Hefezellen++' == lab['Value']]='Hefezellen ++'
        lab['Value']['Hefezelln++' == lab['Value']]='Hefezellen ++'
        lab['Value']['Hefe++' == lab['Value']]='Hefezellen ++'
        lab['Value']['Hefe ++' == lab['Value']]='Hefezellen ++'
        lab['Value']['Hefen++' == lab['Value']]='Hefezellen ++'
        lab['Value']['Hefepilze:++' == lab['Value']]='Hefezellen ++'
        lab['Value']['Hefezellen: ++' == lab['Value']]='Hefezellen ++'
        lab['Value']['Hefezellen+++' == lab['Value']]='Hefezellen +++'
        lab['Value']['Hefepilze+++' == lab['Value']]='Hefezellen +++'
        lab['Value']['Hefe +++' == lab['Value']]='Hefezellen +++'
        lab['Value']['Hefezellen: +++' == lab['Value']]='Hefezellen +++'
        lab['Value']['verz.Hefezellen'== lab['Value']]='vereinzelt Hefezellen'
        lab.values[['Hefezellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Hefezellen_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hefezellen_UBem' and str(lab['Value'].values[i]) == 'Hefezellen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hefezellen_UBem' and str(lab['Value'].values[i]) == 'Hefezellen ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hefezellen_UBem' and str(lab['Value'].values[i]) == 'Hefezellen +' for i in range(lab['Lab_name'].shape[0])],4] = '+'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hefezellen_UBem' and str(lab['Value'].values[i]) == 'Hefezellen +++' for i in range(lab['Lab_name'].shape[0])],4] = '+++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hefezellen_UBem' and str(lab['Value'].values[i]) == 'reichlich Hefezellen' for i in range(lab['Lab_name'].shape[0])],4] = 'reichlich'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hefezellen_UBem' and str(lab['Value'].values[i]) == 'Hefezellen (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hefezellen_UBem' and str(lab['Value'].values[i]) == 'ganz vereinzelt Hefezellen' for i in range(lab['Lab_name'].shape[0])],4] = 'ganz vereinzelt'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hefezellen_UBem' and str(lab['Value'].values[i]) == 'vereinzelt Hefezellen' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hefezellen_UBem' and str(lab['Value'].values[i]) == 'Hefezellen ((+))' for i in range(lab['Lab_name'].shape[0])],4] = '((+))'
        
        lab['Value']['Oxalatkristallen' == lab['Value']]='Oxalatkristalle'
        lab['Value']['Oalatkristalle' == lab['Value']]='Oxalatkristalle'
        lab['Value']['Salzkristalle ++' == lab['Value']]='Oxalatkristalle ++'
        lab['Value']['Oxalatkristallen (+)' == lab['Value']]='Oxalatkristalle (+)'
        lab.values[['Oxalatkristalle' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Oxalatkristalle_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Oxalatkristalle_UBem' and str(lab['Value'].values[i]) == 'Oxalatkristalle' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Oxalatkristalle_UBem' and str(lab['Value'].values[i]) == 'Oxalatkristalle (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Oxalatkristalle_UBem' and str(lab['Value'].values[i]) == 'Oxalatkristalle +' for i in range(lab['Lab_name'].shape[0])],4] = '+'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Oxalatkristalle_UBem' and str(lab['Value'].values[i]) == 'Oxalatkristalle ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        
        
        lab['Value']['Oxalate(+)' == lab['Value']]='Oxalate (+)'
        lab['Value']['Ca Oxalate' == lab['Value']]='Oxalate'
        lab['Value']['Ca. Oxalate' == lab['Value']]='Oxalate'
        lab['Value']['Ca.Oxalate' == lab['Value']]='Oxalate'
        lab['Value']['Caoxalate' == lab['Value']]='Oxalate'
        lab['Value']['CaOxalate' == lab['Value']]='Oxalate'
        lab['Value']['Ca-Oxalate' == lab['Value']]='Oxalate'
        lab['Value']['Ca-oxalate' == lab['Value']]='Oxalate'
        lab['Value']['Calciumoxalate' == lab['Value']]='Oxalate'
        lab['Value']['oxalatkristallen' == lab['Value']]='Oxalate'
        lab['Value']['Calciumoxalat' == lab['Value']]='Oxalate'
        lab['Value']['Calciumoalate' == lab['Value']]='Oxalate'
        lab['Value']['calciumoxalate' == lab['Value']]='Oxalate'
        lab['Value']['vereinzelt Calciumoxalate' == lab['Value']]='vereinzelt Oxalate'
        lab['Value']['vz. Calciumoxalate' == lab['Value']]='vereinzelt Oxalate'
        lab['Value']['Calcium-oxalate' == lab['Value']]='Oxalate'
        lab['Value']['Oxalate+' == lab['Value']]='Oxalate +'
        lab['Value']['Oalate+' == lab['Value']]='Oxalate +'
        lab['Value']['Oxalate++' == lab['Value']]='Oxalate ++'
        lab.values[['Oxalate' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Oxalate_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Oxalate_UBem' and str(lab['Value'].values[i]) == 'Oxalate' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Oxalate_UBem' and str(lab['Value'].values[i]) == 'Oxalate +' for i in range(lab['Lab_name'].shape[0])],4] = '+'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Oxalate_UBem' and str(lab['Value'].values[i]) == 'Oxalate (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Oxalate_UBem' and str(lab['Value'].values[i]) == 'Oxalate ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Oxalate_UBem' and str(lab['Value'].values[i]) == 'vereinzelt Oxalate' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab['Value']['Zellreste+' == lab['Value']]='Zellreste +'
        lab['Value']['Zelleste' == lab['Value']]='Zellreste'
        lab['Value']['Zllreste' == lab['Value']]='Zellreste'
        lab['Value']['Zellrete' == lab['Value']]='Zellreste'
        lab['Value']['zellreste' == lab['Value']]='Zellreste'
        lab['Value']['Zellrste' == lab['Value']]='Zellreste'
        lab['Value']['Zellreeste' == lab['Value']]='Zellreste'
        lab['Value']['Zellhaufen++' == lab['Value']]='Zellhaufen ++'
        lab['Value']['grose kernhaltige Zellen' == lab['Value']]='große kernhaltige Zellen'
        lab['Value']['kernhal.Zellen++' == lab['Value']]='kernhaltige Zellen ++'
        lab.values[['Zelltrauben' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Zellen_UBem'
        lab.values[['Zellreste' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Zellen_UBem'
        lab.values[['Zellhaufen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Zellen_UBem'
        lab.values[['große Zellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Zellen_UBem'
        lab.values[['kernhaltige Zellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Zellen_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'Zellreste' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'Zellreste +' for i in range(lab['Lab_name'].shape[0])],4] = '+'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'große kernhaltige Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'große kernhaltige'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'kernhaltige Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'kernhaltige'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'kernhaltige Zellen ++' for i in range(lab['Lab_name'].shape[0])],4] = 'kernhaltige ++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'vereinz. große kernhaltige Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinz. große kernhaltige'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'Zellreste++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'Zellreste+++' for i in range(lab['Lab_name'].shape[0])],4] = '+++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'Zelltrauben' for i in range(lab['Lab_name'].shape[0])],4] = 'Zelltrauben'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'Zellhaufen ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_UBem' and str(lab['Value'].values[i]) == 'große Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'große'

        
        lab['Value']['Leukoztentrauben' == lab['Value']]='Leukozytentrauben'
        lab['Value']['Leukocytentrauben' == lab['Value']]='Leukozytentrauben'
        lab['Value']['Leukotrauben' == lab['Value']]='Leukozytentrauben'
        lab['Value']['Leucotrauben' == lab['Value']]='Leukozytentrauben'
        lab['Value']['vereinzelt Leukotrauben' == lab['Value']]='vereinzelt Leukozytentrauben'
        lab['Value']['ver.Leukotrauben' == lab['Value']]='vereinzelt Leukozytentrauben'
        lab['Value']['Leukozytentraauben' == lab['Value']]='Leukozytentrauben'
        lab['Value']['g.ver. Leukotrauben' == lab['Value']]='ganz vereinzelt Leukozytentrauben'
        lab['Value']['Leukotrauben Zellreste' == lab['Value']]='Leukozytentrauben Zellreste'
        lab.values[['Leukozytentrauben' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Leukozytentrauben_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leukozytentrauben_UBem' and str(lab['Value'].values[i]) == 'Leukozytentrauben' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leukozytentrauben_UBem' and str(lab['Value'].values[i]) == 'Leukozytentrauben Zellreste' for i in range(lab['Lab_name'].shape[0])],4] = 'Zellreste'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leukozytentrauben_UBem' and str(lab['Value'].values[i]) == 'ganz vereinzelt Leukozytentrauben' for i in range(lab['Lab_name'].shape[0])],4] = 'ganz vereinzelt'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leukozytentrauben_UBem' and str(lab['Value'].values[i]) == 'vereinzelt Leukozytentrauben' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        lab['Value']['Leulozyten konglomerate' == lab['Value']]='Leukozytenkonglomerate'
        lab['Value']['Leukozyten konglomerate' == lab['Value']]='Leukozytenkonglomerate'
        lab['Value']['leukozyten konglomerate' == lab['Value']]='Leukozytenkonglomerate'
        lab.values[['Leukozytenkonglomerate' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Leukozytenkonglomerate_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leukozytenkonglomerate_UBem' and str(lab['Value'].values[i]) == 'Leukozytenkonglomerate' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab['Value']['alles von massenhaft Leukos überlagert' == lab['Value']]='Leukozyten massenhaft'
        lab['Value']['Leukozylinder' == lab['Value']]='Leukozytenzylinder'
        lab.values[['Leukozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Leukozyten_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leukozyten_UBem' and str(lab['Value'].values[i]) == 'Leukozyten massenhaft' for i in range(lab['Lab_name'].shape[0])],4] = 'massenhaft'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leukozyten_UBem' and str(lab['Value'].values[i]) == 'Leukozytenzylinder' for i in range(lab['Lab_name'].shape[0])],4] = 'Zylinder'
        
        
        lab['Value']['Uarate' == lab['Value']]='Urate'
        lab['Value']['Utate' == lab['Value']]='Urate'
        lab['Value']['Urate*' == lab['Value']]='Urate'
        lab['Value']['Urate(+)' == lab['Value']]='Urate (+)'
        lab['Value']['Urate+' == lab['Value']]='Urate +'
        lab['Value']['urate+' == lab['Value']]='Urate +'
        lab['Value']['Urate++' == lab['Value']]='Urate ++'
        lab['Value']['Ureate++' == lab['Value']]='Urate ++'
        lab['Value']['Urate+++' == lab['Value']]='Urate +++'
        lab.values[['Urate' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Urate_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Urate_UBem' and str(lab['Value'].values[i]) == 'Urate' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Urate_UBem' and str(lab['Value'].values[i]) == 'Urate (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'    
        lab.values[[str(lab['Lab_name'].values[i]) == 'Urate_UBem' and str(lab['Value'].values[i]) == 'Urate ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'    
        lab.values[[str(lab['Lab_name'].values[i]) == 'Urate_UBem' and str(lab['Value'].values[i]) == 'Urate +' for i in range(lab['Lab_name'].shape[0])],4] = '+'    
        lab.values[[str(lab['Lab_name'].values[i]) == 'Urate_UBem' and str(lab['Value'].values[i]) == 'Urate +++' for i in range(lab['Lab_name'].shape[0])],4] = '+++'    
        lab.values[[str(lab['Lab_name'].values[i]) == 'Urate_UBem' and str(lab['Value'].values[i]) == 'Urate massenhaft' for i in range(lab['Lab_name'].shape[0])],4] = 'massenhaft'    
        
        lab['Value']['Harsäurekristallen' == lab['Value']]='Harnsäurekristalle'
        lab['Value']['Harsäurkristalle' == lab['Value']]='Harnsäurekristalle'
        lab['Value']['Harnsäurkristalle' == lab['Value']]='Harnsäurekristalle'
        lab['Value']['Harsäurekrist.' == lab['Value']]='Harnsäurekristalle'
        lab['Value']['Harsäurekistalle' == lab['Value']]='Harnsäurekristalle'
        lab['Value']['Harnsäurekrristalle' == lab['Value']]='Harnsäurekristalle'
        lab['Value']['Harnsärekristalle' == lab['Value']]='Harnsäurekristalle'
        lab['Value']['Harnsäurekristallen' == lab['Value']]='Harnsäurekristalle'
        lab['Value']['Harnsäurekristalle+++' == lab['Value']]='Harnsäurekristalle +++'
        lab.values[['Harnsäurekristalle' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Harnsaeurekristalle_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Harnsaeurekristalle_UBem' and str(lab['Value'].values[i]) == 'Harnsäurekristalle' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Harnsaeurekristalle_UBem' and str(lab['Value'].values[i]) == 'Harnsäurekristalle +++' for i in range(lab['Lab_name'].shape[0])],4] = '+++'    
        
        lab['Value']['Kristalle+' == lab['Value']]='Kristalle +'
        lab['Value']['Kristalle(+)' == lab['Value']]='Kristalle (+)'
        lab['Value']['Kristalle++' == lab['Value']]='Kristalle ++'
        lab['Value']['kistalle++' == lab['Value']]='Kristalle ++'
        lab['Value']['Ktistalle+++' == lab['Value']]='Kristalle +++'
        lab['Value']['Kristalle+++'== lab['Value']]='Kristalle +++'
        lab['Value']['Kristalle  +++'== lab['Value']]='Kristalle +++' 
        lab['Value']['Kistalle'== lab['Value']]='Kristalle'
        lab['Value']['Kristall'== lab['Value']]='Kristalle'
        lab['Value']['Krstale+'== lab['Value']]='Kristalle +'
        lab.values[['Kristalle' in str(lab['Value'].values[i]) and 'Phosphat' not in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Kristalle_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kristalle_UBem' and str(lab['Value'].values[i]) == 'Kristalle' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kristalle_UBem' and str(lab['Value'].values[i]) == 'Kristalle ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kristalle_UBem' and str(lab['Value'].values[i]) == 'Kristalle (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kristalle_UBem' and str(lab['Value'].values[i]) == 'Kristalle +++' for i in range(lab['Lab_name'].shape[0])],4] = '+++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kristalle_UBem' and str(lab['Value'].values[i]) == 'Kristalle +' for i in range(lab['Lab_name'].shape[0])],4] = '+'
        
        lab['Value']['Amorphen Phosphate' == lab['Value']]='amorphe Phosphate'
        lab['Value']['amorphen Phospathe' == lab['Value']]='amorphe Phosphate'
        lab['Value']['amorphen Phpsphate' == lab['Value']]='amorphe Phosphate'
        lab['Value']['amorphe Phosohate' == lab['Value']]='amorphe Phosphate'
        lab['Value']['Amorphen Phoshate' == lab['Value']]='amorphe Phosphate'
        lab['Value']['Phospate++' == lab['Value']]='Phosphate ++'
        lab['Value']['Phospate' == lab['Value']]='Phosphate'
        lab['Value']['Phosohate ++' == lab['Value']]='Phosphate ++'
        lab['Value']['Phospahte' == lab['Value']]='Phosphate'
        lab['Value']['Phosphate++' == lab['Value']]='Phosphate ++'
        lab['Value']['Phosphte++' == lab['Value']]='Phosphate ++'
        lab['Value']['Phosphate +' == lab['Value']]='Phosphate +'
        lab['Value']['Phosphat' == lab['Value']]='Phosphate'
        lab['Value']['Phosohate' == lab['Value']]='Phosphate'
        lab['Value']['Phosphate+' == lab['Value']]='Phosphate +'
        lab.values[['Phosphate' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Phosphate_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Phosphate_UBem' and str(lab['Value'].values[i]) == 'Phosphate' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Phosphate_UBem' and str(lab['Value'].values[i]) == 'Phosphate ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Phosphate_UBem' and str(lab['Value'].values[i]) == 'amorphe Phosphate' for i in range(lab['Lab_name'].shape[0])],4] = 'amorphe'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Phosphate_UBem' and str(lab['Value'].values[i]) == 'Phosphate +' for i in range(lab['Lab_name'].shape[0])],4] = '+'
        
        lab.values[['Fadenpilze' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Fadenpilze_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Fadenpilze_UBem' and str(lab['Value'].values[i]) == 'Fadenpilze' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab['Value']['Phosphat Kristalle' == lab['Value']]='Phosphatkristalle'
        lab['Value']['Phospathkristalle' == lab['Value']]='Phosphatkristalle'
        lab['Value']['Phosphat Kristallen' == lab['Value']]='Phosphatkristalle'
        lab['Value']['Phosphat- Kristalle' == lab['Value']]='Phosphatkristalle'
        lab.values[['Phosphatkristalle' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Phosphatkristalle_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Phosphatkristalle_UBem' and str(lab['Value'].values[i]) == 'Phosphatkristalle' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Phosphatkristalle_UBem' and str(lab['Value'].values[i]) == 'Phosphatkristalle ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Phosphatkristalle_UBem' and str(lab['Value'].values[i]) == 'Phosphatkristalle +' for i in range(lab['Lab_name'].shape[0])],4] = '+'

        lab['Value']['Zylinder gran.' == lab['Value']]='granulierter Zylinder'
        lab['Value']['Gr. Zylinder' == lab['Value']]='granulierter Zylinder'
        lab['Value']['Gr. Zyliner' == lab['Value']]='granulierter Zylinder'
        lab['Value']['granul. Zylinder' == lab['Value']]='granulierter Zylinder'
        lab['Value']['granulierte ylinder' == lab['Value']]='granulierter Zylinder'
        lab['Value']['granulierte Zylinder' == lab['Value']]='granulierter Zylinder'
        lab['Value']['granulierte Zlinder' == lab['Value']]='granulierter Zylinder'
        lab['Value']['granulierte Z:ylinder' == lab['Value']]='granulierter Zylinder'
        lab['Value']['granulierte Zylinder ++' == lab['Value']]='granulierter Zylinder ++'
        lab['Value']['ver. granulierte Zylinder' == lab['Value']]='vereinzelt granulierter Zylinder'
        lab['Value']['gran.Zylinder' == lab['Value']]='granulierter Zylinder'
        lab['Value']['hyaline Zylinder' == lab['Value']]='hyaliner Zylinder'
        lab['Value']['hyaline' == lab['Value']]='hyaliner Zylinder'
        lab['Value']['hyaline Zyl.' == lab['Value']]='hyaliner Zylinder'
        lab.values[['Zylinder' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Zylinder_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zylinder_UBem' and str(lab['Value'].values[i]) == 'Zylinder' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zylinder_UBem' and str(lab['Value'].values[i]) == 'granulierter Zylinder' for i in range(lab['Lab_name'].shape[0])],4] = 'granulierte'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zylinder_UBem' and str(lab['Value'].values[i]) == 'hyaliner Zylinder' for i in range(lab['Lab_name'].shape[0])],4] = 'hyaline'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zylinder_UBem' and str(lab['Value'].values[i]) == 'granulierter Zylinder ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zylinder_UBem' and str(lab['Value'].values[i]) == 'Zylinder +' for i in range(lab['Lab_name'].shape[0])],4] = '+'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zylinder_UBem' and str(lab['Value'].values[i]) == 'vereinzelt granulierter Zylinder' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt granulierte'
        
        lab.values[['Eryzylinder' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Eryzylinder_UBem'
        
        lab['Value']['runde Epithelien' == lab['Value']]='Rundepithelien'
        lab['Value']['Rundepitheilien' == lab['Value']]='Rundepithelien'
        lab['Value']['Rundepithelzelle' == lab['Value']]='Rundepithelien'
        lab['Value']['Runepithelien' == lab['Value']]='Rundepithelien'
        lab['Value']['Rundepithel' == lab['Value']]='Rundepithelien'
        lab['Value']['Rundepithel: (+)' == lab['Value']]='Rundepithelien (+)'
        lab['Value']['Rundepithelilien(+)' == lab['Value']]='Rundepithelien (+)'
        lab['Value']['Rundepithel:+' == lab['Value']]='Rundepithelien (+)'
        lab['Value']['Rundepithel: (+)' == lab['Value']]='Rundepithelien (+)'
        lab['Value']['Rundepithelie' == lab['Value']]='Rundepithelien'
        lab['Value']['Rundepithelein' == lab['Value']]='Rundepithelien' 
        lab['Value']['ver. Rundepithelien' == lab['Value']]='vereinzelt Rundepithelien'
        lab['Value']['vereinz. RLundepithelien' == lab['Value']]='vereinzelt Rundepithelien'
        lab.values[['Rundepithelien' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Rundepithelien_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Rundepithelien_UBem' and str(lab['Value'].values[i]) == 'Rundepithelien' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Rundepithelien_UBem' and str(lab['Value'].values[i]) == 'vereinzelt Rundepithelien' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Rundepithelien_UBem' and str(lab['Value'].values[i]) == 'Rundepithelien (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        
        lab['Value']['Tripelphosphat' == lab['Value']]='Tripelphosphate'    
        lab['Value']['Tripelphophate' == lab['Value']]='Tripelphosphate'
        lab['Value']['Tripelphosph.++' == lab['Value']]='Tripelphosphate ++'
        lab['Value']['Tripelphospate' == lab['Value']]='Tripelphosphate'
        lab['Value']['Trippelphosphat' == lab['Value']]='Tripelphosphate'    
        lab['Value']['Triplphosphate' == lab['Value']]='Tripelphosphate'    
        lab['Value']['Triepelphosphate' == lab['Value']]='Tripelphosphate'    
        lab['Value']['Trippelphosphate' == lab['Value']]='Tripelphosphate'    
        lab['Value']['Trippelphosphate massenhaft' == lab['Value']]='Tripelphosphate massenhaft'    
        lab['Value']['Tripelphosphate+' == lab['Value']]='Tripelphosphate +'    
        lab['Value']['Triprlphosphate++' == lab['Value']]='Tripelphosphate ++'
        lab['Value']['Tripelphosphate+++' == lab['Value']]='Tripelphosphate +++'
        lab.values[['Tripelphosphate' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Tripelphosphate_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Tripelphosphate_UBem' and str(lab['Value'].values[i]) == 'Tripelphosphate' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Tripelphosphate_UBem' and str(lab['Value'].values[i]) == 'Tripelphosphate +' for i in range(lab['Lab_name'].shape[0])],4] = '+'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Tripelphosphate_UBem' and str(lab['Value'].values[i]) == 'Tripelphosphate ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Tripelphosphate_UBem' and str(lab['Value'].values[i]) == 'Tripelphosphate massenhaft' for i in range(lab['Lab_name'].shape[0])],4] = 'massenhaft'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Tripelphosphate_UBem' and str(lab['Value'].values[i]) == 'Tripelphosphate +++' for i in range(lab['Lab_name'].shape[0])],4] = '+++'
        
        
        lab['Value']['total bluti' == lab['Value']]='total blutig'
        lab.values[['blutig' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='blutig_UBem'
        lab.values[['Blut' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='blutig_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'blutig_UBem' and str(lab['Value'].values[i]) == 'blutig' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'blutig_UBem' and str(lab['Value'].values[i]) == 'total blutig' for i in range(lab['Lab_name'].shape[0])],4] = 'total'
        lab.values[[str(lab['Lab_name'].values[i]) == 'blutig_UBem' and str(lab['Value'].values[i]) == 'Blut makroskopisch sichtbar' for i in range(lab['Lab_name'].shape[0])],4] = 'makroskopisch sichtbar'
        
        lab['Value']['Schleimfäden++' == lab['Value']]='Schleimfäden ++' 
        lab['Value']['Schleimfäden(+)' == lab['Value']]='Schleimfäden (+)' 
        lab.values[['Schleimfäden' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Schleimfaeden_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Schleimfaeden_UBem' and str(lab['Value'].values[i]) == 'Schleimfäden' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Schleimfaeden_UBem' and str(lab['Value'].values[i]) == 'Schleimfäden (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Schleimfaeden_UBem' and str(lab['Value'].values[i]) == 'Schleimfäden ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        
        lab['Value']['sonst nichts erkennbar wegen mass.Erys' == lab['Value']]='massive Erythrozyten' 
        lab['Value']['vereinz. dysmorphe Erys' == lab['Value']]='vereinzelt dysmorphe Erythrozyten' 
        lab.values[['Erythrozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Erythrozyten_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Erythrozyten_UBem' and str(lab['Value'].values[i]) == 'Erythrozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Erythrozyten_UBem' and str(lab['Value'].values[i]) == 'vereinzelt dysmorphe Erythrozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt dysmorphe'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Erythrozyten_UBem' and str(lab['Value'].values[i]) == 'massive Erythrozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'massiv'
        
        
        lab['Value']['Erytrauben' == lab['Value']]='Erythrozytentrauben' 
        lab.values[['Erythrozytentrauben' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Erythrozytentrauben_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Erythrozytentrauben_UBem' and str(lab['Value'].values[i]) == 'Erythrozytentrauben' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab.values[['Bilirubin' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Bilirubin_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Bilirubin_UBem' and str(lab['Value'].values[i]) == 'Bilirubin' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab['Value']['Spermien++' == lab['Value']]='Spermien ++' 
        lab.values[['Spermien' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Spermien_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Spermien_UBem' and str(lab['Value'].values[i]) == 'Spermien' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Spermien_UBem' and str(lab['Value'].values[i]) == 'Spermien ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Spermien_UBem' and str(lab['Value'].values[i]) == 'Spermien (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        
        lab['Value']['Leucinkugeln++' == lab['Value']]='Leucinkugeln ++' 
        lab.values[['Leucinkugeln' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Leucinkugeln_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leucinkugeln_UBem' and str(lab['Value'].values[i]) == 'Leucinkugeln' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leucinkugeln_UBem' and str(lab['Value'].values[i]) == 'Leucinkugeln ++' for i in range(lab['Lab_name'].shape[0])],4] = '++'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leucinkugeln_UBem' and str(lab['Value'].values[i]) == 'Leucinkugeln (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        
        lab['Value']['geschwänzte Epithelzellen: (+)' == lab['Value']]='geschwänzte Epithelien (+)'
        lab['Value']['geschänzte Epithelien' == lab['Value']]='geschwänzte Epithelien'
        lab.values[['geschwänzte Epithelien' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='geschwaenzte_Epithelien_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'geschwaenzte_Epithelien_UBem' and str(lab['Value'].values[i]) == 'geschwänzte Epithelien' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'geschwaenzte_Epithelien_UBem' and str(lab['Value'].values[i]) == 'geschwänzte Epithelien (+)' for i in range(lab['Lab_name'].shape[0])],4] = '(+)'
        
        lab.values[['Sternheimer-Malbin-Zellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Sternheimer_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Sternheimer_Epithelien_UBem' and str(lab['Value'].values[i]) == 'Sternheimer-Malbin-Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Sternheimer_Epithelien_UBem' and str(lab['Value'].values[i]) == 'einige Sternheimer-Malbin-Zellen' for i in range(lab['Lab_name'].shape[0])],4] = 'einige'

        lab['Value']['vereinzelt Leucinkrist.' == lab['Value']]='vereinzelt Leucinkristalle'        
        lab.values[['Leucinkristlle' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Leucinkristalle_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leucinkristalle_UBem' and str(lab['Value'].values[i]) == 'Leucinkristalle' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leucinkristalle_UBem' and str(lab['Value'].values[i]) == 'vereinzelt Leucinkristalle' for i in range(lab['Lab_name'].shape[0])],4] = 'vereinzelt'
        
        
        lab['Value']['hämatolog.Abklärung' == lab['Value']]='haematolog. Abklaerung' 
        lab['Value']['Abklärung hämatolog. Systemerkrankung?' == lab['Value']]='haematolog. Abklaerung'     
        lab.values[['haematolog. Abklaerung' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Abklaerung_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Abklaerung_BBem' and str(lab['Value'].values[i]) == 'haematolog. Abklaerung' for i in range(lab['Lab_name'].shape[0])],4] = 'haematologisch'
        
        lab.values[['Lymphom' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Lymphom_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Lymphom_BBem' and str(lab['Value'].values[i]) == 'Lymphom' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab.values[['Sichelzellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Sichelzellen_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Sichelzellen_BBem' and str(lab['Value'].values[i]) == 'Sichelzellen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        
        lab.values[['Lymphozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Lymphozyten_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Lymphozyten_BBem' and str(lab['Value'].values[i]) == 'Lymphozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Lymphozyten_BBem' and str(lab['Value'].values[i]) == 'vorwiedend kleine nacktkernige Lymphozyte' for i in range(lab['Lab_name'].shape[0])],4] = 'vorwiegend kleine nacktkernige Lymphozyte'
        
        lab.values[['Makrohämaturie' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Makrohämaturie_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Makrohämaturie_UBem' and str(lab['Value'].values[i]) == 'Makrohämaturie' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        
        lab.values[['Schleim' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Schleim_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Schleim_UBem' and str(lab['Value'].values[i]) == 'Schleim' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab.values[['Pilze' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Pilze_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Pilze_UBem' and str(lab['Value'].values[i]) == 'Pilze' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab.values[['Hyphen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='U-Be1' or lab['Lab_name'].values[i]=='U-Be2' or lab['Lab_name'].values[i]=='U-Be3') for i in range(lab['Value'].shape[0])],2]='Pilze_UBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Hyphen_UBem' and str(lab['Value'].values[i]) == 'Hyphen' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        
        lab['Value']['viele Plasmodien' == lab['Value']]='Viele Plasmodium'
        lab.values[['Plasmodium' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='B-Bem1' or lab['Lab_name'].values[i]=='B-Bem2' or lab['Lab_name'].values[i]=='B-Bem3') for i in range(lab['Value'].shape[0])],2]='Plasmodium_BBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Plasmodium_BBem' and str(lab['Value'].values[i]) == 'Plasmodium Befall' for i in range(lab['Lab_name'].shape[0])],4] = 'Befall'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Plasmodium_BBem' and str(lab['Value'].values[i]) == 'Viele Plasmodium' for i in range(lab['Lab_name'].shape[0])],4] = 'Viele'
        
        lab['Value']['Blutig' == lab['Value']]='blutig'
        lab.values[['blutig' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='P-Bem1' or lab['Lab_name'].values[i]=='P-Bem2' or lab['Lab_name'].values[i]=='P-Bem3') for i in range(lab['Value'].shape[0])],2]='blutig_PBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'blutig_PBem' and str(lab['Value'].values[i]) == 'blutig' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'blutig_PBem' and str(lab['Value'].values[i]) == 'sehr blutig' for i in range(lab['Lab_name'].shape[0])],4] = 'sehr'
        
        lab.values[['Kristalle' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='P-Bem1' or lab['Lab_name'].values[i]=='P-Bem2' or lab['Lab_name'].values[i]=='P-Bem3') for i in range(lab['Value'].shape[0])],2]='Kristalle_PBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Kristalle_PBem' and str(lab['Value'].values[i]) == 'reichlich nadelförmige Kristalle' for i in range(lab['Lab_name'].shape[0])],4] = 'reichlich nadelförmige'
        
        
        lab['Value']['Leukos eventuell ungenau' == lab['Value']]='Leukozyten eventuell ungenau'
        lab['Value']['massenhaft Leukos-sonst nichts auswertbar' == lab['Value']]='massenhaft Leukozyten'
        lab.values[['Leukozyten' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='P-Bem1' or lab['Lab_name'].values[i]=='P-Bem2' or lab['Lab_name'].values[i]=='P-Bem3') for i in range(lab['Value'].shape[0])],2]='Leukozyten_PBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leukozyten_PBem' and str(lab['Value'].values[i]) == 'Leukos eventuell ungenau' for i in range(lab['Lab_name'].shape[0])],4] = 'eventuell ungenau'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Leukozyten_PBem' and str(lab['Value'].values[i]) == 'massenhaft Leukozyten' for i in range(lab['Lab_name'].shape[0])],4] = 'massenhaft'
        
        lab.values[['Zellen' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='P-Bem1' or lab['Lab_name'].values[i]=='P-Bem2' or lab['Lab_name'].values[i]=='P-Bem3') for i in range(lab['Value'].shape[0])],2]='Zellen_PBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Zellen_PBem' and str(lab['Value'].values[i]) == 'Zellen kaputt' for i in range(lab['Lab_name'].shape[0])],4] = 'kaputt'
        
        temp = lab[:][lab['Lab_name']=='Frag']
        temp['Value'][[',' in str(i) for i in temp['Value']]]=[i.replace(',','.') for i in temp['Value'][[',' in str(i) for i in temp['Value']]]]
        temp['Value'][temp['Value']=='negativ']='0'
        lab['Value'][temp.index] = temp['Value']
        
        lab['Value'][[',' in str(i) for i in lab['Value']]]=[i.replace(',','.') for i in lab['Value'][[',' in str(i) for i in lab['Value']]]]
        
        lab = lab.drop(lab[:][lab['Lab_name']=='mibi'].index, axis=0)
         
        lab['Value']['Bakterien vorhanden' == lab['Value']] = 'Bakterien' 
        lab['Value']['Bakterien:+' == lab['Value']] = 'Bakterien +'     
        lab.values[['Bakterien' in str(lab['Value'].values[i]) and (lab['Lab_name'].values[i]=='P-Bem1' or lab['Lab_name'].values[i]=='P-Bem2' or lab['Lab_name'].values[i]=='P-Bem3') for i in range(lab['Value'].shape[0])],2]='Bakterien_PBem'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Bakterien_PBem' and str(lab['Value'].values[i]) == 'Bakterien' for i in range(lab['Lab_name'].shape[0])],4] = 'positiv'
        lab.values[[str(lab['Lab_name'].values[i]) == 'Bakterien_PBem' and str(lab['Value'].values[i]) == 'Bakterien +' for i in range(lab['Lab_name'].shape[0])],4] = '+'
        
        ### Loinc 
        lab['Loinc'][lab['Lab_name_long']=='Bilirubin']='L10235-0'
        lab['Lab_name_long'][lab['Lab_name_long']=='Bilirubin']='Bilirubin gesamt'
        lab['Loinc'][lab['Lab_name']=='pH']='L13515-2'
        
        ### reference values
        lab['lower_ref']['Alkohol'==lab['Lab_name_long']] = 0
        lab['upper_ref']['Alkohol'==lab['Lab_name_long']] = 0
        lab['lower_ref']['O2-Sät'==lab['Lab_name']] = 95
        lab['upper_ref']['O2-Sät'==lab['Lab_name']] = 100
        lab['lower_ref']['dct'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['dct'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['AMP'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['AMP'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['BAR'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['BAR'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['BZD'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['BZD'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['MTD'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['MTD'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['TCA'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['TCA'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['COC'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['COC'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['MET'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['MET'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['MOR'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['MOR'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['THC'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['THC'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['XTC'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['XTC'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['Frag'==lab['Lab_name']] = 0.03
        lab['upper_ref']['Frag'==lab['Lab_name']] = 0.56
        lab['lower_ref']['Kryogl'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['Kryogl'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['ULeu'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['ULeu'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['L-Zz'==lab['Lab_name']] = 0
        lab['lower_ref']['Na/Ra'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['Na/Ra'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['Haut'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['Haut'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['Kathet'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['Kathet'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['Lei/Na'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['Lei/Na'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['SonLok'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['SonLok'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['Sonst'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['Sonst'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['Wu Lok'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['Wu Lok'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['Wunde'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['Wunde'==lab['Lab_name']] = 'negativ'
        lab['lower_ref']['Trache'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['Trache'==lab['Lab_name']] = 'negativ'
        lab['upper_ref']['IL-6'==lab['Lab_name']] = 7.0
        
        lab['Number case']=lab['Number case'].astype('int')
        
        temp = lab[:]['GFR'==lab['Lab_name']]
        temp = temp[:][temp['lower_ref'].isna()]
        for i in range(temp['Number case'].shape[0]):
            age = base['Age'][base['Number case']==temp['Number case'].iloc[i]]
            if age.isnull().any()==False:
                age =  int(age)
                if age>=18:
                    temp['lower_ref'].iloc[i] = 70
                    temp['upper_ref'].iloc[i] = 'no upper ref'
        lab['lower_ref'][temp.index] = temp['lower_ref']
        lab['upper_ref'][temp.index] = temp['upper_ref']
        
        #temp2=temp[:][temp['Value'].notna()]
        #temp2[:][temp2['Value'].astype('int')<70]
        
        temp = lab[:]['Gluc'==lab['Lab_name']]
        temp = temp[:][temp['lower_ref'].isna()]
        for i in range(temp['Number case'].shape[0]):
            age = base['Age'][base['Number case']==temp['Number case'].iloc[i]]
            if age.isnull().any()==False:
                try:
                    age = int(age)
                    if age>=18:
                        temp['lower_ref'].iloc[i] = 60
                        temp['upper_ref'].iloc[i] = 120
                except:
                    if not age.empty:
                        age = int(age.values[0])
                        if age>=18:
                            temp['lower_ref'].iloc[i] = 60
                            temp['upper_ref'].iloc[i] = 120
        lab['lower_ref'][temp.index] = temp['lower_ref']
        lab['upper_ref'][temp.index] = temp['upper_ref']
        
        temp = lab[:]['K'==lab['Lab_name']]
        temp = temp[:][temp['lower_ref'].isna()]
        for i in range(temp['Number case'].shape[0]):
            age = base['Age'][base['Number case']==temp['Number case'].iloc[i]]
            if age.isnull().any()==False:
                try:
                    age =  int(age)
                    if age>=18:
                        temp['lower_ref'].iloc[i] = 3.6
                        temp['upper_ref'].iloc[i] = 5.0
                except:
                    if not age.empty:
                        age = int(age.values[0])
                        if age>=18:
                            temp['lower_ref'].iloc[i] = 3.6
                            temp['upper_ref'].iloc[i] = 5.0
        lab['lower_ref'][temp.index] = temp['lower_ref']
        lab['upper_ref'][temp.index] = temp['upper_ref']
        
        temp = lab[:]['Lact'==lab['Lab_name']]
        temp = temp[:][temp['lower_ref'].isna()]
        for i in range(temp['Number case'].shape[0]):
            age = base['Age'][base['Number case']==temp['Number case'].iloc[i]]
            if age.isnull().any()==False:
                try:
                    age =  int(age)
                    if age>=18:
                        temp['lower_ref'].iloc[i] = 5.7
                        temp['upper_ref'].iloc[i] = 22
                except:
                    if not age.empty:
                        age = int(age.values[0])
                        if age>=18:
                            temp['lower_ref'].iloc[i] = 5.7
                            temp['upper_ref'].iloc[i] = 22
        lab['lower_ref'][temp.index] = temp['lower_ref']
        lab['upper_ref'][temp.index] = temp['upper_ref']
        
        temp = lab[:]['Na'==lab['Lab_name']]
        temp = temp[:][temp['lower_ref'].isna()]
        for i in range(temp['Number case'].shape[0]):
            age = base['Age'][base['Number case']==temp['Number case'].iloc[i]]
            if age.isnull().any()==False:
                try:
                    age =  int(age)
                    if age>=18:
                        temp['lower_ref'].iloc[i] = 135
                        temp['upper_ref'].iloc[i] = 145
                except:
                    if not age.empty:
                        age = int(age.values[0])
                        if age>=18:
                            temp['lower_ref'].iloc[i] = 135
                            temp['upper_ref'].iloc[i] = 145
        lab['lower_ref'][temp.index] = temp['lower_ref']
        lab['upper_ref'][temp.index] = temp['upper_ref']
        
        temp = lab[:]['Hctc'==lab['Lab_name']]
        for i in range(temp['Number case'].shape[0]):
            sex = base['Sex'][base['Number case']==temp['Number case'].iloc[i]]
            if sex.isnull().any()==False:
                sex =  str(sex.iloc[0])
                if sex=='W':
                    temp['lower_ref'].iloc[i] = 36.0
                    temp['upper_ref'].iloc[i] = 46.0
                else:
                    temp['lower_ref'].iloc[i] = 38.0
                    temp['upper_ref'].iloc[i] = 52.0
        lab['lower_ref'][temp.index] = temp['lower_ref']
        lab['upper_ref'][temp.index] = temp['upper_ref']
        #
        
        lab['Lab_name_long'][lab['Lab_name']=='Hctc'] = 'Hämatokrit'
        lab['Lab_name'][lab['Lab_name']=='Hctc'] = 'HK'
        
        lab['Lab_name'][lab['Lab_name']=='U-pH'] = 'UpH'
        
        lab['upper_ref']['CEA'==lab['Lab_name']] = 'NR<3/R<5'

        lab['Lab_name']['ctBil' == lab['Lab_name']] = 'Bili g'
        lab['Lab_name']['-ctBilirubin' == lab['Lab_name']] = 'Bilirubin gesamt'
        lab['Lab_name']['ctHb' == lab['Lab_name']] = 'Hb'
        lab['Lab_name']['-ctHämoglobin' == lab['Lab_name']] = 'Hämoglobin'

        # removing <=, <, >=, > in 'Value' if it isnt included in the reference range 
        temp = lab[:][['<=' in str(i) for i in lab['Value']]]
        temp['Value'][[float(temp['Value'].values[i][2:]) <= float(temp['lower_ref'].values[i]) for i in range(temp['Value'].shape[0])]] = [str(temp['Value'].values[i][2:]) for i in range(temp['Value'].shape[0]) if float(temp['Value'].values[i][2:]) <= float(temp['lower_ref'].values[i])]
        lab['Value'][temp.index] = temp['Value']     
        
        temp = lab[:][['>=' in str(i) for i in lab['Value']]]
        temp['Value'][[float(temp['Value'].values[i][2:]) >= float(temp['upper_ref'].values[i]) for i in range(temp['Value'].shape[0])]] = [str(temp['Value'].values[i][2:]) for i in range(temp['Value'].shape[0]) if float(temp['Value'].values[i][2:]) >= float(temp['upper_ref'].values[i])]
        lab['Value'][temp.index] = temp['Value']   
        
        temp = lab[:][['<' in str(i) for i in lab['Value']]]
        temp['Value'][[float(temp['Value'].values[i][1:]) <= float(temp['lower_ref'].values[i]) for i in range(temp['Value'].shape[0])]] = [str(temp['Value'].values[i][1:]) for i in range(temp['Value'].shape[0]) if float(temp['Value'].values[i][1:]) <= float(temp['lower_ref'].values[i])]
        lab['Value'][temp.index] = temp['Value'] 
        
        temp = lab[:][['>' in str(i) for i in lab['Value']]]
        temp['Value'][[float(temp['Value'].values[i][1:]) >= float(temp['upper_ref'].values[i]) for i in range(temp['Value'].shape[0])]] = [str(temp['Value'].values[i][1:]) for i in range(temp['Value'].shape[0]) if float(temp['Value'].values[i][1:]) >= float(temp['upper_ref'].values[i])]
        lab['Value'][temp.index] = temp['Value']
        
        # removing <,> if it is allowed by original data (no lower range)
        temp = lab[:][lab['Lab_name']=='NTpBNP']
        temp['Value'][['<' in str(i) for i in temp['Value']]] = [str(i[1:]) for i in temp['Value'] if '<' in str(i)]
        lab['Value'][temp.index] = temp['Value']
        
        temp = lab[:][lab['Lab_name']=='IL-6']
        temp['Value'][['<' in str(i) for i in temp['Value']]] = [str(i[1:]) for i in temp['Value'] if '<' in str(i)]
        lab['Value'][temp.index] = temp['Value']
        
        temp = lab[:][lab['Lab_name']=='b-HCG']
        temp['Value'][['<' in str(i) for i in temp['Value']]] = [str(i[1:]) for i in temp['Value'] if '<' in str(i)]
        lab['Value'][temp.index] = temp['Value']
        
        temp = lab[:][lab['Lab_name']=='ASL']
        temp['Value'][['<' in str(i) for i in temp['Value']]] = [str(i[1:]) for i in temp['Value'] if '<' in str(i)]
        lab['Value'][temp.index] = temp['Value']
        
        # delete 'Bemerkungen' 
        lab = lab.drop(lab[:][lab['Lab_name']=='U-Be1'].index, axis=0)
        lab = lab.drop(lab[:][lab['Lab_name']=='U-Be2'].index, axis=0)
        lab = lab.drop(lab[:][lab['Lab_name']=='U-Be3'].index, axis=0)
        lab = lab.drop(lab[:][lab['Lab_name']=='B-Bem1'].index, axis=0)
        lab = lab.drop(lab[:][lab['Lab_name']=='B-Bem2'].index, axis=0)
        lab = lab.drop(lab[:][lab['Lab_name']=='B-Bem3'].index, axis=0)
        lab = lab.drop(lab[:][lab['Lab_name']=='P-Bem1'].index, axis=0)
        
    #print(datetime.now())
    lab['Date'][lab['Lab_name_long']=='Blutzucker 16 Uhr'] = [str(str(i)[:str(i).index(' ')+1] + '16:00') for i in lab['Date'][lab['Lab_name_long']=='Blutzucker 16 Uhr']]
    lab['Date'][lab['Lab_name_long']=='Blutzucker 11 Uhr'] = [str(str(i)[:str(i).index(' ')+1] + '11:00') for i in lab['Date'][lab['Lab_name_long']=='Blutzucker 11 Uhr']]
    lab['Date'][lab['Lab_name_long']=='Blutzucker nüchtern'] = [str(str(i)[:str(i).index(' ')+1]) for i in lab['Date'][lab['Lab_name_long']=='Blutzucker nüchtern']]
    lab['Date'] = pd.to_datetime(lab['Date'],dayfirst=True,errors='coerce')

    lab['Number case']=lab['Number case'].astype('int')
    lab['Lab_name']=lab['Lab_name'].astype('string')
    lab['Lab_name_long']=lab['Lab_name_long'].astype('string')
    lab['Einheit']=lab['Einheit'].astype('string')

    lab['Loinc']=lab['Loinc'].astype('string')
    
    lab=lab.sort_index().sort_values(by='Number case', kind='mergesort')
    lab=lab.drop_duplicates()
 
    bew=data[data['B']=='006BEW']
    
    bew.columns=['Number case','Datatype','Bewegung','Date','Bewegungstyp','Bewegungsart','Bewegungsgrund','Bewegungsgrund Langtext','Fachliche_OE','Fachliche_OE Langtext','Notfallkennzeichen','Erfassungsdatum','Aenderungsdatum','Fachrichtung_aufnehmende_OE_Fall','Fachrichtung_entlassene_OE_Fall','Fachliche_OE_Bew','Pflege_OE_Bew','Einweisungsart','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    bew=bew[['Number case','Date','Bewegungstyp','Bewegungsart','Bewegungsgrund','Bewegungsgrund Langtext','Fachliche_OE','Fachliche_OE Langtext','Notfallkennzeichen','Erfassungsdatum','Fachrichtung_aufnehmende_OE_Fall','Fachrichtung_entlassene_OE_Fall','Fachliche_OE_Bew','Pflege_OE_Bew','Einweisungsart']]

    #bew['Einweisungsart'][bew['Einweisungsart'].notnull()]=[i[:i.index(' ')] for i in bew['Einweisungsart'][bew['Einweisungsart'].notnull()]]
    bew['Einweisungsart']=bew['Einweisungsart'].astype('string')
    bew['Fachrichtung_aufnehmende_OE_Fall']=bew['Fachrichtung_aufnehmende_OE_Fall'].astype('string')
    bew['Fachrichtung_entlassene_OE_Fall']=bew['Fachrichtung_entlassene_OE_Fall'].astype('string')
    bew['Fachliche_OE_Bew']=bew['Fachliche_OE_Bew'].astype('string')
    bew['Pflege_OE_Bew']=bew['Pflege_OE_Bew'].astype('string')

    if sum(bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet')>=1:
        bew['Fachliche_OE'][bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet']=2400
        bew['Fachliche_OE Langtext'][bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet']='Frauenheilkunde und Geburtshilfe'
        bew['Notfallkennzeichen'][bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet']='kein'
        bew['Erfassungsdatum'][bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet']='07.04.2022 16:35'
        bew['Fachrichtung_aufnehmende_OE_Fall'][bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet']='GB'
        bew['Fachrichtung_entlassene_OE_Fall'][bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet']='GB'
        bew['Fachliche_OE_Bew'][bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet']='GB'
        bew['Pflege_OE_Bew'][bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet']='30'
        bew['Einweisungsart'][bew['Bewegungsgrund Langtext']=='Behandlung regulär beendet, invasiv beatmet']=''
    
    bew['Number case']=bew['Number case'].astype('int')
    bew['Bewegungstyp']=bew['Bewegungstyp'].astype('string')
    bew['Bewegungsart']=bew['Bewegungsart'].astype('string')
    bew['Bewegungsgrund']=bew['Bewegungsgrund'].astype('string')
    bew['Fachliche_OE']=bew['Fachliche_OE'].astype(str).astype('string')
    bew['Fachliche_OE Langtext']=bew['Fachliche_OE Langtext'].astype('string')
    bew['Bewegungsgrund Langtext']=bew['Bewegungsgrund Langtext'].astype('string')
    bew['Notfallkennzeichen']=bew['Notfallkennzeichen'].astype('string')
    bew['Date'] = pd.to_datetime(bew['Date'],dayfirst=True,errors='coerce')
    bew=bew.sort_index().sort_values(by='Number case', kind='mergesort')
    bew=bew.drop_duplicates()    
    
    # for i in bew['Number case'].unique():
    #     test=bew[bew['Number case']==i]
    #     helpvar=False
    #     for j in test['Bewegungstyp']:
    #         if 'Aufnahme' in j:
    #             helpvar=True
    #     if helpvar==False:
    #         print(i)
    
    vor=data[data['B']=='007VOR']
    vor.columns=['Number case','Datatype','Vorgang','Date','Vorgangskategorie','Kategoriename','Eingriff','Erfassungsdatum','Datum der letzten Aenderung','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    vor=vor[['Number case','Date','Vorgangskategorie','Kategoriename','Eingriff','Erfassungsdatum']]
    vor['Date'] = pd.to_datetime(vor['Date'],dayfirst=True,errors='coerce')
    vor['Erfassungsdatum'] = pd.to_datetime(vor['Erfassungsdatum'],dayfirst=True,errors='coerce')
    vor['Number case']=vor['Number case'].astype('int')
    vor['Vorgangskategorie']=vor['Vorgangskategorie'].astype('string')
    vor['Kategoriename']=vor['Kategoriename'].astype('string')
    vor['Eingriff']=vor['Eingriff'].astype('string')
    vor=vor.sort_index().sort_values(by='Number case', kind='mergesort')
    vor = vor.drop_duplicates()    
    
    vit=data[data['B']=='008KUR']
    
    vit.columns=['Number case', 'Datatype', 'Kurve', 'Date', 'Wert 1', 'Bezeichnung zu Wert 1', 'Wert 2', 'Bezeichnung zu Wert 2','Bemerkung','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    vit=vit[['Number case', 'Date', 'Wert 1', 'Bezeichnung zu Wert 1', 'Wert 2', 'Bezeichnung zu Wert 2','Bemerkung']]
    vit['Bezeichnung zu Wert 1'][['  ' in str(i) for i in vit['Bezeichnung zu Wert 1']]]=[str(i)[:str(i).index(' ')] for i in vit['Bezeichnung zu Wert 1'][['  ' in str(i) for i in vit['Bezeichnung zu Wert 1']]]]
    vit['Bezeichnung zu Wert 2'][['  ' in str(i) for i in vit['Bezeichnung zu Wert 2']]]=[str(i)[:str(i).index(' ')] for i in vit['Bezeichnung zu Wert 2'][['  ' in str(i) for i in vit['Bezeichnung zu Wert 2']]]]
    vit['Bemerkung'][['  ' in str(i) for i in vit['Bemerkung']]]=[str(i)[:str(i).index('  ')] for i in vit['Bemerkung'][['  ' in str(i) for i in vit['Bemerkung']]]]

    vit.values[[not i for i in vit['Bezeichnung zu Wert 2']],5]=np.nan
    vit.values[[not i for i in vit['Bemerkung']],6]=np.nan
    vit['Date'] = pd.to_datetime(vit['Date'],dayfirst=True)
    
    if vit_complete==True:
        
        ### deleting times (already in 'Date')
        vit['Wert 1'][['(' in str(i) for i in vit['Wert 1']]]=[i[:i.index('(')] for i in vit['Wert 1'][['(' in str(i) for i in vit['Wert 1']]]]
        vit['Wert 2'][['(' in str(i) for i in vit['Wert 2']]]=[i[:i.index('(')] for i in vit['Wert 2'][['(' in str(i) for i in vit['Wert 2']]]]
        vit['Bezeichnung zu Wert 1'][['(' in str(i) for i in vit['Bezeichnung zu Wert 1']]]=[i[:i.index('(')] for i in vit['Bezeichnung zu Wert 1'][['(' in str(i) for i in vit['Bezeichnung zu Wert 1']]]]
        vit['Bezeichnung zu Wert 1'][vit['Bezeichnung zu Wert 1']=='RR ']='RR (sys)'
        
        ### Wert 1: unit as new column 
        # generating column 'Wert 1 Einheit' 
        vit.insert(3,'Wert 1 Einheit',vit['Wert 1']) 
        
        # 'Temperatur','RR (sys)','Puls','Größe','Gewicht','Ausfuhr Urin','Einfuhr','Sauerstoffsättigung','Drainage1','Drainage2',
        # 'Größe_PA','Muttermilch','Kopfumfang','Wärmebett','Flüssigkeitsbilanz' sind alle mit Lücke zu trennen
        
        # Index: separation via gap 
        idx = vit[:][(vit['Bezeichnung zu Wert 1']=='Temperatur') | (vit['Bezeichnung zu Wert 1']=='RR (sys)') | (vit['Bezeichnung zu Wert 1']=='Puls') | 
                     (vit['Bezeichnung zu Wert 1']=='Größe') | (vit['Bezeichnung zu Wert 1']=='Gewicht') | (vit['Bezeichnung zu Wert 1']=='Ausfuhr Urin') | 
                     (vit['Bezeichnung zu Wert 1']=='Einfuhr') | (vit['Bezeichnung zu Wert 1']=='Sauerstoffsättigung') | (vit['Bezeichnung zu Wert 1']=='Drainage1') | 
                     (vit['Bezeichnung zu Wert 1']=='Drainage2') | (vit['Bezeichnung zu Wert 1']=='Größe_PA') | (vit['Bezeichnung zu Wert 1']=='Muttermilch') | 
                     (vit['Bezeichnung zu Wert 1']=='Kopfumfang') | (vit['Bezeichnung zu Wert 1']=='Wärmebett') | (vit['Bezeichnung zu Wert 1']=='Flüssigkeitsbilanz')].index
        temp = vit.loc[idx]
        # Bei 'Wert 1' die Einheit abschneiden
        temp['Wert 1'][[' ' in str(i) for i in temp['Wert 1']]]=[i[:i.index(' ')] for i in temp['Wert 1'][[' ' in str(i) for i in temp['Wert 1']]]]
        vit['Wert 1'][idx]=temp['Wert 1']
        # Bei 'Wert 1 Einheit' die Zahl abschneiden
        temp['Wert 1 Einheit'][[' ' in str(i) for i in temp['Wert 1 Einheit']]]=[i[i.index(' ')+1:-1] for i in temp['Wert 1 Einheit'][[' ' in str(i) for i in temp['Wert 1 Einheit']]]]
        vit['Wert 1 Einheit'][idx]=temp['Wert 1 Einheit']
        # RR (sys) Einheit nach bessern 
        idx = vit[:][vit['Bezeichnung zu Wert 1']=='RR (sys)'].index
        temp = vit.loc[idx]
        temp['Wert 1 Einheit'] = 'mmHg'
        vit['Wert 1 Einheit'][idx]=temp['Wert 1 Einheit']
        
        # Index: no unit of 'Wert 1'
        idx = vit[:][(vit['Bezeichnung zu Wert 1']=='Kostform') | (vit['Bezeichnung zu Wert 1']=='Besondere Kontrollen') | (vit['Bezeichnung zu Wert 1']=='R_Schmerz') | 
                     (vit['Bezeichnung zu Wert 1']=='BMI') | (vit['Bezeichnung zu Wert 1']=='B_Schmerz') | (vit['Bezeichnung zu Wert 1']=='PVK') | 
                     (vit['Bezeichnung zu Wert 1']=='Atemfrequenz') | (vit['Bezeichnung zu Wert 1']=='Tägl. Anordnung') | (vit['Bezeichnung zu Wert 1']=='DK') | 
                     (vit['Bezeichnung zu Wert 1']=='ZVK') | (vit['Bezeichnung zu Wert 1']=='Erbrechen') | (vit['Bezeichnung zu Wert 1']==' CRP ') | 
                     (vit['Bezeichnung zu Wert 1']==" U'status ") | (vit['Bezeichnung zu Wert 1']==' E"lyte ') | (vit['Bezeichnung zu Wert 1']=='Krampfanfall') | (vit['Bezeichnung zu Wert 1']==' EKG ') | 
                     (vit['Bezeichnung zu Wert 1']==' Urin, Kreuzblut ') | (vit['Bezeichnung zu Wert 1']==' Prevenar 13 Ch.-B.:DA8219 ') | (vit['Bezeichnung zu Wert 1']==' Port ex> ') | 
                     (vit['Bezeichnung zu Wert 1']==' BERA ') | (vit['Bezeichnung zu Wert 1']==' BE und BZ 79 ') | (vit['Bezeichnung zu Wert 1']==' o.B. ') |
                     (vit['Bezeichnung zu Wert 1']=='Freitext 5') | (vit['Bezeichnung zu Wert 1']=='Freitext 6') | (vit['Bezeichnung zu Wert 1']=='Freitext 7') |
                     (vit['Bezeichnung zu Wert 1']=='Freitext 8') | (vit['Bezeichnung zu Wert 1']=='Freitext 9') | (vit['Bezeichnung zu Wert 1']=='Freitext 10')].index
        temp = vit.loc[idx]
        # 'Wert 1 Einheit' = nan 
        temp['Wert 1 Einheit'] = np.nan
        vit['Wert 1 Einheit'][idx]=temp['Wert 1 Einheit']
        
        # 'Bezeichnung zu Wert 1'==Stuhl 
        idx = vit[:][vit['Bezeichnung zu Wert 1']=='Stuhl'].index
        temp = vit.loc[idx]
        # delete nan-rows in dataset vit
        vit=vit.drop(temp[:][temp['Wert 1'].isna()].index, axis=0)
        temp=temp.drop(temp[:][temp['Wert 1'].isna()].index, axis=0)
        temp['Wert 1'][[' ' in str(i) for i in temp['Wert 1']]]=[i[:i.index(' ')] for i in temp['Wert 1'][[' ' in str(i) for i in temp['Wert 1']]]]
        temp['Wert 1']['kein'==temp['Wert 1']]='kein Stuhl'
        vit['Wert 1'][temp.index]=temp['Wert 1']
        # Bei 'Wert 1 Einheit' Text abschneiden (doppelte Ausführung zwingend)
        # temp['Wert 1 Einheit'][[' ' in str(i) for i in temp['Wert 1 Einheit']]]=[i[i.index(' ')+1:-1] for i in temp['Wert 1 Einheit'][[' ' in str(i) for i in temp['Wert 1 Einheit']]]]
        # temp['Wert 1 Einheit'][[' ' in str(i) for i in temp['Wert 1 Einheit']]]=[i[i.index(' ')+1:-1] for i in temp['Wert 1 Einheit'][[' ' in str(i) for i in temp['Wert 1 Einheit']]]]
        # temp['Wert 1 Einheit'].unique()
        # temp['Wert 1 Einheit']['m'==temp['Wert 1 Einheit']]='ml' 
        # temp['Wert 1 Einheit'].unique()
        temp['Wert 1 Einheit'] = np.nan
        vit['Wert 1 Einheit'][temp.index]=temp['Wert 1 Einheit']
        
        # 'Bezeichnung zu Wert 1'==Drainage
        idx = vit[:][vit['Bezeichnung zu Wert 1']=='Drainage'].index
        temp = vit.loc[idx]
        temp['Wert 1'][['ml' in str(i) for i in temp['Wert 1']]]=[i.replace('ml','') for i in temp['Wert 1'][['ml' in str(i) for i in temp['Wert 1']]]]
        vit['Wert 1'][idx]=temp['Wert 1']
        temp['Wert 1 Einheit'] = 'ml'
        vit['Wert 1 Einheit'][idx] = temp['Wert 1 Einheit']
        
        # 'Bezeichnung zu Wert 1'==Hämodialyse
        idx = vit[:][vit['Bezeichnung zu Wert 1']=='Hämodialyse'].index
        temp = vit.loc[idx]
        temp['Wert 1'][['ml' in str(i) for i in temp['We*rt 1']]]=[i.replace('ml','') for i in temp['Wert 1'][['ml' in str(i) for i in temp['Wert 1']]]]
        vit['Wert 1'][idx]=temp['Wert 1']
        temp['Wert 1 Einheit'] = 'ml'
        vit['Wert 1 Einheit'][idx] = temp['Wert 1 Einheit']
        
        # 'Bezeichnung zu Wert 1'==Sauerstoffgabe
        idx = vit[:][vit['Bezeichnung zu Wert 1']=='Sauerstoffgabe'].index
        temp = vit.loc[idx]
        temp['Wert 1'][['L' in str(i) for i in temp['Wert 1']]]=[i.replace('L','') for i in temp['Wert 1'][['L' in str(i) for i in temp['Wert 1']]]]
        temp['Wert 1'][['l' in str(i) for i in temp['Wert 1']]]=[i.replace('l','') for i in temp['Wert 1'][['l' in str(i) for i in temp['Wert 1']]]]
        temp['Wert 1'][['iter' in str(i) for i in temp['Wert 1']]]=[i.replace('iter','') for i in temp['Wert 1'][['iter' in str(i) for i in temp['Wert 1']]]]
        temp['Wert 1'][['/min' in str(i) for i in temp['Wert 1']]]=[i.replace('/min','') for i in temp['Wert 1'][['/min' in str(i) for i in temp['Wert 1']]]]
        temp['Wert 1'][['/m' in str(i) for i in temp['Wert 1']]]=[i.replace('/m','') for i in temp['Wert 1'][['/m' in str(i) for i in temp['Wert 1']]]]
        tempidx = temp['Wert 1'][['/h' in str(i) for i in temp['Wert 1']]].index
        temp['Wert 1'][['/h' in str(i) for i in temp['Wert 1']]]=[i.replace('/h','') for i in temp['Wert 1'][['/h' in str(i) for i in temp['Wert 1']]]]
        temp2 = temp.loc[tempidx]
        blub = temp.loc[tempidx]['Wert 1'].astype(float)/60
        temp2['Wert 1'] = blub
        temp['Wert 1'][tempidx] = temp2['Wert 1']
        temp['Wert 1'][['Min' in str(i) for i in temp['Wert 1']]]=[i.replace('Min','') for i in temp['Wert 1'][['Min' in str(i) for i in temp['Wert 1']]]]
        temp['Wert 1'][['lt' in str(i) for i in temp['Wert 1']]]=[i.replace('lt','') for i in temp['Wert 1'][['lt' in str(i) for i in temp['Wert 1']]]]
        vit['Wert 1'][idx]=temp['Wert 1']
        temp['Wert 1 Einheit'] = 'l/min'
        vit['Wert 1 Einheit'][idx] = temp['Wert 1 Einheit']
        
        # 'Bezeichnung zu Wert 1'==Magensonde
        idx = vit[:][vit['Bezeichnung zu Wert 1']=='Magensonde'].index
        temp = vit.loc[idx]
        temp['Wert 1'][['ml' in str(i) for i in temp['Wert 1']]]=[i.replace('ml','') for i in temp['Wert 1'][['ml' in str(i) for i in temp['Wert 1']]]]
        vit['Wert 1'][idx]=temp['Wert 1']
        temp['Wert 1 Einheit'] = 'ml'
        vit['Wert 1 Einheit'][idx] = temp['Wert 1 Einheit']
        
        # 'Bezeichnung zu Wert 1'==Sensorwechsel
        idx = vit[:][vit['Bezeichnung zu Wert 1']=='Sensorwechsel'].index
        temp = vit.loc[idx]
        temp['Wert 1'][['ml' in str(i) for i in temp['Wert 1']]]=[i.replace('ml','') for i in temp['Wert 1'][['ml' in str(i) for i in temp['Wert 1']]]]
        vit['Wert 1'][idx]=temp['Wert 1']
        temp['Wert 1 Einheit'] = 'ml'
        vit['Wert 1 Einheit'][idx] = temp['Wert 1 Einheit']
        
        # 'Bezeichnung zu Wert 1'==Trinkprotokoll
        idx = vit[:][vit['Bezeichnung zu Wert 1']=='Trinkprotokoll'].index
        temp = vit.loc[idx]
        temp['Wert 1'][['ml' in str(i) for i in temp['Wert 1']]]=[i.replace('ml','') for i in temp['Wert 1'][['ml' in str(i) for i in temp['Wert 1']]]]
        temp['Wert 1'][['l' in str(i) for i in temp['Wert 1']]]=[i.replace('l','000') for i in temp['Wert 1'][['l' in str(i) for i in temp['Wert 1']]]]
        vit['Wert 1'][idx]=temp['Wert 1']
        temp['Wert 1 Einheit'] = 'ml'
        vit['Wert 1 Einheit'][idx] = temp['Wert 1 Einheit']
        temp['Wert 1']['Trinkprotokoll'==temp['Bezeichnung zu Wert 1']].unique()
        # adjusting values: only numbers 
        temp['Wert 1'][temp['Wert 1']=='X ']='0'
        temp['Wert 1'][temp['Wert 1']=='X400 ']='400'
        temp['Wert 1'][temp['Wert 1']=='> ']=np.nan
        temp['Wert 1'][[' ' in str(i) for i in temp['Wert 1']]]=[i[:i.index(' ')] for i in temp['Wert 1'][[' ' in str(i) for i in temp['Wert 1']]]]
        temp['Wert 1']['Trinkprotokoll'==temp['Bezeichnung zu Wert 1']].unique()
        vit['Wert 1'][idx]=temp['Wert 1']
        
        ### Wert 2: unit as new column 
        # generating column 'Wert 2 Einheit' 
        vit.insert(6,'Wert 2 Einheit',np.nan) 
        
        # 'Bezeichnung zu Wert 2'==RR
        
        idx = vit[:][vit['Bezeichnung zu Wert 2']=='RR (dia)'].index    
        temp = vit.loc[idx]
        temp['Wert 2'][[' ' in str(i) for i in temp['Wert 2']]]=[i[:i.index(' ')] for i in temp['Wert 2'][[' ' in str(i) for i in temp['Wert 2']]]]
        vit['Wert 2'][idx]=temp['Wert 2']
        temp['Wert 2 Einheit']='mmHg'
        vit['Wert 2 Einheit'][idx]=temp['Wert 2 Einheit']
        
        # 'Bezeichnung zu Wert 2'==Menge
        idx = vit[:][vit['Bezeichnung zu Wert 2']=='Menge'].index
        temp = vit.loc[idx]
        temp['Wert 2'][temp['Wert 2']==' g ']='0'
        temp['Wert 2'][[' ' in str(i) for i in temp['Wert 2']]]=[i[:i.index(' ')] for i in temp['Wert 2'][[' ' in str(i) for i in temp['Wert 2']]]]
        vit['Wert 2'][idx]=temp['Wert 2']
        temp['Wert 2 Einheit']='g'
        vit['Wert 2 Einheit'][idx]=temp['Wert 2 Einheit']
                  
        ### Extension of vit, additional columns  
        # 'Bezeichnung zu Wert 1'==Freitext 5/6/8 --> medication as extra column
        vit_append=pd.DataFrame(columns=vit.columns)
        for i in range(vit['Wert 1'].shape[0]):
            if 'Piri' in str(vit['Wert 1'].values[i]) and (vit['Bezeichnung zu Wert 1'].values[i]=='Freitext 5' or vit['Bezeichnung zu Wert 1'].values[i]=='Freitext 6' or vit['Bezeichnung zu Wert 1'].values[i]=='Freitext 8'):
                app=vit.values[i]
                val=app[2]  # Wert 1 
                newval = str(val)[:val.index('m')]
                app[2] = newval
                app[3] = 'mg'
                app[4] = 'Piritramid_24h'
                vit_append.loc[len(vit_append)]=app
                #print(val)
                #print(newval)
                #print(app)
            if 'albumin' in str(vit['Wert 1'].values[i]) and (vit['Bezeichnung zu Wert 1'].values[i]=='Freitext 5' or vit['Bezeichnung zu Wert 1'].values[i]=='Freitext 6' or vit['Bezeichnung zu Wert 1'].values[i]=='Freitext 8'):
                app=vit.values[i]
                val=app[2]  # Wert 1 
                newval = str(val)[val.index(' ')+1:val.index('ml')]
                app[2] = newval
                app[3] = 'ml'
                app[4] = 'Humanalbumin'
                vit_append.loc[len(vit_append)]=app
                #print(val)
                #print(newval)
                #print(app)
            if 'xygesic' in str(vit['Wert 1'].values[i]) and (vit['Bezeichnung zu Wert 1'].values[i]=='Freitext 5' or vit['Bezeichnung zu Wert 1'].values[i]=='Freitext 6' or vit['Bezeichnung zu Wert 1'].values[i]=='Freitext 8'):
                app=vit.values[i]
                val=app[2]  # Wert 1 
                newval = str(val)[:val.index('m')]
                app[2] = newval
                app[3] = 'mg'
                app[4] = 'Oxygesic_24h'
                vit_append.loc[len(vit_append)]=app
                #print(val)
                #print(newval)
                #print(app)
        vit=pd.concat((vit,vit_append),axis=0,ignore_index=True)
        vit[:]['Piritramid_24h'==vit['Bezeichnung zu Wert 1']]
        
        # Magensonde: two new columns: Magensonde_cat and Magensonde_num
        temp = vit[:][vit['Bezeichnung zu Wert 1']=='Magensonde']
        Magensonde_num = vit[:][vit['Bezeichnung zu Wert 1']=='Magensonde'].iloc[:,2].str.extract('(\d+[,.]\d+|\d+)')  
        Magensonde_cat = vit[:][vit['Bezeichnung zu Wert 1']=='Magensonde'].iloc[:,2]
        for j in range(0,9):
            Magensonde_cat[[str(j) in str(i) for i in Magensonde_cat]]=[i.replace(str(j),'') for i in Magensonde_cat[[str(j) in str(i) for i in Magensonde_cat]]]
        Magensonde_cat[[',' in str(i) for i in Magensonde_cat]]=[i.replace(',','') for i in Magensonde_cat[[',' in str(i) for i in Magensonde_cat]]]
        Magensonde_cat[['  '==str(i) for i in Magensonde_cat]]=np.nan
        Magensonde_cat[['   '==str(i) for i in Magensonde_cat]]=np.nan
        Magensonde_cat
        vit_append_num = vit[:][vit['Bezeichnung zu Wert 1']=='Magensonde']
        vit_append_num['Wert 1'] = Magensonde_num
        vit_append_num['Bezeichnung zu Wert 1'] = 'Magensonde_num'
        vit_append_num = vit_append_num[:][pd.notna(vit_append_num['Wert 1'])]
        vit_append_cat = vit[:][vit['Bezeichnung zu Wert 1']=='Magensonde']
        vit_append_cat['Wert 1'] = Magensonde_cat
        vit_append_cat['Bezeichnung zu Wert 1'] = 'Magensonde_cat'
        vit_append_cat['Wert 1 Einheit'] = np.nan
        vit_append_cat = vit_append_cat[:][pd.notna(vit_append_cat['Wert 1'])]
        vit=pd.concat((vit,vit_append_num,vit_append_cat),axis=0,ignore_index=True)
        
        # Sensorwechsel: two new columns: Sensorwechsel_cat and Sensorwechsel_num
        temp = vit[:][vit['Bezeichnung zu Wert 1']=='Sensorwechsel']
        Sensorwechsel_num = vit[:][vit['Bezeichnung zu Wert 1']=='Sensorwechsel'].iloc[:,2].str.extract('(\d+[,.]\d+|\d+)')
        Sensorwechsel_cat = vit[:][vit['Bezeichnung zu Wert 1']=='Sensorwechsel'].iloc[:,2]
        for j in range(0,9):
            Sensorwechsel_cat[[str(j) in str(i) for i in Sensorwechsel_cat]]=[i.replace(str(j),'') for i in Sensorwechsel_cat[[str(j) in str(i) for i in Sensorwechsel_cat]]]
        Sensorwechsel_cat[[',' in str(i) for i in Sensorwechsel_cat]]=[i.replace(',','') for i in Sensorwechsel_cat[[',' in str(i) for i in Sensorwechsel_cat]]]
        Sensorwechsel_cat[['  '==str(i) for i in Sensorwechsel_cat]]=np.nan
        Sensorwechsel_cat[[' '==str(i) for i in Sensorwechsel_cat]]=np.nan
        Sensorwechsel_cat
        vit_append_num = vit[:][vit['Bezeichnung zu Wert 1']=='Sensorwechsel']
        vit_append_num['Wert 1'] = Sensorwechsel_num
        vit_append_num['Bezeichnung zu Wert 1'] = 'Sensorwechsel_num'
        vit_append_num = vit_append_num[:][pd.notna(vit_append_num['Wert 1'])]
        vit_append_cat = vit[:][vit['Bezeichnung zu Wert 1']=='Sensorwechsel']
        vit_append_cat['Wert 1'] = Sensorwechsel_cat
        vit_append_cat['Bezeichnung zu Wert 1'] = 'Sensorwechsel_cat'
        vit_append_cat['Wert 1 Einheit'] = np.nan
        vit_append_cat = vit_append_cat[:][pd.notna(vit_append_cat['Wert 1'])]
        vit=pd.concat((vit,vit_append_num,vit_append_cat),axis=0,ignore_index=True)
        
        # column 'Number case' as string 
        vit['Number case'] = vit['Number case'].astype('int')
        
        # numerical variables/values: , to . and save as float if possible 
        # idx = vit[:][(vit['Bezeichnung zu Wert 1']=='Temperatur') | (vit['Bezeichnung zu Wert 1']=='R_Schmerz') | (vit['Bezeichnung zu Wert 1']=='RR (sys)') | 
        #              (vit['Bezeichnung zu Wert 1']=='Puls') | (vit['Bezeichnung zu Wert 1']=='Größe') | (vit['Bezeichnung zu Wert 1']=='Gewicht') | 
        #              (vit['Bezeichnung zu Wert 1']=='BMI') | (vit['Bezeichnung zu Wert 1']=='B_Schmerz') | (vit['Bezeichnung zu Wert 1']=='Ausfuhr Urin') | 
        #              (vit['Bezeichnung zu Wert 1']=='Einfuhr') | (vit['Bezeichnung zu Wert 1']=='Sauerstoffsättigung') | (vit['Bezeichnung zu Wert 1']=='Atemfrequenz') | 
        #              (vit['Bezeichnung zu Wert 1']=='Drainage1') | (vit['Bezeichnung zu Wert 1']=='Drainage2') | (vit['Bezeichnung zu Wert 1']=='Größe_PA') | 
        #              (vit['Bezeichnung zu Wert 1']=='Muttermilch') | (vit['Bezeichnung zu Wert 1']=='Kopfumfang') | (vit['Bezeichnung zu Wert 1']=='Wärmebett') | 
        #              (vit['Bezeichnung zu Wert 1']=='Trinkprotokoll') | (vit['Bezeichnung zu Wert 1']=='Flüssigkeitsbilanz') | (vit['Bezeichnung zu Wert 1']=='Piritramid_24h') |
        #              (vit['Bezeichnung zu Wert 1']=='Oxygesic_24h') | (vit['Bezeichnung zu Wert 1']=='Magensonde_num') | (vit['Bezeichnung zu Wert 1']=='Sensorwechsel_num')].index
        # temp = vit.loc[idx]
        # temp2 = temp['Wert 1']
        # temp2[[',' in str(i) for i in temp2]] = [str(i).replace(',','.') for i in temp2[[',' in str(i) for i in temp2]]]
        # temp['Wert 1'] = temp2.astype('float')
        # vit['Wert 1'][idx]=temp['Wert 1']
        # #
        # idx = vit[:][(vit['Bezeichnung zu Wert 2']=='Menge') | (vit['Bezeichnung zu Wert 2']=='RR')].index
        # temp = vit.loc[idx]
        # temp2 = temp['Wert 2']
        # temp2[[',' in str(i) for i in temp2]] = [str(i).replace(',','.') for i in temp2[[',' in str(i) for i in temp2]]]
        # temp['Wert 2'] = temp2.astype('float')
        # vit['Wert 2'][idx]=temp['Wert 2']
        # #
        # idx = vit[:][(vit['Bezeichnung zu Wert 1']=='Sauerstoffgabe') | (vit['Bezeichnung zu Wert 1']=='Magensonde') | (vit['Bezeichnung zu Wert 1']=='Sensorwechsel')].index
        # temp = vit.loc[idx]
        # temp2 = temp['Wert 1']
        # temp2[[',' in str(i) for i in temp2]] = [str(i).replace(',','.') for i in temp2[[',' in str(i) for i in temp2]]]
        # temp['Wert 1'] = temp2
        # vit['Wert 1'][idx]=temp['Wert 1']
        
        # remove gaps at the end      
        vit['Wert 1'] = vit['Wert 1'].apply(lambda x : x.rstrip() if type(x)==str else x )
        vit['Wert 2'] = vit['Wert 2'].apply(lambda x : x.rstrip() if type(x)==str else x )
        
        # ex. / ex --> ex
        vit['Wert 1'][['ex.' in str(i) for i in vit['Wert 1']]] = [str(i).replace('ex.','ex') for i in vit['Wert 1'][['ex.' in str(i) for i in vit['Wert 1']]]]      
        vit['Wert 1']['Ex'==vit['Wert 1']] = [str(i).replace('Ex','ex') for i in vit['Wert 1']['Ex'==vit['Wert 1']]]      
        vit['Wert 1'][['EX' in str(i) for i in vit['Wert 1']]] = [str(i).replace('EX','ex') for i in vit['Wert 1'][['EX' in str(i) for i in vit['Wert 1']]]]      
        
        # x / X --> X
        vit['Wert 1']['x'==vit['Wert 1']] = [str(i).replace('x','X') for i in vit['Wert 1']['x'==vit['Wert 1']]]      
                
        # o.B. / ob / OB --> oB
        vit['Wert 1']['ob'==vit['Wert 1']] = [str(i).replace('ob','oB') for i in vit['Wert 1']['ob'==vit['Wert 1']]]
        vit['Wert 1']['OB'==vit['Wert 1']] = [str(i).replace('OB','oB') for i in vit['Wert 1']['OB'==vit['Wert 1']]]
        vit['Wert 1'][['o.B.' in str(i) for i in vit['Wert 1']]] = [str(i).replace('o.B.','oB') for i in vit['Wert 1'][['o.B.' in str(i) for i in vit['Wert 1']]]]      
        vit['Wert 1'][['o.b' in str(i) for i in vit['Wert 1']]] = [str(i).replace('o.b','oB') for i in vit['Wert 1'][['o.b' in str(i) for i in vit['Wert 1']]]]      
        vit['Wert 1'][['o:B' in str(i) for i in vit['Wert 1']]] = [str(i).replace('o:B','oB') for i in vit['Wert 1'][['o:B' in str(i) for i in vit['Wert 1']]]]      
        vit['Wert 1'][['O.B' in str(i) for i in vit['Wert 1']]] = [str(i).replace('O.B','oB') for i in vit['Wert 1'][['O.B' in str(i) for i in vit['Wert 1']]]]      
        vit['Wert 1'][['o.B' in str(i) for i in vit['Wert 1']]] = [str(i).replace('o.B','oB') for i in vit['Wert 1'][['o.B' in str(i) for i in vit['Wert 1']]]]      
        vit['Wert 1']['Ob'==vit['Wert 1']] = [str(i).replace('Ob','oB') for i in vit['Wert 1']['Ob'==vit['Wert 1']]]
        
        # ko / KO / Ko / usw. --> Kontrolle
        vit['Wert 1']['ko'==vit['Wert 1']] = [str(i).replace('ko','Kontrolle') for i in vit['Wert 1']['ko'==vit['Wert 1']]]
        vit['Wert 1']['Ko'==vit['Wert 1']] = [str(i).replace('Ko','Kontrolle') for i in vit['Wert 1']['Ko'==vit['Wert 1']]]
        vit['Wert 1']['KO'==vit['Wert 1']] = [str(i).replace('KO','Kontrolle') for i in vit['Wert 1']['KO'==vit['Wert 1']]]
        vit['Wert 1']['Kontr'==vit['Wert 1']] = [str(i).replace('Kontr','Kontrolle') for i in vit['Wert 1']['Kontr'==vit['Wert 1']]]
        vit['Wert 1']['Kontr.'==vit['Wert 1']] = [str(i).replace('Kontr.','Kontrolle') for i in vit['Wert 1']['Kontr.'==vit['Wert 1']]]
        vit['Wert 1']['Kont.'==vit['Wert 1']] = [str(i).replace('Kont.','Kontrolle') for i in vit['Wert 1']['Kont.'==vit['Wert 1']]]
        vit['Wert 1']['kontrolle'==vit['Wert 1']] = [str(i).replace('kontrolle','Kontrolle') for i in vit['Wert 1']['kontrolle'==vit['Wert 1']]]
        vit['Wert 1']['kontr.'==vit['Wert 1']] = [str(i).replace('kontr.','Kontrolle') for i in vit['Wert 1']['kontr.'==vit['Wert 1']]]
        vit['Wert 1']['kon.'==vit['Wert 1']] = [str(i).replace('kon.','Kontrolle') for i in vit['Wert 1']['kon.'==vit['Wert 1']]]
        vit['Wert 1']['kont.'==vit['Wert 1']] = [str(i).replace('kont.','Kontrolle') for i in vit['Wert 1']['kont.'==vit['Wert 1']]]
        vit['Wert 1']['kjo'==vit['Wert 1']] = [str(i).replace('kjo','Kontrolle') for i in vit['Wert 1']['kjo'==vit['Wert 1']]]
        vit['Wert 1']['ko.'==vit['Wert 1']] = [str(i).replace('ko.','Kontrolle') for i in vit['Wert 1']['ko.'==vit['Wert 1']]]
        vit['Wert 1']['kont'==vit['Wert 1']] = [str(i).replace('kont','Kontrolle') for i in vit['Wert 1']['kont'==vit['Wert 1']]]
        vit['Wert 1']['ko60'==vit['Wert 1']] = [str(i).replace('ko60','Kontrolle 60') for i in vit['Wert 1']['ko60'==vit['Wert 1']]]
        vit['Wert 1']['ko500'==vit['Wert 1']] = [str(i).replace('ko500','Kontrolle 500') for i in vit['Wert 1']['ko500'==vit['Wert 1']]]
        vit['Wert 1']['Xko'==vit['Wert 1']] = [str(i).replace('Xko','X Kontrolle') for i in vit['Wert 1']['Xko'==vit['Wert 1']]]
        vit['Wert 1']['Ko.'==vit['Wert 1']] = [str(i).replace('Ko.','Kontrolle') for i in vit['Wert 1']['Ko.'==vit['Wert 1']]]
        vit['Wert 1']['ko.'==vit['Wert 1']] = [str(i).replace('ko.','Kontrolle') for i in vit['Wert 1']['ko.'==vit['Wert 1']]]
        vit['Wert 1']['Kontroller'==vit['Wert 1']] = [str(i).replace('Kontroller','Kontrolle') for i in vit['Wert 1']['Kontroller'==vit['Wert 1']]]
        vit['Wert 1'][['ko ' in str(i) for i in vit['Wert 1']]] = [str(i).replace('ko ','Kontrolle ') for i in vit['Wert 1'][['ko ' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['ko,' in str(i) for i in vit['Wert 1']]] = [str(i).replace('ko,','Kontrolle,') for i in vit['Wert 1'][['ko,' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['ko+' in str(i) for i in vit['Wert 1']]] = [str(i).replace('ko+','Kontrolle +') for i in vit['Wert 1'][['ko+' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][[' kontrolle' in str(i) for i in vit['Wert 1']]] = [str(i).replace(' kontrolle',' Kontrolle ') for i in vit['Wert 1'][[' kontrolle' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['Ko.' in str(i) for i in vit['Wert 1']]] = [str(i).replace('Ko.','Kontrolle ') for i in vit['Wert 1'][['Ko.' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['Kontr.' in str(i) for i in vit['Wert 1']]] = [str(i).replace('Kontr.','Kontrolle ') for i in vit['Wert 1'][['Kontr.' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][[' ko' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),str(i)+'.') for i in vit['Wert 1'][[' ko' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][[' ko.' in str(i) for i in vit['Wert 1']]] = [str(i).replace(' ko.',' Kontrolle') for i in vit['Wert 1'][[' ko.' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][[' ko' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),str(i)[0:-1]) for i in vit['Wert 1'][[' ko' in str(i) for i in vit['Wert 1']]]]  
        
        # Erythrozyten-Konz. --> X
        vit['Wert 1'][['EK 276' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['EK 276' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['EK!276' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['EK!276' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['EK! 276' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['EK! 276' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['EK276' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['EK276' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['EK 501' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['EK 501' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['Ek 501' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['Ek 501' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['EK501' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['EK501' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['EK !276' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['EK !276' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['EK: 276' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['EK: 276' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['276-501' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['276-501' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['276 501' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['276 501' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['276501' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['276501' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['275-501' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['275-501' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['501' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['501' in str(i) for i in vit['Wert 1']]]]  
        vit['Wert 1'][['TK276' in str(i) for i in vit['Wert 1']]] = [str(i).replace(str(i),'X') for i in vit['Wert 1'][['TK276' in str(i) for i in vit['Wert 1']]]]  
       
        # Zusammenfassungen in Hämodialyse und Drainage
        idx = vit[:][(vit['Bezeichnung zu Wert 1']=='Drainage') | (vit['Bezeichnung zu Wert 1']=='Hämodialyse')].index
        temp = vit.loc[idx]
        temp['Wert 1'][['entfernt' in str(i) for i in temp['Wert 1']]] = [str(i).replace('entfernt','ex') for i in temp['Wert 1'][['entfernt' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['Entfernt' in str(i) for i in temp['Wert 1']]] = [str(i).replace('Entfernt','ex') for i in temp['Wert 1'][['Entfernt' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['enfernt' in str(i) for i in temp['Wert 1']]] = [str(i).replace('enfernt','ex') for i in temp['Wert 1'][['enfernt' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['entf' in str(i) for i in temp['Wert 1']]] = [str(i).replace('entf','ex') for i in temp['Wert 1'][['entf' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['gezogen' in str(i) for i in temp['Wert 1']]] = [str(i).replace('gezogen','ex') for i in temp['Wert 1'][['gezogen' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['ent' in str(i) for i in temp['Wert 1']]] = [str(i).replace('ent','ex') for i in temp['Wert 1'][['ent' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['Gezogen' in str(i) for i in temp['Wert 1']]] = [str(i).replace('Gezogen','ex') for i in temp['Wert 1'][['Gezogen' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['abgestöpselt' in str(i) for i in temp['Wert 1']]] = [str(i).replace('abgestöpselt','ex') for i in temp['Wert 1'][['abgestöpselt' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['abgesetzt' in str(i) for i in temp['Wert 1']]] = [str(i).replace('abgesetzt','ex') for i in temp['Wert 1'][['abgesetzt' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['ex.' in str(i) for i in temp['Wert 1']]] = [str(i).replace('ex.','ex') for i in temp['Wert 1'][['ex.' in str(i) for i in temp['Wert 1']]]]       
        temp['Wert 1'][['ern.' in str(i) for i in temp['Wert 1']]] = [str(i).replace('ern.','ern') for i in temp['Wert 1'][['ern.' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['ern' in str(i) for i in temp['Wert 1']]] = [str(i).replace('ern','erneuert') for i in temp['Wert 1'][['ern' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['erneuerteuert' in str(i) for i in temp['Wert 1']]] = [str(i).replace('erneuerteuert','erneuert') for i in temp['Wert 1'][['erneuerteuert' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['erneuert.' in str(i) for i in temp['Wert 1']]] = [str(i).replace('erneuert.','erneuert') for i in temp['Wert 1'][['erneuert.' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['gewechselt' in str(i) for i in temp['Wert 1']]] = [str(i).replace('gewechselt','Wechsel') for i in temp['Wert 1'][['gewechselt' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['gew.' in str(i) for i in temp['Wert 1']]] = [str(i).replace('gew.','Wechsel') for i in temp['Wert 1'][['gew.' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['gewechset.' in str(i) for i in temp['Wert 1']]] = [str(i).replace('gewechset.','Wechsel') for i in temp['Wert 1'][['gewechset.' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['Gewechselt' in str(i) for i in temp['Wert 1']]] = [str(i).replace('Gewechselt','Wechsel') for i in temp['Wert 1'][['Gewechselt' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['Gewechsellt' in str(i) for i in temp['Wert 1']]] = [str(i).replace('Gewechsellt','Wechsel') for i in temp['Wert 1'][['Gewechsellt' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][[' wechsel' in str(i) for i in temp['Wert 1']]] = [str(i).replace(' wechsel',' Wechsel') for i in temp['Wert 1'][[' wechsel' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1']['wechsel'==temp['Wert 1']] = [str(i).replace('wechsel','Wechsel') for i in temp['Wert 1']['wechsel'==temp['Wert 1']]]
        temp['Wert 1'][['blütig' in str(i) for i in temp['Wert 1']]] = [str(i).replace('blütig','blutig') for i in temp['Wert 1'][['blütig' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['Soog' in str(i) for i in temp['Wert 1']]] = [str(i).replace('Soog','Sog') for i in temp['Wert 1'][['Soog' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['sog' in str(i) for i in temp['Wert 1']]] = [str(i).replace('sog','Sog') for i in temp['Wert 1'][['sog' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['redyrob' in str(i) for i in temp['Wert 1']]] = [str(i).replace('redyrob','Redyrob') for i in temp['Wert 1'][['redyrob' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['Redy Rob' in str(i) for i in temp['Wert 1']]] = [str(i).replace('Redy Rob','Redyrob') for i in temp['Wert 1'][['Redy Rob' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['Redy-Rob' in str(i) for i in temp['Wert 1']]] = [str(i).replace('Redy-Rob','Redyrob') for i in temp['Wert 1'][['Redy-Rob' in str(i) for i in temp['Wert 1']]]]  
        temp['Wert 1'][['redy rop' in str(i) for i in temp['Wert 1']]] = [str(i).replace('redy rop','Redyrob') for i in temp['Wert 1'][['redy rop' in str(i) for i in temp['Wert 1']]]]  
        vit['Wert 1'][idx]=temp['Wert 1'] 
       
        # Größe_PA = Größe 
        vit['Bezeichnung zu Wert 1'][vit['Bezeichnung zu Wert 1']=='Größe_PA'] = 'Größe'
        
        # remove BMI > 100 
        idx = vit[:][(vit['Bezeichnung zu Wert 1']=='BMI')].index
        temp = vit.loc[idx] 
        temp2 = temp['Wert 1']
        temp2[[',' in str(i) for i in temp2]] = [str(i).replace(',','.') for i in temp2[[',' in str(i) for i in temp2]]]
        temp['Wert 1'] = temp2.astype('float')
        vit['Wert 1'][idx]=temp['Wert 1']
        vit = vit.drop(temp['Number case'][temp['Wert 1']>100].index)
        
        # Temperatur Range 31-43
        idx = vit[:][(vit['Bezeichnung zu Wert 1']=='Temperatur')].index
        temp = vit.loc[idx] 
        temp2 = temp['Wert 1']
        temp2[[',' in str(i) for i in temp2]] = [str(i).replace(',','.') for i in temp2[[',' in str(i) for i in temp2]]]
        temp['Wert 1'] = temp2.astype('float')
        vit['Wert 1'][idx]=temp['Wert 1']
        vit = vit.drop(temp['Number case'][temp['Wert 1']>43].index)
        vit = vit.drop(temp['Number case'][temp['Wert 1']<31].index)
        
        # remove RR (sys) / RR < 10 or >= 1000
        idx = vit[:][vit['Bezeichnung zu Wert 1']=='RR (sys)'].index
        temp = vit.loc[idx] 
        vit = vit.drop(temp[:][temp['Wert 1'].astype('float')<10].index, axis=0)
        vit = vit.drop(temp[:][temp['Wert 1'].astype('float')>=1000].index, axis=0)
        idx = vit[:][vit['Bezeichnung zu Wert 2']=='RR (dia)'].index
        temp = vit.loc[idx] 
        vit = vit.drop(temp[:][temp['Wert 2'].astype('float')<10].index, axis=0)
        vit = vit.drop(temp[:][temp['Wert 2'].astype('float')>=1000].index, axis=0)
        
        ### reference values
        vit.insert(5,'lower_ref_1',np.nan)
        vit.insert(6,'upper_ref_1',np.nan)
        vit.insert(10,'lower_ref_2',np.nan)
        vit.insert(11,'upper_ref_2',np.nan)
        for i in range(vit['Number case'].shape[0]):
            age = base['Age'][base['Number case']==vit['Number case'].iloc[i]]
            if age.isnull().any()==False:
                age =  int(age)
                if vit['Bezeichnung zu Wert 1'].iloc[i]=='Puls':
                    if (age==1 or age==2):
                        vit['lower_ref_1'].iloc[i]=98
                        vit['upper_ref_1'].iloc[i]=140
                    if (age==3 or age==4 or age==5):
                        vit['lower_ref_1'].iloc[i]=80
                        vit['upper_ref_1'].iloc[i]=120
                    if (age==6 or age==7 or age==8 or age==9 or age==10 or age==11):
                        vit['lower_ref_1'].iloc[i]=75
                        vit['upper_ref_1'].iloc[i]=118
                    if age>=12:
                        vit['lower_ref_1'].iloc[i]=60
                        vit['upper_ref_1'].iloc[i]=100
                if vit['Bezeichnung zu Wert 1'].iloc[i]=='Atemfrequenz':
                    if (age==1 or age==2):
                        vit['lower_ref_1'].iloc[i]=22
                        vit['upper_ref_1'].iloc[i]=37
                    if (age==3 or age==4 or age==5):
                        vit['lower_ref_1'].iloc[i]=20
                        vit['upper_ref_1'].iloc[i]=28
                    if (age==6 or age==7 or age==8 or age==9 or age==10 or age==11):
                        vit['lower_ref_1'].iloc[i]=18
                        vit['upper_ref_1'].iloc[i]=25
                    if (age==12 or age==13 or age==14 or age==15):
                        vit['lower_ref_1'].iloc[i]=12
                        vit['upper_ref_1'].iloc[i]=20
                    if age>=16:
                        vit['lower_ref_1'].iloc[i]=12
                        vit['upper_ref_1'].iloc[i]=18
                if vit['Bezeichnung zu Wert 1'].iloc[i]=='Sauerstoffsättigung':
                    vit['lower_ref_1'].iloc[i]=95
                    vit['upper_ref_1'].iloc[i]=100
                    if age<18:
                        vit['lower_ref_1'].iloc[i]=90
                if vit['Bezeichnung zu Wert 1'].iloc[i]=='RR (sys)':
                    if (age==1 or age==2):
                        vit['lower_ref_1'].iloc[i]=86
                        vit['upper_ref_1'].iloc[i]=106
                    if (age==3 or age==4 or age==5):
                        vit['lower_ref_1'].iloc[i]=89
                        vit['upper_ref_1'].iloc[i]=112
                    if (age==6 or age==7 or age==8 or age==9):
                        vit['lower_ref_1'].iloc[i]=97
                        vit['upper_ref_1'].iloc[i]=115
                    if (age==10 or age==11):
                        vit['lower_ref_1'].iloc[i]=102
                        vit['upper_ref_1'].iloc[i]=120
                    if (age==12 or age==13 or age==14 or age==15):
                        vit['lower_ref_1'].iloc[i]=110
                        vit['upper_ref_1'].iloc[i]=131
                    if age>=16:
                        vit['lower_ref_1'].iloc[i]=90
                        vit['upper_ref_1'].iloc[i]=120
                if vit['Bezeichnung zu Wert 2'].iloc[i]=='RR (dia)':
                    if (age==1 or age==2):
                        vit['lower_ref_2'].iloc[i]=42
                        vit['upper_ref_2'].iloc[i]=63
                    if (age==3 or age==4 or age==5):
                        vit['lower_ref_2'].iloc[i]=46
                        vit['upper_ref_2'].iloc[i]=72
                    if (age==6 or age==7 or age==8 or age==9):
                        vit['lower_ref_2'].iloc[i]=57
                        vit['upper_ref_2'].iloc[i]=76
                    if (age==10 or age==11):
                        vit['lower_ref_2'].iloc[i]=61
                        vit['upper_ref_2'].iloc[i]=80
                    if (age==12 or age==13 or age==14 or age==15):
                        vit['lower_ref_2'].iloc[i]=64
                        vit['upper_ref_2'].iloc[i]=83
                    if age>=16:
                        vit['lower_ref_2'].iloc[i]=60
                        vit['upper_ref_2'].iloc[i]=90
                if vit['Bezeichnung zu Wert 1'].iloc[i]=='Temperatur':
                    if vit['Wert 2'].iloc[i]=='rectal':
                        vit['lower_ref_1'].iloc[i]=36.6
                        vit['upper_ref_1'].iloc[i]=38.0
                    if vit['Wert 2'].iloc[i]=='Ohr':
                        vit['lower_ref_1'].iloc[i]=35.8
                        vit['upper_ref_1'].iloc[i]=38.0
                    if vit['Wert 2'].iloc[i]=='sublingual':
                        vit['lower_ref_1'].iloc[i]=35.5
                        vit['upper_ref_1'].iloc[i]=37.5
                    if vit['Wert 2'].iloc[i]=='axillär':
                        vit['lower_ref_1'].iloc[i]=36.5
                        vit['upper_ref_1'].iloc[i]=37.5
        
        # all variables: save as string
        vit['Wert 1']=vit['Wert 1'].astype('string')
        vit['Wert 1 Einheit']=vit['Wert 1 Einheit'].astype('string')
        vit['Wert 2']=vit['Wert 2'].astype('string')
        vit['Wert 2 Einheit']=vit['Wert 2 Einheit'].astype('string')
        vit['Bezeichnung zu Wert 1']=vit['Bezeichnung zu Wert 1'].astype('string')
        vit['Bezeichnung zu Wert 2']=vit['Bezeichnung zu Wert 2'].astype('string')
        vit['lower_ref_1']=vit['lower_ref_1'].astype('string')
        vit['lower_ref_2']=vit['lower_ref_2'].astype('string')
        vit['upper_ref_1']=vit['upper_ref_1'].astype('string')
        vit['upper_ref_2']=vit['upper_ref_2'].astype('string')

    vit['Number case']=vit['Number case'].astype('int')
    
    vit['Bemerkung']=vit['Bemerkung'].astype('string')    
    
    vit=vit.sort_index().sort_values(by='Number case', kind='mergesort')
    
    ops=data[data['B']=='009OPS']
    ops.columns=['Number case','Datatype','OPS','Date','OPS Version','Schlüssel','Leitendes Verfahren','Hauptverfahren','Nebenverfahren','Klartext','Erfassungsdatum','Datum der letzten Aenderung','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    ops=ops[['Number case','Date','OPS Version','Schlüssel','Leitendes Verfahren','Hauptverfahren','Nebenverfahren','Klartext','Erfassungsdatum']]
    ops['Erfassungsdatum'] = pd.to_datetime(ops['Erfassungsdatum'],dayfirst=True,errors='coerce')

    ops['Number case']=ops['Number case'].astype('int')
    ops['OPS Version']=ops['OPS Version'].astype('string')
    ops['Schlüssel']=ops['Schlüssel'].astype('string')
    ops['Leitendes Verfahren']=ops['Leitendes Verfahren'].astype('string')
    ops['Hauptverfahren']=ops['Hauptverfahren'].astype('string')
    ops['Nebenverfahren']=ops['Nebenverfahren'].astype('string')
    ops['Klartext']=ops['Klartext'].astype('string')
    ops['Date'] = pd.to_datetime(ops['Date'],dayfirst=True)
    ops=ops.sort_index().sort_values(by='Number case', kind='mergesort')
    ops=ops.drop_duplicates()
    
    drg=data[data['B']=='010DRG']
    drg.columns=['Number case','Datatype','DRG','Date','DRG Version','Value','Klartext','Untere Verweildauer','Mittlere Verweildauer','Obere Verweildauer','ICD Version','ICD Hauptdiagnose','ICD Nebendiagnose 1','ICD Nebendiagnose 2','ICD Nebendiagnose 3','ICD Nebendiagnose 4','ICD Nebendiagnose 5','OPS Version','OPS Verfahren 1','OPS Verfahren 2','OPS Verfahren 3','OPS Verfahren 4','OPS Verfahren 5','OPS Verfahren 6','OPS Verfahren 7','OPS Verfahren 8','OPS Verfahren 9','Erfassungsdatum']
    drg=drg[['Number case','Date','DRG Version','Value','Klartext','ICD Version','ICD Hauptdiagnose','ICD Nebendiagnose 1','ICD Nebendiagnose 2','ICD Nebendiagnose 3','ICD Nebendiagnose 4','ICD Nebendiagnose 5','OPS Version','OPS Verfahren 1','OPS Verfahren 2','OPS Verfahren 3','OPS Verfahren 4','OPS Verfahren 5','OPS Verfahren 6','OPS Verfahren 7','OPS Verfahren 8','OPS Verfahren 9','Erfassungsdatum']]    
    drg['Erfassungsdatum'][drg['Erfassungsdatum'].notnull()]=[i[:i.index('  ')] for i in drg['Erfassungsdatum'][drg['Erfassungsdatum'].notnull()]]
    drg['Erfassungsdatum'] = pd.to_datetime(drg['Erfassungsdatum'],dayfirst=True)
    drg['Date'] = pd.to_datetime(drg['Date'],dayfirst=True)
    drg['Number case']=drg['Number case'].astype('int')
    drg['DRG Version']=drg['DRG Version'].astype('string')
    drg['Value']=drg['Value'].astype('string')
    drg['Klartext']=drg['Klartext'].astype('string')
    drg['ICD Version']=drg['ICD Version'].astype('string')
    drg['ICD Hauptdiagnose']=drg['ICD Hauptdiagnose'].astype('string')
    drg['ICD Nebendiagnose 1']=drg['ICD Nebendiagnose 1'].astype('string')
    drg['ICD Nebendiagnose 2']=drg['ICD Nebendiagnose 2'].astype('string')
    drg['ICD Nebendiagnose 3']=drg['ICD Nebendiagnose 3'].astype('string')
    drg['ICD Nebendiagnose 4']=drg['ICD Nebendiagnose 4'].astype('string')
    drg['ICD Nebendiagnose 5']=drg['ICD Nebendiagnose 5'].astype('string')
    drg['OPS Version']=drg['OPS Version'].astype('string')
    drg['OPS Verfahren 1']=drg['OPS Verfahren 1'].astype('string')
    drg['OPS Verfahren 2']=drg['OPS Verfahren 2'].astype('string')
    drg['OPS Verfahren 3']=drg['OPS Verfahren 3'].astype('string')
    drg['OPS Verfahren 4']=drg['OPS Verfahren 4'].astype('string')
    drg['OPS Verfahren 5']=drg['OPS Verfahren 5'].astype('string')
    drg['OPS Verfahren 6']=drg['OPS Verfahren 6'].astype('string')
    drg['OPS Verfahren 7']=drg['OPS Verfahren 7'].astype('string')
    drg['OPS Verfahren 8']=drg['OPS Verfahren 8'].astype('string')
    drg['OPS Verfahren 9']=drg['OPS Verfahren 9'].astype('string')
    drg=drg.sort_index().sort_values(by='Number case', kind='mergesort')
    drg=drg.drop_duplicates()
    
    goa=data[data['B']=='011GOA']
    goa.columns=['Number case','Datatype','GOAE','Date','Katalog kurz','Katalog lang','Tarifziffer','Menge','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    goa=goa[['Number case','Date','Katalog kurz','Tarifziffer','Menge']]
    goa['Number case']=goa['Number case'].astype('int')
    goa['Menge'][goa['Menge'].notnull()]=[i[:i.index(' ')] for i in goa['Menge'][goa['Menge'].notnull()]]
    goa['Date'] = pd.to_datetime(goa['Date'],dayfirst=True, errors='coerce')
    goa['Katalog kurz']=goa['Katalog kurz'].astype('string')
    goa['Tarifziffer']=goa['Tarifziffer'].astype('string')
    goa['Menge']=goa['Menge'].astype('int')
    goa=goa.sort_index().sort_values(by='Number case', kind='mergesort')
    goa=goa.drop_duplicates()

    ala=data[data['B']=='012ALA']
    ala.columns=['Number case','Datatype','ALARM','Date','Kurzbezeichnung','Langbezeichnung','Gueltigkeit Beginn','Gueltigkeit Ende','Erfassungsdatum','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    ala=ala[['Number case','Date','Kurzbezeichnung','Langbezeichnung','Gueltigkeit Beginn','Gueltigkeit Ende','Erfassungsdatum']]
    ala['Number case']=ala['Number case'].astype('int')
    ala['Erfassungsdatum'][ala['Erfassungsdatum'].notnull()]=[i[:i.index('  ')] for i in ala['Erfassungsdatum'][ala['Erfassungsdatum'].notnull()]]
    ala['Erfassungsdatum'] = pd.to_datetime(ala['Erfassungsdatum'],dayfirst=True, errors='coerce')
    ala['Date'] = pd.to_datetime(ala['Date'],dayfirst=True, errors='coerce')
    ala['Kurzbezeichnung']=ala['Kurzbezeichnung'].astype('string')
    ala['Langbezeichnung']=ala['Langbezeichnung'].astype('string')
    ala['Gueltigkeit Beginn'] = pd.to_datetime(ala['Gueltigkeit Beginn'],dayfirst=True, errors='coerce')
    ala['Gueltigkeit Ende'] = pd.to_datetime(ala['Gueltigkeit Ende'],dayfirst=True, errors='coerce')
    ala=ala.sort_index().sort_values(by='Number case', kind='mergesort')
    ala['Gueltigkeit Ende'][ala['Gueltigkeit Ende']>np.datetime64('2050','Y')]=np.datetime64('nat')
    ala=ala.drop_duplicates()        
    
    pkz=data[data['B']=='013PKZ']
    pkz.columns=['Number case','Datatype','PFLKZ','Date','Kurzbezeichnung','Langbezeichnung','Gueltigkeit Beginn','Gueltigkeit Ende','Erfassungsdatum','Datum der letzten Änderung','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    pkz=pkz[['Number case','Date','Kurzbezeichnung','Langbezeichnung','Gueltigkeit Beginn','Gueltigkeit Ende']]
    pkz['Number case']=pkz['Number case'].astype('int')
    pkz['Date'] = pd.to_datetime(pkz['Date'],dayfirst=True, errors='coerce')
    pkz['Kurzbezeichnung']=pkz['Kurzbezeichnung'].astype('string')
    pkz['Langbezeichnung']=pkz['Langbezeichnung'].astype('string')
    pkz['Gueltigkeit Beginn'] = pd.to_datetime(pkz['Gueltigkeit Beginn'],dayfirst=True, errors='coerce')
    pkz['Gueltigkeit Ende'] = pd.to_datetime(pkz['Gueltigkeit Ende'],dayfirst=True, errors='coerce')
    pkz=pkz.sort_index().sort_values(by='Number case', kind='mergesort')
    pkz=pkz.drop_duplicates()

    tim=data[data['B']=='014TIM']
    tim.columns=['Number case','Datatype','Zeitpunkt','Date','Kurzbezeichnung','Langbezeichnung','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    tim=tim[['Number case','Date','Kurzbezeichnung','Langbezeichnung']]
    tim['Number case']=tim['Number case'].astype('int')
    tim['Langbezeichnung'][tim['Langbezeichnung'].notnull()]=[i[:i.index('  ')] for i in tim['Langbezeichnung'][tim['Langbezeichnung'].notnull()]]
    tim['Date'] = pd.to_datetime(tim['Date'],dayfirst=True, errors='coerce')
    tim['Kurzbezeichnung']=tim['Kurzbezeichnung'].astype('string')
    tim['Langbezeichnung']=tim['Langbezeichnung'].astype('string')
    tim=tim.sort_index().sort_values(by='Number case', kind='mergesort')
    tim=tim.drop_duplicates()
    
    tri=data[data['B']=='015TRI']
    tri.columns=['Number case','Datatype','Triage','Date','Triagestatus vor der Aenderung','Triagestatus durch jetzige Aenderung','Triageminuten','Einheit','Triagestatus','Erfassungsdatum','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    tri=tri[['Number case','Date','Triagestatus vor der Aenderung','Triagestatus durch jetzige Aenderung','Triageminuten','Einheit','Triagestatus','Erfassungsdatum']]
    tri['Number case']=tri['Number case'].astype('int')
    tri['Date'] = pd.to_datetime(tri['Date'],dayfirst=True, errors='coerce')
    tri['Erfassungsdatum'][tri['Erfassungsdatum'].notnull()]=[i[:i.index('  ')] for i in tri['Erfassungsdatum'][tri['Erfassungsdatum'].notnull()]]
    tri['Erfassungsdatum'] = pd.to_datetime(tri['Erfassungsdatum'],dayfirst=True, errors='coerce')
    tri['Triagestatus vor der Aenderung']=tri['Triagestatus vor der Aenderung'].astype('string')
    tri['Triagestatus durch jetzige Aenderung']=tri['Triagestatus durch jetzige Aenderung'].astype('string')
    tri['Triagestatus']=tri['Triagestatus'].astype('string')
    tri['Triageminuten']=tri['Triageminuten'].astype(int)
    tri=tri.sort_index().sort_values(by='Number case', kind='mergesort')
    tri=tri.drop_duplicates()
    
    att=data[data['B']=='016ATT']
    att.columns=['Number case','Datatype','ATT','Attribut','Beginn','Ende','Erfassungsdatum','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    att=att[['Number case','Attribut','Beginn','Ende','Erfassungsdatum']]
    att['Number case']=att['Number case'].astype('int')
    att['Erfassungsdatum'][att['Erfassungsdatum'].notnull()]=[i[:i.index('  ')] for i in att['Erfassungsdatum'][att['Erfassungsdatum'].notnull()]]
    att['Erfassungsdatum'] = pd.to_datetime(att['Erfassungsdatum'],dayfirst=True, errors='coerce')
    att['Attribut']=att['Attribut'].astype('string')
    att['Beginn'] = pd.to_datetime(att['Beginn'],dayfirst=True, errors='coerce')
    att['Ende'] = pd.to_datetime(att['Ende'],dayfirst=True, errors='coerce')
    att=att.sort_index().sort_values(by='Number case', kind='mergesort')
    att=att.drop_duplicates()

    medgab=data[data['B']=='017MED']
    medgab.columns=['Number case','Datatype','Medikation_text','Date','ATC','Name_med','Menge','Einheit','Uhrzeit','Status','Status_Klartext','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan','nan']
    medgab=medgab[['Number case','Date','ATC','Name_med','Menge','Einheit','Uhrzeit','Status','Status_Klartext']]
    medgab['Status_Klartext'][['  ' in str(i) for i in medgab['Status_Klartext']]]=[i[:i.index('  ')] for i in medgab['Status_Klartext'] if '  ' in str(i)]
    medgab['Number case']=medgab['Number case'].astype('int')
    medgab['Date']=pd.to_datetime(medgab['Date'].values + ' ' + medgab['Uhrzeit'].values,dayfirst=True, errors='coerce')
    medgab=medgab.drop('Uhrzeit',1)
    medgab['ATC']=medgab['ATC'].astype('string')
    medgab['Ingredient'] = np.zeros((medgab['Name_med'].shape[0],1))
    medgab['Ingredient'][['[' in i for i in medgab['Name_med']]] = [i[i.index('[')+1:-1] for i in medgab['Name_med'][['[' in i for i in medgab['Name_med']]]]
    
    #This is a list comprehension. it takes the i-th values of the Column with the names of the medications and then takes the ingredient out of it
    medgab['Name_med'][['[' in i for i in medgab['Name_med']]] = [i[:i.index('[')] for i in medgab['Name_med'][['[' in i for i in medgab['Name_med']]]]
    
    medgab['Menge'][[',' in str(i) for i in medgab['Menge']]] = [i.replace(',','.') for i in medgab['Menge'][[',' in str(i) for i in medgab['Menge']]]]
    
    medgab['Menge'] = medgab['Menge'].astype('float')
    
    if medgab[['Xa' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['Xa' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['Xa' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['0' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['0' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['0' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
        
    if medgab[['09:28' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['09:28' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['09:28' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['12:00' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['12:00' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['12:00' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['12:22' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['12:22' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['12:22' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['08:48' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['08:48' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['08:48' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
            
    if medgab[['14:00' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['14:00' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['14:00' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])


    if medgab[['22:00' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['22:00' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['22:00' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['08:00' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['08:00' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['08:00' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['05:18' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['05:18' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['05:18' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['20:48' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['20:48' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['20:48' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['20:00' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['20:00' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['20:00' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['02:00' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['02:00' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['02:00' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    if medgab[['18:00' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['18:00' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['18:00' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])

    if medgab[['01:35' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['01:35' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['01:35' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
            
    if medgab[['19:20' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['19:20' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['19:20' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])

    if medgab[['15:00' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['15:00' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['15:00' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
            
    if medgab[['01:57' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['01:57' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['01:57' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
            
    if medgab[['23:00' == str(i) for i in medgab['Status']]].shape[0]>0:
        a=medgab[['23:00' == str(i) for i in medgab['Status']]]
        for i in range(a.shape[0]):
            a=medgab[['23:00' == str(i) for i in medgab['Status']]]
            a.index[0]
            medgab=medgab.drop(a.index[0])
    
    medgab['Einheit'] = medgab['Einheit'].astype('string')
    
    list_ATC = []
    medgab['Ingredient'] = [i.strip() for i in medgab['Ingredient']]
    #medgab[medgab['Einheit']=='Stk.']['Einheit']=
    for i in range(medgab.shape[0]):
        if pd.notna(medgab.iloc[i,medgab.columns.get_loc('ATC')]):
        #if not medgab.iloc[i,medgab.columns.get_loc('ATC')]=='A01AB33':
            if medgab.iloc[i,medgab.columns.get_loc('ATC')] in list_ATC:
                #Allgemeine EInheitsumformungen von Stück, tropfen und mikrogramm etc.
                if (str(medgab.iloc[i,medgab.columns.get_loc('Einheit')])=='Stk.' or str(medgab.iloc[i,medgab.columns.get_loc('Einheit')])=='Sprühst.' or str(medgab.iloc[i,medgab.columns.get_loc('Einheit')])=='Hub' or str(medgab.iloc[i,medgab.columns.get_loc('Einheit')])=='Appl.' or str(medgab.iloc[i,medgab.columns.get_loc('Einheit')])=='Amp' or str(medgab.iloc[i,medgab.columns.get_loc('Einheit')])=='Btl' or str(medgab.iloc[i,medgab.columns.get_loc('Einheit')])=='Fl') and find_nth(str(medgab.iloc[i,medgab.columns.get_loc('Ingredient')]),' ',2)!=-1:
                    medgab.iloc[i,medgab.columns.get_loc('Einheit')]=str(medgab.iloc[i,medgab.columns.get_loc('Ingredient')])[find_nth(medgab.iloc[i,medgab.columns.get_loc('Ingredient')],' ',1)+1:find_nth(medgab.iloc[i,medgab.columns.get_loc('Ingredient')],' ',2)]
                    medgab.iloc[i,medgab.columns.get_loc('Menge')] = medgab.iloc[i,medgab.columns.get_loc('Menge')]*float(medgab.iloc[i,medgab.columns.get_loc('Ingredient')][:find_nth(medgab.iloc[i,medgab.columns.get_loc('Ingredient')],' ',1)])
                if medgab.iloc[i,medgab.columns.get_loc('Einheit')]=='Trpf':
                    medgab.iloc[i,medgab.columns.get_loc('Einheit')] = 'ml'
                    medgab.iloc[i,medgab.columns.get_loc('Menge')] = float(medgab.iloc[i,medgab.columns.get_loc('Menge')])/20
                if medgab.iloc[i,medgab.columns.get_loc('Einheit')] == 'µg':
                    medgab.iloc[i,medgab.columns.get_loc('Einheit')] = 'mg'
                    medgab.iloc[i,medgab.columns.get_loc('Menge')] = float(medgab.iloc[i,medgab.columns.get_loc('Menge')])/1000
                
                #Spezialfälle innerhalb der Liste
                if medgab.iloc[i,medgab.columns.get_loc('ATC')] == 'A11CC05':
                    if str(medgab.iloc[i,medgab.columns.get_loc('Einheit')]) == 'I.E.':
                        medgab.iloc[i,medgab.columns.get_loc('Einheit')]=str(medgab.iloc[i,medgab.columns.get_loc('Ingredient')])[find_nth(medgab.iloc[i,medgab.columns.get_loc('Ingredient')],' ',1)+1:find_nth(medgab.iloc[i,medgab.columns.get_loc('Ingredient')],' ',2)]
                        if float(medgab.iloc[i,medgab.columns.get_loc('Menge')]) == 1:
                            medgab.iloc[i,medgab.columns.get_loc('Menge')] = float(medgab.iloc[i,medgab.columns.get_loc('Ingredient')][:find_nth(medgab.iloc[i,medgab.columns.get_loc('Ingredient')],' ',1)])
                        else:
                            medgab.iloc[i,medgab.columns.get_loc('Menge')] = medgab.iloc[i,medgab.columns.get_loc('Menge')]*0.025/1000
                    elif str(medgab.iloc[i,medgab.columns.get_loc('Einheit')]) == 'ml' and (medgab.iloc[i,medgab.columns.get_loc('Ingredient')] == '500 µg Colecalciferol' or medgab.iloc[i,medgab.columns.get_loc('Ingredient')] == '0.5 mg Colecalciferol'):
                        medgab.iloc[i,medgab.columns.get_loc('Einheit')] = 'mg'
                        medgab.iloc[i,medgab.columns.get_loc('Menge')] = medgab.iloc[i,medgab.columns.get_loc('Menge')]*20000*0.025/1000
                
                if medgab.iloc[i,medgab.columns.get_loc('ATC')] == 'R03BA02' and str(medgab.iloc[i,medgab.columns.get_loc('Einheit')])=='ml':
                    medgab.iloc[i,medgab.columns.get_loc('Einheit')] = 'mg'
                    medgab.iloc[i,medgab.columns.get_loc('Menge')] = medgab.iloc[i,medgab.columns.get_loc('Menge')]*float(medgab.iloc[i,medgab.columns.get_loc('Ingredient')][:find_nth(medgab.iloc[i,medgab.columns.get_loc('Ingredient')],' ',1)])/2
                if medgab.iloc[i,medgab.columns.get_loc('ATC')] == 'N03AX14' and str(medgab.iloc[i,medgab.columns.get_loc('Einheit')])=='ml':
                    medgab.iloc[i,medgab.columns.get_loc('Einheit')] = 'mg'
                    if medgab.iloc[i,medgab.columns.get_loc('Menge')]!=1000:
                        medgab.iloc[i,medgab.columns.get_loc('Menge')] = medgab.iloc[i,medgab.columns.get_loc('Menge')]*float(100)
            
            #Spezialfälle außerhalb der Liste
            if medgab.iloc[i,medgab.columns.get_loc('ATC')]=='R05CP':
                medgab.iloc[i,medgab.columns.get_loc('Einheit')]='mg'
                medgab.iloc[i,medgab.columns.get_loc('Menge')]=medgab.iloc[i,medgab.columns.get_loc('Menge')]*300
            if medgab.iloc[i,medgab.columns.get_loc('ATC')]=='L01EC01':
                medgab.iloc[i,medgab.columns.get_loc('Einheit')]='mg'
                medgab.iloc[i,medgab.columns.get_loc('Menge')]=medgab.iloc[i,medgab.columns.get_loc('Menge')]*240
            if medgab.iloc[i,medgab.columns.get_loc('Ingredient')]=='Pankreas-Pulver vom Schwein':
                medgab.iloc[i,medgab.columns.get_loc('Einheit')]='mg'
                medgab.iloc[i,medgab.columns.get_loc('Menge')]=medgab.iloc[i,medgab.columns.get_loc('Menge')]*150
            if medgab.iloc[i,medgab.columns.get_loc('Ingredient')]=='besilat] AbZ 5 mg Tabletten[6.935 mg Amlodipin besilat':
                medgab.iloc[i,medgab.columns.get_loc('Ingredient')]='6.935 mg Amlodipin besilat'
                medgab.iloc[i,medgab.columns.get_loc('Einheit')]='mg'
                medgab.iloc[i,medgab.columns.get_loc('Menge')] = float(medgab.iloc[i,medgab.columns.get_loc('Menge')]) * 6.935
            if medgab.iloc[i,medgab.columns.get_loc('Ingredient')]=='besilat] AbZ 10 mg Tabletten[13.87 mg Amlodipin besilat':
                medgab.iloc[i,medgab.columns.get_loc('Ingredient')]='13.87 mg Amlodipin besilat'
                medgab.iloc[i,medgab.columns.get_loc('Einheit')]='mg'
                medgab.iloc[i,medgab.columns.get_loc('Menge')] = float(medgab.iloc[i,medgab.columns.get_loc('Menge')]) * 13.87
            if medgab.iloc[i,medgab.columns.get_loc('Ingredient')]=='Enzianwurzel, TE o.w.A.|    Primelblüten-Trockenextrakt|    Sauerampferkraut-Trockenextrakt|    Holunderblüten, TE o.w.A.|    Eisenkraut-Trockenextrakt':
                medgab.iloc[i,medgab.columns.get_loc('Einheit')]='mg'
                medgab.iloc[i,medgab.columns.get_loc('Menge')] = float(medgab.iloc[i,medgab.columns.get_loc('Menge')]) * 160
            
                    
            
    medgab['Status'][pd.notna(medgab['Status'])] = medgab['Status'][pd.notna(medgab['Status'])].astype('int')
    medgab['Status_Klartext'] = medgab['Status_Klartext'].astype('string')
    medgab=medgab.sort_index().sort_values(by='Number case', kind='mergesort')
    medgab=medgab.drop_duplicates()

    return base, diag, dauerdiag, medaid, lab, bew, vor, vit, ops, drg, goa, ala, pkz, tim, tri, att, medgab
