install.packages(setdiff(c("tidyverse", "pROC"), rownames(installed.packages())))

library(tidyverse)
library(pROC)

path = 'C:/Users/wendl/Documents/GitHub/GTT-Trigger-tool/Preprocessing_data_cleaning/Version13b/DatensätzeR24'

###Andere KIs

ypred_tod = read.csv(paste(path,'/y_pred_tod_alternativ48.csv',sep=''),header = TRUE,sep = ',')
ytest_tod = read.csv(paste(path,'/ytest_tod_alternativ48.csv',sep=''),header = TRUE,sep = ',')
ypred_its = read.csv(paste(path,'/y_pred_its48.csv',sep=''),header = TRUE,sep = ',')
ytest_its = read.csv(paste(path,'/ytest_its48.csv',sep=''),header = TRUE,sep = ',')
ypred_beat = read.csv(paste(path,'/y_pred_beat48.csv',sep=''),header = TRUE,sep = ',')
ytest_beat = read.csv(paste(path,'/ytest_beat48.csv',sep=''),header = TRUE,sep = ',')

ypred_tod_cat = read.csv(paste(path,'/y_pred_tod_alternativ_cat48.csv',sep=''),header = TRUE,sep = ',')
#ypred_tod_cat = read.csv(paste(path,'/y_pred_tod_alternativ_cat48.csv',sep=''),header = TRUE,sep = ',')
ytest_tod_cat = read.csv(paste(path,'/ytest_tod_alternativ_cat48.csv',sep=''),header = TRUE,sep = ',')
ypred_tod_rf_cat = read.csv(paste(path,'/y_pred_tod_rf_cat48.csv',sep=''),header = TRUE,sep = ',')
ytest_tod_rf_cat = read.csv(paste(path,'/ytest_tod_alternativ_cat48.csv',sep=''),header = TRUE,sep = ',')
ypred_its_cat = read.csv(paste(path,'/y_pred_its_cat48.csv',sep=''),header = TRUE,sep = ',')
#ypred_its_cat = read.csv(paste(path,'/y_pred_its_cat48.csv',sep=''),header = TRUE,sep = ',')
ytest_its_cat = read.csv(paste(path,'/ytest_its_cat48.csv',sep=''),header = TRUE,sep = ',')
#ytest_its_cat = read.csv(paste(path,'/ytest_its_cat48.csv',sep=''),header = TRUE,sep = ',')
ypred_beat_cat = read.csv(paste(path,'/y_pred_beat_cat48.csv',sep=''),header = TRUE,sep = ',')
ytest_beat_cat = read.csv(paste(path,'/ytest_beat_cat48.csv',sep=''),header = TRUE,sep = ',')

##### ROC Kurven mit Konfidenzintervallen #####

plotROC <- function(ytest,ypred){
  ROC = roc(ytest,ypred,ci=TRUE)
  ROC.CI = ci.thresholds(ROC)
  
  plot(ROC,print.auc = TRUE, grid = TRUE, main = '', print.auc.x=0.625, print.auc.y=0.165,print.auc.cex=1.7,cex.axis=1.65,cex.lab=1.8)
}

#Enpunkt Tod (LogReg)
png(file=paste(path,'/proc/ROC_tod_logreg48.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_tod$X0,ypred_tod$X0)
dev.off()

#Endpunkt ITS (XGBoost)
png(file=paste(path,'/proc/ROC_its48.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_its$X0,ypred_its$X0)
dev.off()

#Endpunkt Beatmung (RandomForest)
png(file=paste(path,'/proc/ROC_beat48.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_beat$X0,ypred_beat$X0)
dev.off()

#Enpunkt Tod (LogReg)
png(file=paste(path,'/proc/ROC_tod_logreg_cat48.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_tod_cat$X0,ypred_tod_cat$X0)
dev.off()

roc_todlg=roc(ytest_tod_cat$X0,ypred_tod_cat$X0)
coords(roc_todlg,x="best")

#Enpunkt Tod (RF)
png(file=paste(path,'/proc/ROC_tod_rf_cat48.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_tod_rf_cat$X0,ypred_tod_rf_cat$X0)
dev.off()

#Endpunkt ITS (XGBoost)
png(file=paste(path,'/proc/ROC_its_cat48.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_its_cat$X0,ypred_its_cat$X0)
dev.off()

#Endpunkt Beatmung (RandomForest)
png(file=paste(path,'/proc/ROC_beat_cat48.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_beat_cat$X0,ypred_beat_cat$X0)
dev.off()


ypred_tod = read.csv(paste(path,'/y_pred_tod_alternativ24.csv',sep=''),header = TRUE,sep = ',')
ytest_tod = read.csv(paste(path,'/ytest_tod_alternativ24.csv',sep=''),header = TRUE,sep = ',')
ypred_its = read.csv(paste(path,'/y_pred_its24.csv',sep=''),header = TRUE,sep = ',')
ytest_its = read.csv(paste(path,'/ytest_its24.csv',sep=''),header = TRUE,sep = ',')
ypred_beat = read.csv(paste(path,'/y_pred_beat24.csv',sep=''),header = TRUE,sep = ',')
ytest_beat = read.csv(paste(path,'/ytest_beat24.csv',sep=''),header = TRUE,sep = ',')



ypred_tod_cat = read.csv(paste(path,'/y_pred_tod_alternativ_cat24.csv',sep=''),header = TRUE,sep = ',')
#ypred_tod_cat = read.csv(paste(path,'/y_pred_tod_alternativ_cat24.csv',sep=''),header = TRUE,sep = ',')
ytest_tod_cat = read.csv(paste(path,'/ytest_tod_alternativ_cat24.csv',sep=''),header = TRUE,sep = ',')
ypred_tod_rf_cat = read.csv(paste(path,'/y_pred_tod_rf_cat24.csv',sep=''),header = TRUE,sep = ',')
ytest_tod_rf_cat = read.csv(paste(path,'/ytest_tod_rf_cat24.csv',sep=''),header = TRUE,sep = ',')
ypred_its_cat = read.csv(paste(path,'/y_pred_its_cat24.csv',sep=''),header = TRUE,sep = ',')
#ypred_its_cat = read.csv(paste(path,'/y_pred_its_cat24.csv',sep=''),header = TRUE,sep = ',')
ytest_its_cat = read.csv(paste(path,'/ytest_its_cat24.csv',sep=''),header = TRUE,sep = ',')
#ytest_its_cat = read.csv(paste(path,'/ytest_its_cat24.csv',sep=''),header = TRUE,sep = ',')
ypred_beat_cat = read.csv(paste(path,'/y_pred_beat_cat24.csv',sep=''),header = TRUE,sep = ',')
ytest_beat_cat = read.csv(paste(path,'/ytest_beat_cat24.csv',sep=''),header = TRUE,sep = ',')


#Enpunkt Tod (LogReg)
png(file=paste(path,'/proc/ROC_tod_logreg24.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_tod$X0,ypred_tod$X0)
dev.off()

#Endpunkt ITS (XGBoost)
png(file=paste(path,'/proc/ROC_its24.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_its$X0,ypred_its$X0)
dev.off()

#Endpunkt Beatmung (RandomForest)
png(file=paste(path,'/proc/ROC_beat24.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_beat$X0,ypred_beat$X0)
dev.off()

#Enpunkt Tod (LogReg)
png(file=paste(path,'/proc/ROC_tod_logreg_cat24.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_tod_cat$X0,ypred_tod_cat$X0)
dev.off()

#Enpunkt Tod (RF)
png(file=paste(path,'/proc/ROC_tod_rf_cat24.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_tod_rf_cat$X0,ypred_tod_rf_cat$X0)
dev.off()

#Endpunkt ITS (XGBoost)
png(file=paste(path,'/proc/ROC_its_cat24.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_its_cat$X0,ypred_its_cat$X0)
dev.off()


#Endpunkt Beatmung (RandomForest)
png(file=paste(path,'/proc/ROC_beat_cat24.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_beat_cat$X0,ypred_beat_cat$X0)
dev.off()
