
install.packages(setdiff(c("tidyverse", "precrec","pROC"), rownames(installed.packages())))

library(precrec)
library(tidyverse)
library(pROC)


##### ROC Kurven #####

plotROC <- function(ytest,ypred){
  ROC = roc(ytest,ypred,ci=TRUE)
  #ROC.CI = ci.thresholds(ROC)
  
  plot(ROC,print.auc = TRUE, grid = TRUE, main = '', print.auc.x=0.625, print.auc.y=0.165,print.auc.cex=1.7,cex.axis=1.5,cex.lab=1.5)
}

plotROC_ci <- function(ytest,ypred){
  ROC = roc(ytest,ypred,ci=TRUE)
  #ROC.CI = ci.sp(ROC,conf.level=sqrt(1-0.05), sensivities=seq(0, 1, l=500))
  #ROC.CI2 = ci.se(ROC,conf.level=sqrt(1-0.05), specificities=seq(0, 1, l=500))
  ROC.CI = ci.thresholds(ROC)
  plot(ROC,print.auc = TRUE, grid = TRUE, main = '', print.auc.x=0.625, print.auc.y=0.165,print.auc.cex=1.7,cex.axis=1.5,cex.lab=1.5)
  
  plot(ROC.CI, type='s',col='blue')
}



#Logistische Regression
path = 'C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Modelle_KH/Ergebnisse'

ypred_lab10 = read.csv(paste(path,'/y_pred_aki.csv',sep=''),header = TRUE,sep = ',')
ytest_lab10 = read.csv(paste(path,'/ytest_aki.csv',sep=''),header = TRUE,sep = ',')

png(file=paste(path,'/plotroc/ROC_LR.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_lab10$X0,ypred_lab10$X0)
dev.off()


#ypred_lab10 = read.csv(paste(path,'/y_pred_aki_k.csv',sep=''),header = TRUE,sep = ',')
#ytest_lab10 = read.csv(paste(path,'/ytest_aki_k.csv',sep=''),header = TRUE,sep = ',')

#png(file=paste(path,'/plotroc/ROC_LR_k.png',sep=''),height=2000,width=2000,res=300)
#plotROC(ytest_lab10$X0,ypred_lab10$X0)
#dev.off()


#Random Forest
path = 'C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Modelle_KH/Ergebnisse_RF'

ypred_lab10_RF = read.csv(paste(path,'/y_pred_aki_RF.csv',sep=''),header = TRUE,sep = ',')
ytest_lab10_RF = read.csv(paste(path,'/ytest_aki_RF.csv',sep=''),header = TRUE,sep = ',')


png(file=paste(path,'/plotroc/ROC_RF.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_lab10_RF$X0,ypred_lab10_RF$X0)
dev.off()


#ypred_lab10_RF = read.csv(paste(path,'/y_pred_aki_RF_k.csv',sep=''),header = TRUE,sep = ',')
#ytest_lab10_RF = read.csv(paste(path,'/ytest_aki_RF_k.csv',sep=''),header = TRUE,sep = ',')


#png(file=paste(path,'/plotroc/ROC_RF_k.png',sep=''),height=2000,width=2000,res=300)
#plotROC(ytest_lab10_RF$X0,ypred_lab10_RF$X0)
#dev.off()


#XGBoost
path = 'C:/Users/lisan/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz/Modelle_KH/Ergebnisse_XGB'

ypred_lab10_XGB = read.csv(paste(path,'/y_pred_aki_XGB.csv',sep=''),header = TRUE,sep = ',')
ytest_lab10_XGB = read.csv(paste(path,'/ytest_aki_XGB.csv',sep=''),header = TRUE,sep = ',')

png(file=paste(path,'/plotroc/ROC_XGB.png',sep=''),height=2000,width=2000,res=300)
plotROC(ytest_lab10_XGB$X0,ypred_lab10_XGB$X0)
dev.off()

#ypred_lab10_XGB = read.csv(paste(path,'/y_pred_aki_XGB_k.csv',sep=''),header = TRUE,sep = ',')
#ytest_lab10_XGB = read.csv(paste(path,'/ytest_aki_XGB_k.csv',sep=''),header = TRUE,sep = ',')

#png(file=paste(path,'/plotroc/ROC_XGB_k.png',sep=''),height=2000,width=2000,res=300)
#plotROC(ytest_lab10_XGB$X0,ypred_lab10_XGB$X0)
#dev.off()
