## Further Analysis (Evaluation metrics and calibration)
import sklearn
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from torchmetrics.functional import calibration_error
import torch
from sklearn.calibration import calibration_curve
import pandas as pd
from sklearn.metrics import roc_curve, auc, fbeta_score, balanced_accuracy_score, accuracy_score, f1_score
import numpy as np


path = 'C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Analyse_Gesamtdaten_Krankenhaus'

y_pred_aki_rf=pd.read_csv(path + '/Ergebnisse_RF/y_pred_ans_lab10_RF.csv')
y_pred_aki_rf=y_pred_aki_rf.iloc[:,1]
y_test_aki_rf=pd.read_csv(path + '/Ergebnisse_RF/ytest_ans_lab10_RF.csv')
y_test_aki_rf=y_test_aki_rf.iloc[:,1]
brier_aki_rf = sklearn.metrics.brier_score_loss(y_test_aki_rf,y_pred_aki_rf)
print(brier_aki_rf)
ece_aki_rf_5 = calibration_error(torch.tensor(y_pred_aki_rf),torch.tensor(y_test_aki_rf),n_bins=5,norm='l1',task='binary')
print(ece_aki_rf_5)
ece_aki_rf_10 = calibration_error(torch.tensor(y_pred_aki_rf),torch.tensor(y_test_aki_rf),n_bins=10,norm='l1',task='binary')
print(ece_aki_rf_10)
ece_aki_rf_15 = calibration_error(torch.tensor(y_pred_aki_rf),torch.tensor(y_test_aki_rf),n_bins=15,norm='l1',task='binary')
print(ece_aki_rf_15)
#15 standard
ece_aki_rf = calibration_error(torch.tensor(y_pred_aki_rf),torch.tensor(y_test_aki_rf),task='binary')
print(ece_aki_rf)
confusion_matrix=sklearn.metrics.confusion_matrix(y_test_aki_rf, y_pred_aki_rf>0.5)
TP =confusion_matrix[1,1]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]
TN = confusion_matrix[0,0]
PPV = TP/(TP+FP)
print(PPV)
NPV = TN/(TN+FN)
print(NPV)
Fbeta = fbeta_score(y_test_aki_rf,y_pred_aki_rf>0.5, beta=1)
print(Fbeta)
bal_acc = balanced_accuracy_score(y_test_aki_rf, y_pred_aki_rf>0.5)
print(bal_acc)
acc = accuracy_score(y_test_aki_rf, y_pred_aki_rf>0.5)
print(acc)
prec = TP/(TP+FP)
print(prec)
rec = TP /(TP+FN)
print(rec)

thresholds = np.arange(0, 1.01, 0.01)
best_f1 = 0.0
best_threshold = 0.0
for threshold in thresholds:
    # Convert probabilities to binary predictions based on the current threshold
    y_pred = (y_pred_aki_rf> threshold).astype(int)
    
    # Calculate the F1 score for the current threshold
    f1 = f1_score(y_test_aki_rf, y_pred)
    
    # Update the best threshold if the current F1 score is higher
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(best_threshold)

#best_thresh=0.3783783783783784
best_thresh = 0.17
confusion_matrix=sklearn.metrics.confusion_matrix(y_test_aki_rf, y_pred_aki_rf>best_thresh)
TP =confusion_matrix[1,1]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]
TN = confusion_matrix[0,0]
PPV = TP/(TP+FP)
print(PPV)
NPV = TN/(TN+FN)
print(NPV)
Fbeta = fbeta_score(y_test_aki_rf,y_pred_aki_rf>best_thresh, beta=1)
print(Fbeta)
bal_acc = balanced_accuracy_score(y_test_aki_rf, y_pred_aki_rf>best_thresh)
print(bal_acc)
acc = accuracy_score(y_test_aki_rf, y_pred_aki_rf>best_thresh)
print(acc)
prec = TP/(TP+FP)
print(prec)
rec = TP /(TP+FN)
print(rec)

y_pred_aki_lrk=pd.read_csv(path + '/Ergebnisse/y_pred_ans_lab10_k.csv')
y_pred_aki_lrk=y_pred_aki_lrk.iloc[:,1]
y_test_aki_lrk=pd.read_csv(path + '/Ergebnisse/ytest_ans_lab10_k.csv')
y_test_aki_lrk=y_test_aki_lrk.iloc[:,1]
brier_aki_lrk = sklearn.metrics.brier_score_loss(y_test_aki_lrk,y_pred_aki_lrk)
print(brier_aki_lrk)
ece_aki_lrk_5 = calibration_error(torch.tensor(y_pred_aki_lrk),torch.tensor(y_test_aki_lrk),n_bins=5,norm='l1',task='binary')
print(ece_aki_lrk_5)
ece_aki_lrk_10 = calibration_error(torch.tensor(y_pred_aki_lrk),torch.tensor(y_test_aki_lrk),n_bins=10,norm='l1',task='binary')
print(ece_aki_lrk_10)
ece_aki_lrk_15 = calibration_error(torch.tensor(y_pred_aki_lrk),torch.tensor(y_test_aki_lrk),n_bins=15,norm='l1',task='binary')
print(ece_aki_lrk_15)
#15 standard
ece_aki_lrk = calibration_error(torch.tensor(y_pred_aki_lrk),torch.tensor(y_test_aki_lrk),task='binary')
print(ece_aki_lrk)
confusion_matrix=sklearn.metrics.confusion_matrix(y_test_aki_lrk, y_pred_aki_lrk>0.5)
TP =confusion_matrix[1,1]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]
TN = confusion_matrix[0,0]
PPV = TP/(TP+FP)
print(PPV)
NPV = TN/(TN+FN)
print(NPV)
Fbeta = fbeta_score(y_test_aki_lrk,y_pred_aki_lrk>0.5, beta=1)
print(Fbeta)
bal_acc = balanced_accuracy_score(y_test_aki_lrk, y_pred_aki_lrk>0.5)
print(bal_acc)
acc = accuracy_score(y_test_aki_lrk, y_pred_aki_lrk>0.5)
print(acc)
prec = TP/(TP+FP)
print(prec)
rec = TP /(TP+FN)
print(rec)

thresholds = np.arange(0, 1.01, 0.01)
best_f1 = 0.0
best_threshold = 0.0
for threshold in thresholds:
    # Convert probabilities to binary predictions based on the current threshold
    y_pred = (y_pred_aki_lrk> threshold).astype(int)
    
    # Calculate the F1 score for the current threshold
    f1 = f1_score(y_test_aki_lrk, y_pred)
    
    # Update the best threshold if the current F1 score is higher
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(best_threshold)

#best_thresh=0.3783783783783784
best_thresh = 0.11
#best_thresh=0.49
confusion_matrix=sklearn.metrics.confusion_matrix(y_test_aki_lrk, y_pred_aki_lrk>best_thresh)
TP =confusion_matrix[1,1]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]
TN = confusion_matrix[0,0]
PPV = TP/(TP+FP)
print(PPV)
NPV = TN/(TN+FN)
print(NPV)
Fbeta = fbeta_score(y_test_aki_lrk,y_pred_aki_lrk>best_thresh, beta=1)
print(Fbeta)
bal_acc = balanced_accuracy_score(y_test_aki_lrk, y_pred_aki_lrk>best_thresh)
print(bal_acc)
acc = accuracy_score(y_test_aki_lrk, y_pred_aki_lrk>best_thresh)
print(acc)
prec = TP/(TP+FP)
print(prec)
rec = TP /(TP+FN)
print(rec)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=5,strategy="uniform")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=5,strategy="uniform")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_uniform_aki.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=5,strategy="quantile")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=5,strategy="quantile")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_quantile_aki.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=10,strategy="uniform")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=10,strategy="uniform")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_uniform_aki_10.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=10,strategy="quantile")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=10,strategy="quantile")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_quantile_aki_10.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=15,strategy="uniform")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=15,strategy="uniform")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_uniform_aki_15.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=15,strategy="quantile")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=15,strategy="quantile")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_quantile_aki_15.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=20,strategy="uniform")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=20,strategy="uniform")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_uniform_aki_20.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=20,strategy="quantile")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=20,strategy="quantile")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_quantile_aki_20.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=25,strategy="uniform")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=25,strategy="uniform")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_uniform_aki_25.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=25,strategy="quantile")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=25,strategy="quantile")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_quantile_aki_25.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=50,strategy="uniform")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=50,strategy="uniform")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_uniform_aki_50.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=50,strategy="quantile")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Numerical RF (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=50,strategy="quantile")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('Dichotomous LR (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_quantile_aki_50.jpg')
plt.show()








path_n = r'C:\Users\wendland\Documents\GitHub\GTT-Trigger-tool\Lisanne\Neuer_Ansatz\Modelle_KH\Ergebnisse'

y_pred_aki_rf=pd.read_csv(path_n + '/y_pred_aki.csv')
y_pred_aki_rf=y_pred_aki_rf.iloc[:,1]
y_test_aki_rf=pd.read_csv(path_n + '/ytest_aki.csv')
y_test_aki_rf=y_test_aki_rf.iloc[:,1]
brier_aki_rf = sklearn.metrics.brier_score_loss(y_test_aki_rf,y_pred_aki_rf)
print(brier_aki_rf)
ece_aki_rf_5 = calibration_error(torch.tensor(y_pred_aki_rf),torch.tensor(y_test_aki_rf),n_bins=5,norm='l1',task='binary')
print(ece_aki_rf_5)
ece_aki_rf_10 = calibration_error(torch.tensor(y_pred_aki_rf),torch.tensor(y_test_aki_rf),n_bins=10,norm='l1',task='binary')
print(ece_aki_rf_10)
ece_aki_rf_15 = calibration_error(torch.tensor(y_pred_aki_rf),torch.tensor(y_test_aki_rf),n_bins=15,norm='l1',task='binary')
print(ece_aki_rf_15)
#15 standard
ece_aki_rf = calibration_error(torch.tensor(y_pred_aki_rf),torch.tensor(y_test_aki_rf),task='binary')
print(ece_aki_rf)
confusion_matrix=sklearn.metrics.confusion_matrix(y_test_aki_rf, y_pred_aki_rf>0.5)
TP =confusion_matrix[1,1]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]
TN = confusion_matrix[0,0]
PPV = TP/(TP+FP)
print(PPV)
NPV = TN/(TN+FN)
print(NPV)
Fbeta = fbeta_score(y_test_aki_rf,y_pred_aki_rf>0.5, beta=1)
print(Fbeta)
bal_acc = balanced_accuracy_score(y_test_aki_rf, y_pred_aki_rf>0.5)
print(bal_acc)
acc = accuracy_score(y_test_aki_rf, y_pred_aki_rf>0.5)
print(acc)
prec = TP/(TP+FP)
print(prec)
rec = TP /(TP+FN)
print(rec)

thresholds = np.arange(0, 1.01, 0.01)
best_f1 = 0.0
best_threshold = 0.0
for threshold in thresholds:
    # Convert probabilities to binary predictions based on the current threshold
    y_pred = (y_pred_aki_rf> threshold).astype(int)
    
    # Calculate the F1 score for the current threshold
    f1 = f1_score(y_test_aki_rf, y_pred)
    
    # Update the best threshold if the current F1 score is higher
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(best_threshold)

#best_thresh=0.3783783783783784
best_thresh = 0.12
confusion_matrix=sklearn.metrics.confusion_matrix(y_test_aki_rf, y_pred_aki_rf>best_thresh)
TP =confusion_matrix[1,1]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]
TN = confusion_matrix[0,0]
PPV = TP/(TP+FP)
print(PPV)
NPV = TN/(TN+FN)
print(NPV)
Fbeta = fbeta_score(y_test_aki_rf,y_pred_aki_rf>best_thresh, beta=1)
print(Fbeta)
bal_acc = balanced_accuracy_score(y_test_aki_rf, y_pred_aki_rf>best_thresh)
print(bal_acc)
acc = accuracy_score(y_test_aki_rf, y_pred_aki_rf>best_thresh)
print(acc)
prec = TP/(TP+FP)
print(prec)
rec = TP /(TP+FN)
print(rec)

path_t = r'C:\Users\wendland\Documents\GitHub\GTT-Trigger-tool\Lisanne\Neuer_Ansatz\KH_Daten_auf_Mimic_Modelle'

y_test_aki_lrk=pd.read_csv(path_t + '/y_LR.csv')
y_test_aki_lrk=y_test_aki_lrk.iloc[:,1]
y_pred_aki_lrk=pd.read_csv(path_t + '/y_pred_LR.csv')
y_pred_aki_lrk=y_pred_aki_lrk.iloc[:,1]

y_test_aki_lrk=np.array(y_test_aki_lrk[y_pred_aki_lrk==y_pred_aki_lrk])
y_pred_aki_lrk=np.array(y_pred_aki_lrk[y_pred_aki_lrk==y_pred_aki_lrk])
#y_pred_aki_lrk[y_pred_aki_lrk!=y_pred_aki_lrk]

brier_aki_lrk = sklearn.metrics.brier_score_loss(y_test_aki_lrk,y_pred_aki_lrk)
print(brier_aki_lrk)
ece_aki_lrk_5 = calibration_error(torch.tensor(y_pred_aki_lrk),torch.tensor(y_test_aki_lrk),n_bins=5,norm='l1',task='binary')
print(ece_aki_lrk_5)
ece_aki_lrk_10 = calibration_error(torch.tensor(y_pred_aki_lrk),torch.tensor(y_test_aki_lrk),n_bins=10,norm='l1',task='binary')
print(ece_aki_lrk_10)
ece_aki_lrk_15 = calibration_error(torch.tensor(y_pred_aki_lrk),torch.tensor(y_test_aki_lrk),n_bins=15,norm='l1',task='binary')
print(ece_aki_lrk_15)
#15 standard
ece_aki_lrk = calibration_error(torch.tensor(y_pred_aki_lrk),torch.tensor(y_test_aki_lrk),task='binary')
print(ece_aki_lrk)
confusion_matrix=sklearn.metrics.confusion_matrix(y_test_aki_lrk, y_pred_aki_lrk>0.5)
TP =confusion_matrix[1,1]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]
TN = confusion_matrix[0,0]
PPV = TP/(TP+FP)
print(PPV)
NPV = TN/(TN+FN)
print(NPV)
Fbeta = fbeta_score(y_test_aki_lrk,y_pred_aki_lrk>0.5, beta=1)
print(Fbeta)
bal_acc = balanced_accuracy_score(y_test_aki_lrk, y_pred_aki_lrk>0.5)
print(bal_acc)
acc = accuracy_score(y_test_aki_lrk, y_pred_aki_lrk>0.5)
print(acc)
prec = TP/(TP+FP)
print(prec)
rec = TP /(TP+FN)
print(rec)

thresholds = np.arange(0, 1.01, 0.01)
best_f1 = 0.0
best_threshold = 0.0
for threshold in thresholds:
    # Convert probabilities to binary predictions based on the current threshold
    y_pred = (y_pred_aki_lrk> threshold).astype(int)
    
    # Calculate the F1 score for the current threshold
    f1 = f1_score(y_test_aki_lrk, y_pred)
    
    # Update the best threshold if the current F1 score is higher
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(best_threshold)

#best_thresh=0.3783783783783784
best_thresh = 0.23
#best_thresh=0.49
confusion_matrix=sklearn.metrics.confusion_matrix(y_test_aki_lrk, y_pred_aki_lrk>best_thresh)
TP =confusion_matrix[1,1]
FP = confusion_matrix[0,1]
FN = confusion_matrix[1,0]
TN = confusion_matrix[0,0]
PPV = TP/(TP+FP)
print(PPV)
NPV = TN/(TN+FN)
print(NPV)
Fbeta = fbeta_score(y_test_aki_lrk,y_pred_aki_lrk>best_thresh, beta=1)
print(Fbeta)
bal_acc = balanced_accuracy_score(y_test_aki_lrk, y_pred_aki_lrk>best_thresh)
print(bal_acc)
acc = accuracy_score(y_test_aki_lrk, y_pred_aki_lrk>best_thresh)
print(acc)
prec = TP/(TP+FP)
print(prec)
rec = TP /(TP+FN)
print(rec)

import matplotlib.pyplot as plt

brier_aki_lrk=0.12

path = 'C:/Users/wendland/Documents/GitHub/GTT-Trigger-tool/Lisanne/Neuer_Ansatz'

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=10,strategy="uniform")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Ghmlc (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=10,strategy="uniform")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('MIMIC-IV (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_uniform_aki_10.jpg')
plt.show()

fig, ax = plt.subplots(dpi=300, figsize=(6,6))
plt.xlim(0,1)
plt.ylim(0,1)
prob_true_aki_rf, prob_pred_aki_rf = calibration_curve(y_test_aki_rf, y_pred_aki_rf,n_bins=10,strategy="quantile")
plt.plot(prob_pred_aki_rf,prob_true_aki_rf, marker='o', linewidth=1, label=str('Ghmlc (Brier: ' + str(np.around(brier_aki_rf,3)) + ')'),markersize=10)
prob_true_aki_lrk, prob_pred_aki_lrk = calibration_curve(y_test_aki_lrk, y_pred_aki_lrk,n_bins=10,strategy="quantile")
plt.plot(prob_pred_aki_lrk,prob_true_aki_lrk, marker='o', linewidth=1, label=str('MIMIC-IV (Brier: ' + str(np.around(brier_aki_lrk,3)) + ')'),markersize=10)

line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
ax.set_xlabel('Predicted risk',fontsize=16)
ax.set_ylabel('Observed proportion',fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=14)
plt.savefig(path+'/calibration_quantile_aki_10.jpg')
plt.show()
