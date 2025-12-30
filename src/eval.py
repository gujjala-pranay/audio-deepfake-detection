import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def compute_eer(bonafide_scores, spoof_scores):
    """
    Compute the Equal Error Rate (EER).
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = (frr[min_index] + far[min_index]) / 2
    return eer, thresholds[min_index]

def compute_det_curve(target_scores, nontarget_scores):
    """
    Compute False Rejection Rate (FRR) and False Acceptance Rate (FAR).
    """
    n_targets = len(target_scores)
    n_nontargets = len(nontarget_scores)

    target_scores = np.sort(target_scores)
    nontarget_scores = np.sort(nontarget_scores)

    thresholds = np.unique(np.concatenate((target_scores, nontarget_scores)))
    frr = np.zeros(len(thresholds))
    far = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        frr[i] = np.sum(target_scores < t) / n_targets
        far[i] = np.sum(nontarget_scores >= t) / n_nontargets

    return frr, far, thresholds

def plot_confusion_matrix(y_true, y_pred, classes=['Fake', 'Real']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt.gcf()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    return plt.gcf()
