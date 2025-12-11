
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os


def accuracy(pred, true):
    """Compute accuracy of predictions."""
    correct_prediction = np.equal(np.argmax(pred, 1), true).astype(np.float32)

    return np.mean(correct_prediction)


def auc(pred, true, is_logit=True):
    """Computes the Area Under the Curve (AUC)."""
    if is_logit:
        pos_probs = softmax(pred, axis=1)[:, 1]     # Probability of class 1 (TDC)
    else:
        pos_probs = pred[:, 0]                      # Or use ASD prob directly
    try:
        auc_out = metrics.roc_auc_score(true, pos_probs)
    except:
        auc_out = 0
    return auc_out


def prf(pred, true):
    """ Computes precision, recall, and f1-score relatively to the positive class: 0-ASD.
    input: preds, true_labels"""

    p, r, f, s = metrics.precision_recall_fscore_support(
        y_pred = pred, 
        y_true = true, 
        pos_label=0, 
        average='binary', 
        zero_division=0)
    return [p, r, f]


def specificity(pred, true):
    """ Computes  relatively to the positive class: 0-ASD
        input: preds, true_labels"""
    
    tn, fp, fn, tp = metrics.confusion_matrix(
        y_pred = pred, 
        y_true = true, 
        labels=[1, 0]   # Label 1 = negative (TDC), Label 0 = positive (ASD)
        ).ravel()

    specificity = tn / (tn + fp)

    return specificity


def npv(pred, true):
    """ Computes  relatively to the positive class: 0-ASD.
            input: preds, true_labels"""

    tn, fp, fn, tp = metrics.confusion_matrix(
        y_pred = pred, 
        y_true = true, 
        labels=[1, 0]
        ).ravel()
    
    denom = tn + fn

    if denom == 0:
        return 0.0
    
    npv = tn / (tn + fn)

    return npv


def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def plot_accuracies(train_accs, val_accs, fig_dir, fold=None):
    plt.figure()

    if fold == None:
        my_file = 'acc_final.png'
        plt.title(f"Training and validation accuracy (final model)")
    else:
        my_file = f'acc_{fold+1}.png' 
        plt.title(f"Training and validation accuracy (fold {fold + 1})")
    
    plt.plot(train_accs, label="Training accuracy")
    plt.plot(val_accs, label="Validation accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(fig_dir, my_file), dpi=300, bbox_inches='tight')
    plt.close()

def plot_losses(train_losses, val_losses, fig_dir, fold=None):
    plt.figure()
    
    if fold == None:
        my_file = 'loss_final.png'
        plt.title(f"Training and validation loss (final model)")
    else:
        my_file = f'loss_{fold+1}.png' 
        plt.title(f"Training and validation loss (fold {fold + 1})")

    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim([0, 1.0])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(fig_dir, my_file), dpi=300, bbox_inches='tight')
    plt.close()


def plot_cv_results(fold_results, save_path):
    """
    Generate and save a grouped bar chart of cross-validation results across folds.
    Metrics included: train_accuracy, val_accuracy, test_accuracy, F1, AUC, etc.
    """
    cv_metrics = ['best_train_acc', 'best_val_acc', 'best_precision', 'best_recall', 'best_f1', 'best_auc', 'specificity', 'npv']
    
    folds = [r.get('fold', i + 1) for i, r in enumerate(fold_results)]
    n_folds = len(folds)

    if n_folds == 0:
        print("No fold results to plot.")
        return

    # Prepare data for each metric
    data = {m: [r.get(m, 0) for r in fold_results] for m in cv_metrics}
    x = np.arange(n_folds)
    width = 0.1
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    plt.figure(figsize=(14, 6))
    for i, metric in enumerate(cv_metrics):
        plt.bar(x + i * width, data[metric], width, label=metric.replace('_', ' ').capitalize(), color=colors[i])

    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Cross-Validation Results per Fold', fontsize=14, weight='bold')
    plt.xticks(x + width * (len(cv_metrics) / 2), folds)
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved CV results plot to: {save_path}")
    plt.close()