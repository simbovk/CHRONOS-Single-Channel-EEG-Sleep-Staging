"""Single source of truth for paper metrics."""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def per_class_specificity(confusion):
    """Compute one-vs-rest specificity for each class."""
    cm=np.asarray(confusion); tp=np.diag(cm); fp=cm.sum(0)-tp; fn=cm.sum(1)-tp; tn=cm.sum()-(tp+fp+fn)
    return np.divide(tn, tn+fp, out=np.zeros_like(tn,dtype=float), where=(tn+fp)!=0)


def compute_metrics(y_true, y_pred, probabilities, num_classes=5, labels=None):
    """Calculate notebook accuracy, kappa, macro and per-class metrics."""
    yt=np.asarray(y_true); yp=np.asarray(y_pred); order=list(range(num_classes))
    precision, recall, f1, support=precision_recall_fscore_support(yt,yp,labels=order,zero_division=0)
    cm=confusion_matrix(yt,yp,labels=order); specificity=per_class_specificity(cm)
    try: auroc=roc_auc_score(yt, probabilities, multi_class="ovr", average=None, labels=order)
    except Exception: auroc=np.zeros(num_classes)
    return {"overall":{"accuracy":float(accuracy_score(yt,yp)),"kappa":float(cohen_kappa_score(yt,yp)),
            "macro_f1":float(np.mean(f1)),"macro_sensitivity":float(np.mean(recall)),"macro_specificity":float(np.mean(specificity))},
            "per_class":{"labels":labels or order,"f1":f1.tolist(),"precision":precision.tolist(),"sensitivity":recall.tolist(),
                         "specificity":specificity.tolist(),"auroc":auroc.tolist(),"support":support.tolist()},
            "confusion_matrix":cm.tolist()}
