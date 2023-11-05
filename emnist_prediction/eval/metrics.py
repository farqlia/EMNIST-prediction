import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from emnist_prediction.utils.constants import CLASS_LABELS
from emnist_prediction.utils.utils import id_2_label


def min_f1_score(y_true, y_pred):
    f1_score_by_class = f1_score(y_true, y_pred, average=None)
    return f1_score_by_class.min()


def get_classification_report(y_true, y_pred):
    precision, recall, f1score, support = precision_recall_fscore_support(y_true, y_pred)
    clf_report_df = pd.DataFrame({'precision': precision, 'recall': recall, 'f1_score': f1score, 'support': support},
                                 index=id_2_label(np.arange(len(CLASS_LABELS))))

    return clf_report_df
