import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, \
    roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')
print(df.head())

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')
print(df.head())
actual = df.actual_label.values
model_RF = df.model_RF.values
model_LR = df.model_LR.values
predicted_RF = df.predicted_RF.values
predicted_LR = df.predicted_LR.values

conf_matr = confusion_matrix(df.actual_label.values, df.predicted_RF.values)
print("confusion_matrix:\n", conf_matr)


def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))


def find_conf_matrix_values(y_true, y_pred):
    """
    :param y_true: List with true data of classification
    :param y_pred: List with predicted data of classification
    :return: TP, FN, FP, TN
    """
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def Oleksiichuk_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


print("Oleksiichuk_confusion_matrix:\n", Oleksiichuk_confusion_matrix(actual, predicted_RF))

assert np.array_equal(Oleksiichuk_confusion_matrix(actual, predicted_RF),
                      confusion_matrix(actual, predicted_RF)), \
    'my confusion_matrix() is not correct for RF'

assert np.array_equal(Oleksiichuk_confusion_matrix(actual, predicted_LR),
                      confusion_matrix(actual, predicted_LR)), \
    'my confusion_matrix() is not correct for lR'

# Accuracy
score = accuracy_score(actual, predicted_RF)
print("Accuracy score on RF:", score)


def Oleksiichuk_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)


assert Oleksiichuk_accuracy_score(actual, predicted_RF) == accuracy_score(actual, predicted_RF), \
    'my accuracy_score failed RF'

assert Oleksiichuk_accuracy_score(actual, predicted_LR) == accuracy_score(actual, predicted_LR), \
    'my accuracy_score failed LR'

print("My accuracy score on RF:", Oleksiichuk_accuracy_score(actual, predicted_RF))
print("My accuracy score on LR:", Oleksiichuk_accuracy_score(actual, predicted_LR))

# Recall
print('Recall score on RF:', recall_score(actual, predicted_RF))


def Oleksiichuk_recal_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


assert Oleksiichuk_recal_score(actual, predicted_RF) == recall_score(actual, predicted_RF),\
    'my recal_score fails on RF'

assert Oleksiichuk_recal_score(actual, predicted_LR) == recall_score(actual, predicted_LR),\
    'my recal_score fails on LR'

print("My recall score on RF:", Oleksiichuk_recal_score(actual, predicted_RF))
print("My recall score on LR:", Oleksiichuk_recal_score(actual, predicted_LR))

# Precision
print("Precision score on RF:", precision_score(actual, predicted_RF))

def Oleksiichuk_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


assert Oleksiichuk_precision_score(actual, predicted_RF) == precision_score(actual, predicted_RF),\
    'my precision_score fails on RF'

assert Oleksiichuk_precision_score(actual, predicted_LR) == precision_score(actual, predicted_LR),\
    'my precision_score fails on LR'

print("My precision score on RF:", Oleksiichuk_precision_score(actual, predicted_RF))
print("My precision score on LR:", Oleksiichuk_precision_score(actual, predicted_LR))

# F1 score
print("F1 score on RF", f1_score(actual, predicted_RF))


def Oleksiichuk_f1_score(y_true, y_pred):
    precision = Oleksiichuk_precision_score(y_true, y_pred)
    recall = Oleksiichuk_recal_score(y_true, y_pred)
    return (2 * (precision * recall)) / (precision + recall)


assert Oleksiichuk_f1_score(actual, predicted_RF) == f1_score(actual, predicted_RF),\
    'my f1_score fails on RF'

assert Oleksiichuk_f1_score(actual, predicted_LR) == f1_score(actual, predicted_LR),\
    'my f1_score fails on LR'

print("My F1 score score on RF:", Oleksiichuk_f1_score(actual, predicted_RF))
print("My F1 score score on LR:", Oleksiichuk_f1_score(actual, predicted_LR))
print()


def test_thresholds(threshold: float = .5):
    print(f"Scores with threshold = {threshold}")
    predicted = (df.model_RF >= threshold).astype('int')

    print("Accuracy RF:", Oleksiichuk_accuracy_score(actual, predicted))
    print("Precision RF:", Oleksiichuk_precision_score(actual, predicted))
    print("Recall RF:", Oleksiichuk_recal_score(actual, predicted))
    print("F1 RF:", Oleksiichuk_f1_score(actual, predicted))
    print()


test_thresholds()
test_thresholds(.25)
test_thresholds(.6)
test_thresholds(.20)

# ROC
# Curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(actual, model_RF)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(actual, model_LR)

# AUC
auc_RF = roc_auc_score(actual, model_RF)
auc_LR = roc_auc_score(actual, model_LR)

print("AUC RF:", auc_RF)
print("AUC LR:", auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label=f'AUC RF: {auc_RF}')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'AUC LR: {auc_LR}')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')

plt.legend()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
