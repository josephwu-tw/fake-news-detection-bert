# setup

import __path__ as path
import numpy as np
import pandas as pd
import sklearn.metrics as skmetric
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.models import load_model

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

keras.backend.clear_session()

# functions

def plot_confusion_matrix(cm, title):
    fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
    sns.heatmap(cm/1000, 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                annot = True, fmt = 'g',
                linewidths=.2,linecolor="Darkblue", cmap="Blues")
    plt.title(title, fontsize=14)
    plt.show()

# load model

model_bert = load_model(path.model_dir + 'bert.keras')
model_concat = load_model(path.model_dir + 'concat.keras')

# test data import

x_test_bert = pd.read_csv(path.data_dir + 'x_test_bert.csv')
x_test_mlp = pd.read_csv(path.data_dir + 'x_test_mlp.csv')
y_test = pd.read_csv(path.data_dir + 'y_test.csv')

# data transform

x_test_bert = np.array(x_test_bert, dtype = float)
x_test_mlp = np.array(x_test_mlp, dtype = float)
y_test = np.array(y_test, dtype = float)

# evaluate

evl_bert = model_bert.evaluate(x_test_bert, y_test, verbose = 1)
print('Test - Bret  : %.3f' % evl_bert[1])
evl_concat = model_concat.evaluate([x_test_bert, x_test_mlp], y_test, verbose = 1)
print('Test - Concat: %.3f' % evl_concat[1])

# prediction and confusion matrix

pred_bert = model_bert.predict(x_test_bert, verbose=1)
pred_bert = pred_bert[:, 0].round()

cm_bert = skmetric.confusion_matrix(y_test, pred_bert)
plot_confusion_matrix(cm = cm_bert, title = 'Confusion Matrix - BERT')

pred_concat = model_concat.predict([x_test_bert, x_test_mlp], verbose=1)
pred_concat = pred_concat[:, 0].round()

cm_concat = skmetric.confusion_matrix(y_test, pred_concat)
plot_confusion_matrix(cm = cm_concat, title = 'Confusion Matrix - Concat')

# compare report and roc curve

accuracy_bert = accuracy_score(y_test, pred_bert)
precision_bert = precision_score(y_test, pred_bert)
f1_bert = f1_score(y_test, pred_bert)
rp_bert = skmetric.classification_report(y_test, pred_bert)

accuracy_concat = accuracy_score(y_test, pred_concat)
precision_concat = precision_score(y_test, pred_concat)
f1_concat = f1_score(y_test, pred_concat)
rp_concat = skmetric.classification_report(y_test, pred_concat)

print('')
print('BERT')
print('')
print(rp_bert)
print('')
print('Accuracy : %f' % accuracy_bert)
print('Precision: %f' % precision_bert)
print('F1-Score : %f' % f1_bert)
print('------------------------------------------')
print('')
print('Concat')
print('')
print(rp_concat)
print('')
print('Accuracy : %f' % accuracy_concat)
print('Precision: %f' % precision_concat)
print('F1-Score : %f' % f1_concat)


bfpr, btpr, bthresholds = roc_curve(y_test, pred_bert)
bauc = auc(bfpr, btpr)

cfpr, ctpr, cthresholds = roc_curve(y_test, pred_concat)
cauc = auc(cfpr, ctpr)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(cfpr, btpr, label='BERT(AUC = {:.3f})'.format(bauc))
plt.plot(cfpr, ctpr, label='Concat(AUC = {:.3f})'.format(cauc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()