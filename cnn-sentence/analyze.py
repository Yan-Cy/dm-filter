import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from  cnn_sentence import CNN_Sentence
import data_helpers

n_classes = 2

posfile = '../danmu/quanzhi/quanzhi6_1w.pos'
negfile = '../danmu/quanzhi/quanzhi6.neg'
x_raw, y_test = data_helpers.load_data_and_labels(posfile, negfile) 
y_test = np.argmax(y_test, axis=1)
cnn_sentence = CNN_Sentence()
scores = cnn_sentence.predict(x_raw, print_info = False)

minscore = scores.min()
scorerange = scores.max() - minscore

y_score = np.zeros([len(scores)])
for i, score in enumerate(scores):
    neg = (score[0] - minscore) / scorerange
    pos = (score[1] - minscore) / scorerange
    #print [neg, pos]
    y_score[i] = pos / (pos + neg)
    print y_score[i] 

fpr = dict()
tpr = dict()
roc_auc = dict()

'''
y_score = scores
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
'''

fpr[0], tpr[0], _ = roc_curve(y_test, y_score)
roc_auc[0] = auc(fpr[0], tpr[0])



plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('result.png')
#plt.show()
