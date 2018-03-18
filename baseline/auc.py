import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc



labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

'''
Calculate auc score for each label, and return an average auc score
'''
def auc(test, subm):
	aucs = []
	fpr = dict()
	tpr = dict()
	for label in labels:
		true = np.array(test[label].values)
		pred = np.array(subm[label].values)
		auc = metrics.roc_auc_score(true,pred)
		print(label, auc)
		fpr[label], tpr[label], _ = roc_curve(true, pred)
		aucs.append(auc)
	avg_auc = np.average(np.array(aucs))
	plt.figure()
	lw = 2

	colors=['aqua', 'darkorange', 'cornflowerblue', 'green', 'yellow','navy']
	i=0
	for label, color in zip(labels, colors):
		 plt.plot(fpr[label], tpr[label], color=color, lw=lw,
             label='ROC curve of '+label+' (area = {1:0.2f})'
             ''.format(i, aucs[i]))
		 i+=1

	# plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
	plt.legend(loc="lower right")
	plt.savefig('auc.jpg')
	return avg_auc

if __name__ == '__main__':
	test = pd.read_csv('../input/tr_test.csv')
	subm = pd.read_csv('../output/pred_tr_test.csv')
	avg_auc = auc(test,subm)
	print("average auc ", avg_auc)
