import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics

test = pd.read_csv('../input/tr_test.csv')
subm = pd.read_csv('../output/pred_tr_test.csv')

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

'''
Calculate auc score for each label, and return an average auc score
'''
def auc(test, subm):
	aucs = []
	for label in labels:
		true = np.array(test[label].values)
		pred = np.array(subm[label].values)
		auc = metrics.roc_auc_score(true,pred)
		print(label, auc)
		aucs.append(auc)
	avg_auc = np.average(np.array(aucs))
	return avg_auc

if __name__ == '__main__':
	avg_auc = auc(test,subm)
	print("average auc ", avg_auc)
