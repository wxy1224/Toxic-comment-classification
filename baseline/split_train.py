import pandas as pd, numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

label_cols = ['id','comment_text','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
n = int(train.shape[0]*0.7)
tr_train = train.values[:n,:]
tr_test = train.values[n:,:]
print(train.shape, tr_train.shape, tr_test.shape)

'''
Generate train csv
'''
def split_train():
	train = pd.DataFrame(tr_train, columns = label_cols)
	train.to_csv('../input/tr_train.csv', index=False)

'''
Genearte test csv
'''
def split_test():
	test = pd.DataFrame(tr_test, columns = label_cols)
	test.to_csv('../input/tr_test.csv', index=False)


# Uncomment the two function calls if running for the first time
if __name__ == '__main__':
	print("generating new test/train sets")
	#split_train()
	#split_test()

