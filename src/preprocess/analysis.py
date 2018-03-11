import pandas as pd

train = pd.read_csv('./input/train.csv')
positive_train = train[train['identity_hate']>0]
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for label in labels:
    positive_train = train[train[label] > 0]
    print label
    print positive_train.shape