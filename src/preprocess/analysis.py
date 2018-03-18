import pandas as pd

train = pd.read_csv('./input/train.csv')
positive_train = train[train['identity_hate']>0]
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for label in labels:
    positive_train = train[train[label] > 0]
    print( label)
    print(positive_train.shape)

max = 0
max_line = None
for i,line in train.iterrows():
    length = len(line['comment_text'])
    if length>max:
        max =length
        max_line = line

print( 'longest: ', max)
print( 'line: ', max_line)