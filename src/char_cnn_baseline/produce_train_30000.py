import pandas as pd


train = pd.read_csv("./input/train.csv")
train = train[:30000]
headers = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
new_train = pd.DataFrame(train, columns=headers)

new_train.to_csv("./input/train_30000.csv",index=False)