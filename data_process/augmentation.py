# !/usr/bin/env python -W ignore::DeprecationWarning
import pandas as pd, numpy as np
import random
import codecs

def augmentation(orig_tain_path, names_list_path, output_path,
    label_name="identity_hate"):
  label_cols = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene',
                'threat', 'insult', 'identity_hate']

  train = pd.read_csv(orig_train_path)

  names = []
  with codecs.open(names_list_path, encoding='utf-8') as file:

    names = file.readlines()
    names = map(lambda x: x.strip(), names)
  # names = names.split("\n")
  n = len(names)
  print(names[0])
  selected = train.loc[train[label_name] == 1].as_matrix()
  # select_comments = selected["comment_text"]
  # comments = select_comments.as_matrix()
  construct = []
  print ("size of selected",selected.shape)
  for i in range(0, selected.shape[0]):
    selected_row = selected[i]
    comment = selected_row[label_cols.index('comment_text')]
    # if i == 0:
    #   print("comment", comment)
    for k in range(n):
      s =  names[k]#.decode("UTF-8")
      if i == 0 and k ==0:
        print ("name",s,"comment", comment)
      if (comment.find(s) > 0):
        randints = random.sample(range(n), 30)
        for j in randints:
          if j == k:
            continue
          # j = random.randint(0,len(names)-1)
          new_word = names[j]
          modified = comment.replace(s, new_word)
          # print(s, new_word, modified)
          new_selected_row = selected_row.copy()
          new_selected_row[label_cols.index('comment_text')] = modified
          construct.append(new_selected_row)
        # break
        # if i%100==0 and len(construct)>0:
        # 	construct_arr = np.array(construct)
        # 	new_train = pd.DataFrame(construct_arr, columns = label_cols)
        # 	new_train.to_csv('new_train_data/train_'+str(i)+'.csv', index=False)
        # 	construct=[]
        # 	print(i)
  print("!!!!construct",len(construct))
  construct_arr = np.array(construct)
  print("construct_arr", construct_arr.shape)
  new_train = pd.DataFrame(construct_arr, columns=label_cols)
  new_train.to_csv(output_path, index=False)


if __name__ == '__main__':
  orig_train_path = '../input/train.csv'
  names_list_path = 'names.txt'
  output_path = 'new_train_data/new_train.csv'
  augmentation(orig_train_path, names_list_path, output_path)
