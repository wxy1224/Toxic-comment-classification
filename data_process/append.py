# !/usr/bin/env python -W ignore::DeprecationWarning
import pandas as pd, numpy as np

def append(train_file_path, new_file_path, save_path):
    train = pd.read_csv(train_file_path)
    new_file = pd.read_csv(new_file_path)

    label_cols = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene',
                'threat', 'insult', 'identity_hate']
    all_data = np.vstack((train, new_file))
    print(all_data.shape)

    all_data_csv = pd.DataFrame(all_data, columns = label_cols)
    all_data_csv.to_csv(save_path, index=False)

def verify_new_train(train_file_path):
	train = pd.read_csv(train_file_path)
	print(train.shape)
	print(train.values[0])

if __name__ == '__main__':
    train_file_path = "../input/train.csv"
    new_file_path = "new_train_data/new_train.csv"
    save_path = "../preprocessing_wrapper_demo_output/0/all_train_hate.csv"
    append(train_file_path, new_file_path, save_path)
    verify_new_train(save_path)
    new_file_path2 = 'new_train_data/new_train_threat.csv'
    save_path2 = "../preprocessing_wrapper_demo_output/1/all_train_threat.csv"
    append(train_file_path, new_file_path2, save_path2)
    verify_new_train(save_path2)