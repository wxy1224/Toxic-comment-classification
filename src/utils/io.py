import pandas as pd
import

class preprocessing_wrapper(object):
    def prepare_data_folder(input_path, output_folder):
        train = pd.read_csv(input_path)
        n = int(train.shape[0] * 0.0001)
        tr_train = train.values[:n, :]

        label_cols = ['id','comment_text','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        tr_train.to_csv('../output_folder/tr_train.csv', index=False)
        print(train.shape, tr_train.shape, tr_train.shape)
