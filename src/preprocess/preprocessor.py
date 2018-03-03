
from keras.preprocessing import text, sequence
import pandas as pd
from src.config.static_config import StaticConfig
from src.utils.io import create_folder

class SeqProcessor(object):

    def __init__(self):
        self.global_config = StaticConfig()
        self.tokenizer = text.Tokenizer(num_words=self.global_config.max_features)

    def extract_y(self, train):
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        return train[list_classes].values

    def extract_x_train(self, train):
        list_sentences_train = train["comment_text"].fillna("CVxTz").values
        self.tokenizer.fit_on_texts(list(list_sentences_train))
        list_tokenized_train = self.tokenizer.texts_to_sequences(list_sentences_train)
        return sequence.pad_sequences(list_tokenized_train, maxlen=self.global_config.maxlen)


    def preprocess_train(self, train):
        train = train.sample(frac=1)
        return (self.extract_x_train(train), self.extract_y(train))

    def preprocess_test(self,test):
        list_sentences_test = test["comment_text"].fillna("CVxTz").values
        list_tokenized_test = self.tokenizer.texts_to_sequences(list_sentences_test)
        return sequence.pad_sequences(list_tokenized_test, maxlen=self.global_config.maxlen)

    def prepare_data_folder(self, train_input_path, output_folder_path, train_test_factor=1.0, debug_factor=1.0):
        '''
        This method will take the train data and then divide it into train and test sets by a factor of train_test_factor.
        Then the train dataset is splitted according the column name first.
        The folders under splitted_output_folder splitted by the label names:
        'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
        Then the folder has static_config.preprocess_splits splitted slices of data. Each data has all of the label=1 from
        the above label of the training data set along with equal number of label=0 data rows from the training dataset as well.

        The test data is output to the test data folder for validation.

        :param train_input_path: file path for original training
        :param splitted_output_folder: output folder
        :param train_test_factor: ratio to split train and test
        :param debug_factor: if run with this config, only sample a small proportion of the raw data. 1.0 means no debug
        :return:
        '''
        global_config = StaticConfig()
        create_folder(output_folder_path)
        label_cols = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        raw_data = pd.read_csv(train_input_path)

        train_data_size = int(raw_data.shape[0] * train_test_factor * debug_factor)
        train = pd.DataFrame(raw_data[:train_data_size],columns = label_cols)

        # test = pd.DataFrame(raw_data[train_data_size:] if debug_factor == 1.0 else raw_data[-100:],columns = label_cols)
        # test_name = "{}/{}".format(output_folder_path,"test.csv")
        # test.to_csv(test_name)


        for label_name in global_config.labels:
            label_output = output_folder_path +"/"+ label_name
            create_folder(label_output)
            positive_train = train[train[label_name]>0]
            if positive_train.shape[0] == 0:
                return
            negative_train = train[train[label_name] == 0]
            ratio = negative_train.shape[0]/positive_train.shape[0]
            slice_size = positive_train.shape[0]
            for i in range(int(ratio)):

                sub_train_df = positive_train

                sub_test_df = negative_train[(slice_size*i):(slice_size*(i+1))]

                sub_train_df = pd.concat([sub_train_df, sub_test_df], ignore_index=True)
                sub_train_df = sub_train_df.sample(frac=1).reset_index(drop=True)
                sub_train_output_file_path = '{}/tr_train_{}.csv'.format(label_output, i)
                sub_train_df.to_csv(sub_train_output_file_path)#, index=False)
                print ('output subset {} to file '.format(i, sub_train_output_file_path))


if __name__ == "__main__":
    wrapper = SeqProcessor()
    wrapper.prepare_data_folder('./input/train.csv', './preprocessing_wrapper_demo_output',debug_factor=0.0001)

