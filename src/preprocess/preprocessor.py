
from keras.preprocessing import text, sequence
import pandas as pd
from src.config.static_config import StaticConfig
from src.utils.utils import create_folder
import pickle
class SeqProcessor(object):

    def __init__(self, tokenizer = None):
        self.global_config = StaticConfig()
        self.tokenizer = tokenizer

    def set_tokenizer(self, tok):
        self.tokenizer = tok

    # The function "text_to_wordlist" is from
    # https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
    def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.

        # Convert words to lower case and split them
        text = text.lower().split()

        # Optionally, remove stop words
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]

        text = " ".join(text)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            stemmed_words = [stemmer.stem(word) for word in text]
            text = " ".join(stemmed_words)

        # Return a list of words
        return (text)
    def extract_y(self, train):
        list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        return train[list_classes]

    def extract_x(self, train):
        list_sentences_train = train["comment_text"].fillna("CVxTz").values
        list_tokenized_train = self.tokenizer.texts_to_sequences(list_sentences_train)
        return sequence.pad_sequences(list_tokenized_train, maxlen=self.global_config.maxlen)


    def preprocess_train(self, train, submission=False):
        # train = train.sample(frac=1)
        if not submission:
            return (self.extract_x(train), self.extract_y(train))
        else:
            return (self.extract_x(train),None)
    def preprocess_test(self,test):
        # create_folder(model_save_folder_path)
        # create_folder("{}/{}".format(model_save_folder_path, sub_folder))
        # tokenizer_save_path = "{}/{}/{}".format(model_save_folder_path, sub_folder,sample+"_"+
        #                                         self.global_config.tokenizer_save_name)
        # pickle.dump(self.tokenizer, open(tokenizer_save_path, "wb"))
        return self.extract_x(test)

    # def prepare_data_folder(self, train_input_path, output_folder_path, train_test_factor=1.0, debug_factor=1.0):
    def prepare_data_folder(self, train_input_path, output_folder_path, train_test_factor=0.9, debug_factor=1.0):
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

        print("##################### preprocessor starts ########################")
        global_config = StaticConfig()
        create_folder(output_folder_path)
        label_cols = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        raw_data = pd.read_csv(train_input_path)
        raw_data = raw_data.sample(frac=1)
        self.tokenizer = text.Tokenizer(num_words=self.global_config.max_features)
        list_sentences_raw_data = raw_data["comment_text"].fillna("CVxTz").values
        self.tokenizer.fit_on_texts(list(list_sentences_raw_data))
        pickle.dump(self.tokenizer, open('{}/{}'.format(output_folder_path, self.global_config.tokenizer_save_name), "wb"))


        train_data_size = int(raw_data.shape[0] * train_test_factor * debug_factor)
        train = pd.DataFrame(raw_data[:train_data_size],columns = label_cols)

        if self.global_config.use_raw_for_test:
            test = pd.DataFrame(raw_data)
        else:
            test = pd.DataFrame(raw_data[train_data_size:] if debug_factor == 1.0 else raw_data[-100:])
        test_name = "{}/{}".format(output_folder_path,"test.csv")
        test.to_csv(test_name)

        for label_name in global_config.model_names:
            label_output = output_folder_path +"/"+ label_name
            create_folder(label_output)
            sub_train_output_file_path = '{}/tr_train_{}.csv'.format(label_output, label_name)
            train.to_csv(sub_train_output_file_path)  # , index=False)
            print ('output train for No. {} subset to file '.format(label_name, sub_train_output_file_path))


if __name__ == "__main__":
    wrapper = SeqProcessor()
    wrapper.prepare_data_folder('./input/train.csv', './preprocessing_wrapper_demo_output',
                                train_test_factor = StaticConfig().train_test_factor, debug_factor=1.0)


