import pandas as pd
import src.config.static_config as static_config
import os
class preprocessing_wrapper(object):

    def create_folder(self, path):
        import os
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    def prepare_data_folder(self, train_input_path, output_folder_path, train_test_factor, debug_factor=1.0):
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
        self.create_folder(output_folder_path)
        label_cols = ['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        raw_data = pd.read_csv(train_input_path)

        train_data_size = int(raw_data.shape[0] * train_test_factor * debug_factor)
        train = pd.DataFrame(raw_data[:train_data_size],columns = label_cols)

        test = pd.DataFrame(raw_data[train_data_size:] if debug_factor == 1.0 else raw_data[-100:],columns = label_cols)
        test_name = "{}/{}".format(output_folder_path,"test.csv")
        test.to_csv(test_name, index=False)


        for label_name in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
            label_output = output_folder_path +"/"+ label_name
            self.create_folder(label_output)
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
    wrapper = preprocessing_wrapper()
    wrapper.prepare_data_folder('./input/train.csv', './preprocessing_wrapper_demo_output',0.7,debug_factor=0.0001)

