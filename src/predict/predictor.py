from src.preprocess.preprocessor import SeqProcessor
from src.config.static_config import StaticConfig
from src.train.bidirectional_lstm_model import Bidirectional_LSTM_Model
from src.train.pretrained_embedding_bidirectional_lstm_model import Bidirectional_LSTM_Model_Pretrained_Embedding
from src.train.attention_lstm_model import Attention_LSTM_Model
from src.train.bidirectional_lstm_model_layers_above import Bidirectional_LSTM_Layers_Model
import pandas as pd
import pickle
import numpy as np
from src.utils.utils import list_files_under_folder, create_folder, is_dir_exist
import sys
from src.train.bidirectional_lstm_model_layers_no_embedding import Bidirectional_LSTM_Model_Layers_No_Embedding
class Predictor(object):
    def __init__(self):
        self.x_test = None
        self.global_config = StaticConfig()
        self.preprocessor = None

    def load_data(self, test_data_file_path, preprocessor_folder_path):
        self.x_test = pd.read_csv(test_data_file_path)
        tokenizer = pickle.load(open('{}/{}'.format(preprocessor_folder_path, self.global_config.tokenizer_save_name)
                , "rb"))
        self.preprocessor = SeqProcessor(tokenizer)

    def predict(self, empty_model_object, models_parent_folder_path, prediction_save_path, submission=False
                , load_sample_submission_file_path=None, use_attention= False):
        print("##################### predict starts ########################")
        create_folder(prediction_save_path)
        if not submission:
            original_labels = self.preprocessor.extract_y(self.x_test)
            original_labels.to_csv("{}/{}".format(prediction_save_path, self.global_config.original_label_file_name))
        predict = None

        for folder_name in self.global_config.model_names:
            print("processing ",folder_name)
            x_test = self.preprocessor.preprocess_train(self.x_test, submission)
            predict_for_model = self.predict_for_model_under_same_folder(
                empty_model_object.get_model(folder_name, preprocessor=self.preprocessor),
                                                                         models_parent_folder_path,
                                                                         folder_name,
                                                                         prediction_save_path, x_test[0], is_attention=use_attention)
            if predict_for_model is None:
                continue
            if predict is None:
                predict = predict_for_model
            else:
                predict += predict_for_model
        predict = predict/len(self.global_config.labels)

        if submission:
            sample = pd.read_csv(load_sample_submission_file_path)
            sample[self.global_config.labels] = predict
            sample.to_csv("{}/{}".format(prediction_save_path, self.global_config.ensembled_submission_file_name), index=False)

        else:
            predict.to_csv("{}/{}".format(prediction_save_path, self.global_config.ensembled_predict_file_name))
        print("##################### predict ends ########################")

    def predict_for_model_under_same_folder(self,
                                            empty_model_object,
                                            models_folder,
                                            folder_name,
                                            prediction_save_path, x_test, is_attention=False):
        model_path = "{}/{}".format(models_folder, folder_name)
        print('predict_for_model_under_same_folder model_path', model_path)
        if not is_dir_exist(model_path):
            return None
        model_names = list_files_under_folder(model_path)
        y_test_list = []
        model_folder = prediction_save_path + "/" + folder_name
        create_folder(model_folder)

        for model_name in model_names:
            one_model_path = model_path+"/"+model_name
            new_model = empty_model_object
            new_model.load_weights(one_model_path)
            if is_attention:
                x_test = np.expand_dims(x_test, axis=-1)
            y_test = new_model.predict(x_test, batch_size=self.global_config.batch_size)

            save_path_for_one = model_folder +"/"+self.global_config.predict_save_name
            pd.DataFrame(y_test,columns=self.global_config.labels).to_csv(save_path_for_one)
            y_test_list.append(y_test)
            if not self.global_config.enable_rebalancing_sampling:
                break
        average_y_test = y_test_list[0]
        average_df = pd.DataFrame(average_y_test, columns = self.global_config.labels)
        # average_df.to_csv(save_path_for_average)
        return average_df



if __name__ == '__main__':
    predictor = Predictor()
    # predictor.load_data('./input/test.csv', "./preprocessing_wrapper_demo_output/")
    test_file_location = sys.argv[1] # ./preprocessing_wrapper_demo_output/test.csv
    preprocessing_folder = sys.argv[2] # ./preprocessing_wrapper_demo_output/
    training_folder = sys.argv[3] # ../training_demo_output_augmented
    predict_folder = sys.argv[4] # ./predict_demo_output_5_augmented
    use_att = (sys.argv[5] == 'use_att')
    use_layers = (sys.argv[5] == 'use_layers')
    use_no_embedding = (sys.argv[5] == 'use_no_embedding')
    use_two_layers = (sys.argv[5] == 'use_two_layers')
    predictor.load_data(test_file_location, preprocessing_folder)


    if use_layers:
        predictor.predict(Bidirectional_LSTM_Layers_Model(), training_folder, predict_folder)
    elif use_no_embedding:
        predictor.predict(Bidirectional_LSTM_Model_Layers_No_Embedding(), training_folder, predict_folder)
    elif use_att:
        predictor.predict(Attention_LSTM_Model(), training_folder, predict_folder, use_attention=use_att)
    elif use_two_layers:
        predictor.predict(Attention_LSTM_Model(), training_folder, predict_folder)
    else:
        predictor.predict(Bidirectional_LSTM_Model(), training_folder, predict_folder)
