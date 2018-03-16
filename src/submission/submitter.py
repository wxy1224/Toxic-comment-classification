
from src.config.static_config import StaticConfig
from src.train.bidirectional_lstm_model import Bidirectional_LSTM_Model
from src.train.pretrained_embedding_bidirectional_lstm_model import Bidirectional_LSTM_Model_Pretrained_Embedding
from src.train.attention_lstm_model import Attention_LSTM_Model
from src.train.bidirectional_lstm_model_layers_above import Bidirectional_LSTM_Layers_Model
from src.predict.predictor import Predictor
import pandas as pd
import sys
from src.train.bidirectional_lstm_model_layers_no_embedding import Bidirectional_LSTM_Model_Layers_No_Embedding
class Submitter(object):

    def __init__(self):
        self.global_config = StaticConfig()
        self.predictor = Predictor()

    def load_data(self, input_test_file, preprocessing_folder):
        # self.predictor.load_data('./input/test.csv', "./preprocessing_wrapper_demo_output/")
        self.predictor.load_data(input_test_file, preprocessing_folder)
    def submit(self,model, train_output_folder, submit_output_folder, sample_input_file, use_att= False):
        # self.predictor.predict(Bidirectional_LSTM_Model_Pretrained_Embedding(), './training_demo_output_augmented',
        #                        './submit_demo_output_5_augmented', submission=True, load_sample_submission_file_path='./input/sample_submission.csv')
        self.predictor.predict(model,
                               train_output_folder,
                               submit_output_folder, submission=True,
                               load_sample_submission_file_path=sample_input_file, use_attention=use_att)

if __name__ == '__main__':
    submitter = Submitter()
    input_test_file= sys.argv[1] #'./input/test.csv'
    preprocessing_folder= sys.argv[2] #"./preprocessing_wrapper_demo_output/"
    train_output_folder= sys.argv[3] # './training_demo_output_augmented'
    submit_output_folder= sys.argv[4] # ./submit_demo_output_5_augmented
    sample_input_file= sys.argv[5] # ./input/sample_submission.csv
    use_att = (sys.argv[6] == 'use_att')
    use_layers = (sys.argv[6] == 'use_layers')
    use_no_embedding = (sys.argv[6] == 'use_no_embedding')
    submitter.load_data(input_test_file, preprocessing_folder)
    if use_layers:
        submitter.submit(Bidirectional_LSTM_Layers_Model(), train_output_folder, submit_output_folder, sample_input_file)
    elif use_no_embedding:
        submitter.submit(Bidirectional_LSTM_Model_Layers_No_Embedding(), train_output_folder, submit_output_folder,
                         sample_input_file)
    elif use_att:
        submitter.submit(Attention_LSTM_Model(), train_output_folder, submit_output_folder, sample_input_file, use_att=use_att)
    else:
        submitter.submit(Bidirectional_LSTM_Model(), train_output_folder, submit_output_folder, sample_input_file)





