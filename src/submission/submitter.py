
from src.config.static_config import StaticConfig
from src.train.bidirectional_lstm_model import Bidirectional_LSTM_Model
from src.train.pretrained_embedding_bidirectional_lstm_model import Bidirectional_LSTM_Model_Pretrained_Embedding
from src.predict.predictor import Predictor
import pandas as pd
class Submitter(object):

    def __init__(self):
        self.global_config = StaticConfig()
        self.predictor = Predictor()

    def load_data(self):
        self.predictor.load_data('./input/test.csv', "./preprocessing_wrapper_demo_output/")
    def submit(self):
        self.predictor.predict(Bidirectional_LSTM_Model_Pretrained_Embedding(), './training_demo_output',
                               './submit_demo_output', submission=True, load_sample_submission_file_path='./input/sample_submission.csv')



if __name__ == '__main__':
    submitter = Submitter()
    submitter.load_data()
    submitter.submit()



