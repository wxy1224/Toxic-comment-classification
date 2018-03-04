
from src.config.static_config import StaticConfig
from src.train.bidirectional_lstm_model import Bidirectional_LSTM_Model
from src.predict.predictor import Predictor
import pandas as pd
class Submitter(object):

    def __init__(self):
        self.global_config = StaticConfig()
        self.predictor = Predictor()

    def load_data(self):
        self.predictor.load_data('./input/test.csv', "./preprocessing_wrapper_demo_output_0/")
    def submit(self):
        self.predictor.predict(Bidirectional_LSTM_Model().get_model(), './training_demo_output_0',
                               './submit_demo_output_0', submission=True, load_sample_submission_file_path='./input/sample_submission.csv')



if __name__ == '__main__':
    # submitter = Submitter()
    # submitter.load_data()
    # submitter.submit()
    original = pd.read_csv("./submit_demo_output_0/submission_predict_file.csv")
    original = original[["id","toxic","severe_toxic","obscene","threat","insult","identity_hate"]]
    original.to_csv("./submit_demo_output_0/submission_predict_file_1.csv", index=False)

