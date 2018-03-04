
from src.config.static_config import StaticConfig
from src.train.bidirectional_lstm_model import Bidirectional_LSTM_Model
from src.predict.predictor import Predictor
class Submitter(object):

    def __init__(self):
        self.global_config = StaticConfig()
        self.predictor = Predictor()

    def load_data(self):
        self.predictor.load_data('./input/test.csv', "./preprocessing_wrapper_demo_output_0/")
    def submit(self):
        self.predictor.predict(Bidirectional_LSTM_Model().get_model(), './training_demo_output_0',
                               './submit_demo_output_0', submission=True)



if __name__ == '__main__':
    submitter = Submitter()
    submitter.load_data()
    submitter.submit()

