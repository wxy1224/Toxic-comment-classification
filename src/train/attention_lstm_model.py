
from src.train.abstract_model import BaseModel
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from src.config.static_config import StaticConfig
from src.config.dynamic_config import DynamicConfig
from keras import metrics
from keras.regularizers import l2
from keras.layers import Conv1D, MaxPooling1D
from src.extern.attention_lstm.attention_lstm import model_attention_applied_after_lstm
class Attention_LSTM_Model(BaseModel):
    def __init__(self):
        # self._model = None
        self.global_config = StaticConfig()
        self.dynamic_config = DynamicConfig()
        # self.num_called = 0

    def get_model(self, count, lstm_length=50, dense_dim=30, drop_out = 0.1, preprocessor=None):

        lstm_length = self.dynamic_config.config[count]['lstm_length']
        maxlen = self.global_config.maxlen

        model = model_attention_applied_after_lstm(maxlen,lstm_length, input_dim=1,output_dim=6)

        return model

