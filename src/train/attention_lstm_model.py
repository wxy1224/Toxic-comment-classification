
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

    def get_model(self, count, lstm_length=50, dense_dim=30, drop_out = 0.1):

        lstm_length = self.dynamic_config.config[count]['lstm_length']
        dense_dim = self.dynamic_config.config[count]['dense_dim']
        drop_out = self.dynamic_config.config[count]['drop_out']
        embed_size = self.global_config.lstm_embed_size
        max_features = self.global_config.max_features
        maxlen = self.global_config.maxlen
        kernel_size = self.global_config.cnn_kernel_size
        filters = self.global_config.cnn_filters
        pool_size = self.global_config.cnn_pool_size

        model = model_attention_applied_after_lstm(maxlen,lstm_length, input_dim=1,output_dim=6)
        # print(model.summary())
        # self._model = model
        return model


        # model = Sequential()
        # model.add(Embedding(max_features, embedding_size, input_length=maxlen))
        # model.add(Dropout(0.25))
        # model.add(Conv1D(filters,
        #                  kernel_size,
        #                  padding='valid',
        #                  activation='relu',
        #                  strides=1))
        # model.add(MaxPooling1D(pool_size=pool_size))
        # model.add(LSTM(lstm_output_size))
        # model.add(Dense(1))
        # model.add(Activation('sigmoid'))