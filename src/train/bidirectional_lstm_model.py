
from src.train.abstract_model import BaseModel
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from src.config.static_config import StaticConfig
from src.config.dynamic_config import DynamicConfig
from keras import metrics
from keras.regularizers import l2
from keras.layers import Conv1D, MaxPooling1D
class Bidirectional_LSTM_Model(BaseModel):
    def __init__(self):
        # self._model = None
        self.global_config = StaticConfig()
        self.dynamic_config = DynamicConfig()
        # self.num_called = 0

    def get_model(self, count, lstm_length=50, dense_dim=30, drop_out = 0.1, preprocessor=None):
        lstm_length = self.dynamic_config.config[count]['lstm_length']
        dense_dim = self.dynamic_config.config[count]['dense_dim']
        drop_out = self.dynamic_config.config[count]['drop_out']
        embed_size = self.global_config.lstm_embed_size
        max_features = self.global_config.max_features
        maxlen = self.global_config.maxlen
        kernel_size = self.global_config.cnn_kernel_size
        filters = self.global_config.cnn_filters
        pool_size = self.global_config.cnn_pool_size

        inp = Input(shape=(maxlen,))
        x = Embedding(max_features, embed_size)(inp)
        # x = Dropout(drop_out)(x)
        if self.global_config.l2_regularizer != 0.0:
            regularizer = l2(self.global_config.l2_regularizer)
        x =Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1)(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = Bidirectional(LSTM(lstm_length, return_sequences=True,dropout_U = drop_out, dropout_W = drop_out))(x)
        x = Dense(6, activation="sigmoid")(x)


        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[metrics.categorical_accuracy])
        return model
