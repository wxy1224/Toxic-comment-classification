
from src.train.abstract_model import BaseModel
from keras.models import Model
from keras.models import Sequential

import numpy as np
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

from src.config.static_config import StaticConfig
from src.config.dynamic_config import DynamicConfig
from keras import metrics
from keras.regularizers import L1L2
class FeedforwardEnsemblingModel(BaseModel):
    def __init__(self):
        self.global_config = StaticConfig()
        self.dynamic_config = DynamicConfig()

    def get_model(self, input_dim=6*18, hidden_dim = 6*18):
        drop_out = 0.1
        model = Sequential()
        # model.add(Dense(hidden_dim, input_dim=input_dim, activation='relu'))
        # model.add(Dropout(drop_out))
        model.add(Dense(6, activation='softmax',kernel_regularizer=L1L2(l1=0.0, l2=0.1),
input_dim=input_dim) )
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=[metrics.categorical_accuracy])
        # print(model.summary())
        # self._model = model
        return model