
from src.train.abstract_model import BaseModel
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.config.static_config import StaticConfig
from keras import metrics
class Bidirectional_LSTM_Model(BaseModel):
    def __init__(self):
        self._model = None
        self.global_config = StaticConfig()

    def get_model(self):
        if not self._model is None:
            return self._model
        embed_size = self.global_config.lstm_embed_size
        max_features = self.global_config.max_features
        maxlen = self.global_config.maxlen
        inp = Input(shape=(maxlen,))
        x = Embedding(max_features, embed_size)(inp)
        x = Bidirectional(LSTM(50, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[metrics.categorical_accuracy])
        # print(model.summary())
        self._model = model
        return model