
from src.train.abstract_model import BaseModel
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.config.static_config import StaticConfig
from keras.layers.normalization import BatchNormalization
from src.config.dynamic_config import DynamicConfig
from keras import metrics
from numpy import zeros
from keras.regularizers import l2
from numpy import asarray
from keras.layers import Conv1D, MaxPooling1D
class Bidirectional_LSTM_Layers_Model(BaseModel):
    def __init__(self):
        self.global_config = StaticConfig()
        self.dynamic_config = DynamicConfig()

    def embedding_index(self):
        self.embeddings_index = dict()
        f = open('./input/glove.6B.300d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(self.embeddings_index))

    def get_model(self, count, lstm_length=50, dense_dim=30, drop_out = 0.1, preprocessor=None):
        self.embedding_index()
        tokenizer = preprocessor.tokenizer
        voc_size = len(tokenizer.word_index) + 1

        lstm_length = self.dynamic_config.config[count]['lstm_length']
        dense_dim = self.dynamic_config.config[count]['dense_dim']
        drop_out = self.dynamic_config.config[count]['drop_out']
        embed_size = self.global_config.lstm_embed_size
        max_features = self.global_config.max_features
        maxlen = self.global_config.maxlen
        kernel_size = self.global_config.cnn_kernel_size
        filters = self.global_config.cnn_filters
        pool_size = self.global_config.cnn_pool_size

        embedding_matrix = zeros((voc_size, embed_size))
        for word, i in tokenizer.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        inp = Input(shape=(maxlen,), dtype='int32')
        x = Embedding(voc_size, embed_size ,input_length = maxlen, weights=[embedding_matrix])(inp)

        x = Bidirectional(LSTM(maxlen, return_sequences=True, dropout=drop_out, recurrent_dropout=drop_out))(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(drop_out)(x)
        if self.global_config.l2_regularizer != 0.0:
            regularizer = l2(self.global_config.l2_regularizer)
            x = Dense(dense_dim, activation="relu", kernel_regularizer=regularizer)(x)
        else:
            x = Dense(dense_dim, activation="relu")(x)
        x = Dropout(drop_out)(x)
        if self.global_config.l2_regularizer != 0.0:
            regularizer = l2(self.global_config.l2_regularizer)
            x = Dense(dense_dim, activation="relu", kernel_regularizer=regularizer)(x)
        else:
            x = Dense(dense_dim, activation="relu")(x)

        x = Dropout(drop_out)(x)
        if self.global_config.l2_regularizer != 0.0:
            regularizer = l2(self.global_config.l2_regularizer)
            x = Dense(dense_dim, activation="relu", kernel_regularizer=regularizer)(x)
        else:
            x = Dense(dense_dim, activation="relu")(x)

        x = Dropout(drop_out)(x)
        x = Dense(6, activation="sigmoid")(x)


        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[metrics.categorical_accuracy])
        return model