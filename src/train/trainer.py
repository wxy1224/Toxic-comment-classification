from src.config.static_config import StaticConfig
from src.utils.utils import list_files_under_folder, create_folder
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.preprocess.preprocessor import SeqProcessor
from src.train.bidirectional_lstm_model import Bidirectional_LSTM_Model
from src.train.pretrained_embedding_bidirectional_lstm_model import Bidirectional_LSTM_Model_Pretrained_Embedding


import pickle
class Trainer(object):
    def __init__(self):
        self.data_sets = []

        self.global_config = StaticConfig()
        self.preprocessor = None



    def load_data(self, train_data_folder_path, submission=False):
        tokenizer = pickle.load(open('{}/{}'.format(train_data_folder_path, self.global_config.tokenizer_save_name)
                , "rb"))
        self.preprocessor = SeqProcessor(tokenizer)
        for sub_folder in self.global_config.model_names:
            samples = list_files_under_folder("{}/{}/".format(train_data_folder_path, sub_folder))
            if len(samples) == 0:
                continue
            for sample in samples:
                full_sample_path = "{}/{}/{}".format(train_data_folder_path, sub_folder, sample)
                loaded_sample = pd.read_csv(full_sample_path)
                self.data_sets.append((self.preprocessor.preprocess_train(loaded_sample, submission)
                                       , train_data_folder_path, sub_folder, sample))

    def train(self, model, model_save_folder_path, attention_model=False):
        print("##################### training starts ########################")
        history_dic = {}
        create_folder(model_save_folder_path)
        for dataset in self.data_sets:
            x_train, y = dataset[0]
            batch_size = self.global_config.batch_size
            epochs = self.global_config.epoches
            model_to_train = model.get_model(dataset[2], preprocessor=self.preprocessor)
            model_save_path =  "{}/{}".format(model_save_folder_path, dataset[2])
            create_folder(model_save_path)
            file_path = "{}/{}".format(model_save_path, dataset[3]+"_"+self.global_config.model_save_name)
            # check point
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            early = EarlyStopping(monitor="val_loss", mode="min", patience=self.global_config.patience)
            callbacks_list = [checkpoint, early]
            if attention_model:
                x_train = np.expand_dims(x_train, axis=-1)
            hist = model_to_train.fit(x_train, y, batch_size=batch_size, epochs=epochs, validation_split=self.global_config.validation_split, callbacks=callbacks_list)
            print(hist)
            history_dic[dataset[2]] = hist
        print("##################### training ends ########################")
        return history_dic

if __name__ == '__main__':
    trainer = Trainer()
    output_path = './training_demo_output'
    trainer.load_data('./preprocessing_wrapper_demo_output')
    history_dic = trainer.train(Bidirectional_LSTM_Model_Pretrained_Embedding(), output_path)
    print(history_dic)