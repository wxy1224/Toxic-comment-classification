from src.config.static_config import StaticConfig
from src.train.bidirectional_lstm_model import Bidirectional_LSTM_Model
from src.predict.predictor import Predictor
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src.neural_ensembling.feedforward_ensembling_model import FeedforwardEnsemblingModel
from src.utils.utils import create_folder, is_dir_exist
import pandas as pd
import numpy as np
class NeuralEnsembler(object):

    def __init__(self, label_file_path):
        self.global_config = StaticConfig()
        self.label =self.path_to_numpy_array(label_file_path)
        self.existing_predicts = []
        self.model = FeedforwardEnsemblingModel()

    def path_to_numpy_array(self, path):
        return pd.read_csv(path)[self.global_config.labels]

    def load_models_to_ensemble(self, predict_output_path):
        for model_name in self.global_config.model_names:

            curr_predict_folder_path = "{}/{}".format(submit_folder_path, model_name)
            if not is_dir_exist(curr_predict_folder_path):
                continue
            model_output_file = "{}/{}/{}".format(predict_output_path, model_name, self.global_config.predict_save_name)
            self.existing_predicts.append(self.path_to_numpy_array(model_output_file))
            self.ensemble_raw_data = np.concatenate(self.existing_predicts, axis=-1)


    def train_ensembler(self, ensemble_folder_path):
        print("##################### train_ensembler starts ########################")
        create_folder(ensemble_folder_path)

        keras_model = self.model.get_model()
        #

        create_folder(ensemble_folder_path)
        file_path = "{}/{}".format(ensemble_folder_path, self.global_config.ensemble_model_save_name)
        #
        checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        early = EarlyStopping(monitor="val_loss", mode="min", patience=self.global_config.patience)
        callbacks_list = [checkpoint, early]
        #
        keras_model.fit(self.ensemble_raw_data,self.label, batch_size=self.global_config.batch_size,
                  epochs=self.global_config.epoches, validation_split=self.global_config.validation_split,
                  callbacks=callbacks_list)

        print("##################### train_ensembler ends ########################")

    def ensembler_predict(self,ensemble_folder_path,load_sample_submission_file_path):
        print("##################### ensembler_predict starts ########################")
        create_folder(ensemble_folder_path)
        keras_model = self.model.get_model()
        keras_model.load_weights("{}/{}".format(ensemble_folder_path, self.global_config.ensemble_model_save_name))
        y_test = keras_model.predict(self.ensemble_raw_data)
        # y_test_df = pd.DataFrame(y_test, columns = self.global_config.labels)
        print('y_test.shape',y_test.shape)

        sample = pd.read_csv(load_sample_submission_file_path)
        print('Sample shape, ', sample.shape)
        sample[self.global_config.labels] = y_test
        sample.to_csv("{}/{}".format(ensemble_folder_path, self.global_config.neural_ensembled_predict_file_name),
                      index=False)
        # y_test_df.to_csv("{}/{}".format(ensemble_folder_path, self.global_config.neural_ensembled_predict_file_name), index=False)
        print("##################### ensembler_predict ends ########################")



if __name__ == '__main__':
    submit_folder_path = "./submit_demo_output/"
    neural_ensemble_path = "./ensemble_demo_output/"
    predict_folder_path = './predict_demo_output'
    original_label_full_path = "{}/{}".format(predict_folder_path, "original_label_save.csv")
    ensembler = NeuralEnsembler(original_label_full_path)

    # ensembler.load_models_to_ensemble(predict_folder_path)
    # ensembler.train_ensembler(neural_ensemble_path)
    #
    ensembler.load_models_to_ensemble(submit_folder_path)
    ensembler.ensembler_predict(neural_ensemble_path, "./input/sample_submission.csv")


