

class StaticConfig:
    preprocess_splits = 9
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    max_features = 20000
    maxlen = 100
    patience = 1
    validation_split = 0.5
    model_save_name = "weights_base.best.hdf5"
    tokenizer_save_name = "tokenizer_save.p"
    original_label_file_name = "original_label_save.csv"
    predict_save_name = "predict_save.csv"
    average_predict_save_name = "average_predict_save.csv"
    ensembled_predict_file_name = "ensembled_predict_file.csv"
    is_debug = True


