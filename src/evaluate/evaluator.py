import pandas as pd
from src.config.static_config import StaticConfig
from sklearn import metrics
import json
import numpy as np
import sys
from src.utils.utils import is_dir_exist, list_files_under_folder, create_folder

class Evaluator(object):

    def __init__(self, label_file_path):
        self.global_config = StaticConfig()
        self.label = pd.read_csv(label_file_path)
    def auc(self, test, subm):
        aucs = []
        for label in self.global_config.labels:
            true = np.array(test[label].values)
            pred = np.array(subm[label].values)
            try:
                auc = metrics.roc_auc_score(true, pred)
            except:
                auc = 0.0
            print(label, auc)
            aucs.append(auc)
        avg_auc = np.average(np.array(aucs))
        result = {
            'avg_auc':avg_auc,
            'aucs':aucs
        }
        return result

    def compute_auc(self, predict_folder_path, predict_file_name, evaluation_folder_path):

        predict = pd.read_csv("{}/{}".format(predict_folder_path, predict_file_name))

        with open('{}/{}'.format(evaluation_folder_path, predict_file_name + "__" + self.global_config.auc_file_name), 'w') as outfile:
            output =self.auc(self.label, predict)
            # json.dump(output, outfile, ensure_ascii=False)
            json_str = json.dumps(output)
            outfile.write(json_str)

    def evaluate(self, predict_folder_path, evaluation_folder_path):
        print("##################### evaluation starts ########################")
        create_folder(evaluation_folder_path)
        self.compute_auc( predict_folder_path, self.global_config.ensembled_predict_file_name,
                         evaluation_folder_path)
        for label_name in self.global_config.model_names:

            curr_predict_folder_path = "{}/{}".format(predict_folder_path, label_name)
            if not is_dir_exist(curr_predict_folder_path):
                continue
            curr_eval_folder_path = "{}/{}".format(evaluation_folder_path, label_name)
            create_folder(curr_eval_folder_path)
            files_to_eval = list_files_under_folder(curr_predict_folder_path)
            for file_name in files_to_eval:
                self.compute_auc(curr_predict_folder_path, file_name, curr_eval_folder_path)
        print("##################### evaluation ends ########################")
if __name__ == '__main__':
    # predict_folder_path = "./predict_demo_output/"
    predict_folder_path = sys.argv[1] # ./predict_demo_output/
    evaluate_folder_path = sys.argv[2]#"./evaluate_demo_output/"
    original_label_full_path = "{}/{}".format(predict_folder_path, "original_label_save.csv")
    evaluator = Evaluator(original_label_full_path)
    evaluator.evaluate(predict_folder_path,evaluate_folder_path)





