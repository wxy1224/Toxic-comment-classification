#!/bin/sh
#python src/preprocess/preprocessor.py 2&> log.txt
MODEL_INDEX=7
TRAIN_FOLDER=training_demo_output_${MODEL_INDEX}
PREPROCESS_FOLDER=preprocessing_wrapper_demo_output
PREDICT_FOLDER=predict_demo_output_${MODEL_INDEX}
EVALUATE_FOLDER=evaluate_demo_${MODEL_INDEX}_output
SUBMIT_FOLDER=submit_demo_output_${MODEL_INDEX}
INPUT=./input/test.csv
SAMPLE_FILE=./input/sample_submission.csv
#rm -rf ${TRAIN_FOLDER} ${PREDICT_FOLDER} ${EVALUATE_FOLDER} ${SUBMIT_FOLDER}
#echo "python src/train/trainer.py ./${TRAIN_FOLDER} ./${PREPROCESS_FOLDER} 2&> log.txt"
#python src/train/trainer.py ./${TRAIN_FOLDER} ./${PREPROCESS_FOLDER} use_att 2&> log.txt
#python src/predict/predictor.py ./${PREPROCESS_FOLDER}/test.csv ./${PREPROCESS_FOLDER}/ \
#  ./${TRAIN_FOLDER} ./${PREDICT_FOLDER} use_att 2&>> log.txt
#python src/evaluate/evaluator.py ./${PREDICT_FOLDER}/ ./${EVALUATE_FOLDER} use_att 2&>> log.txt
#python src/submission/submitter.py ${INPUT} ./${PREPROCESS_FOLDER}/ \
# ./${TRAIN_FOLDER} \
# ./${SUBMIT_FOLDER} ${SAMPLE_FILE} use_att 2&>> log.txt






echo "python src/train/trainer.py ./${TRAIN_FOLDER} ./${PREPROCESS_FOLDER} use_layers 2&> log.txt"
#python src/train/trainer.py ./${TRAIN_FOLDER} ./${PREPROCESS_FOLDER} use_layers 2&> log.txt
echo "python src/predict/predictor.py ./${PREPROCESS_FOLDER}/test.csv ./${PREPROCESS_FOLDER}/ \
  ./${TRAIN_FOLDER} ./${PREDICT_FOLDER} use_layers 2&>> log.txt"
python src/predict/predictor.py ./${PREPROCESS_FOLDER}/test.csv ./${PREPROCESS_FOLDER}/ \
  ./${TRAIN_FOLDER} ./${PREDICT_FOLDER} use_layers 2&>> log.txt
python src/evaluate/evaluator.py ./${PREDICT_FOLDER}/ ./${EVALUATE_FOLDER} use_layers  2&>> log.txt
python src/submission/submitter.py ${INPUT} ./${PREPROCESS_FOLDER}/ \
 ./${TRAIN_FOLDER} \
 ./${SUBMIT_FOLDER} ${SAMPLE_FILE}  use_layers 2&>> log.txt