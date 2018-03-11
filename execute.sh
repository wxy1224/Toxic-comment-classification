#python src/preprocess/preprocessor.py 2&> log.txt
#python src/train/trainer.py 2&> log.txt
python src/train/trainer.py 2&>> log.txt
python src/predict/predictor.py 2&>> log.txt
python src/evaluate/evaluator.py 2&>> log.txt
python src/submission/submitter.py 2&>> log.txt
