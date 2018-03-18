
The driver is in the execute.sh

All the external code is located src/extern and src/char_cnn_baseline. The code introduced are MIT or Apache license on github.
You can run bash execute.sh to run one of the model.
For example, you go to execute.sh and uncomment the attention block. You will be training for attention model.

The pre-requisite is the data file under ./input folder. Just put all the

Download stanford NER https://nlp.stanford.edu/software/CRF-NER.html to udr data_preprocess folder under root
and do data augmentation by running daya_augmentation under data_process.

Please use conda envirinment with requirement file to set up environments with python 3.6

