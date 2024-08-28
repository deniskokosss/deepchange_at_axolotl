
set -eux 

if [ ! -e ../data/english/semcor.train.en.tsv ]; then
    pip install nltk
    rm WSD_Evaluation_Framework.zip
    rm -rf WSD_Evaluation_Framework
    wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
    unzip WSD_Evaluation_Framework

    python reformat_semcor.py
fi

# We trained the models on 8xA100, you may need to tweek training parameters (batch size, gradient accumulation, etc)
accelerate launch --num-processes 8 finetune_GR.py --dataset FiEnRu
# accelerate launch --num-processes 8 finetune_GR.py --dataset Fi --sg
# accelerate launch --num-processes 8 finetune_GR.py --dataset Ru