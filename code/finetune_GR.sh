
set -eux 

if [ ! -e ../data/english/semcor.train.en.tsv ]; then
    pip install nltk
    rm WSD_Evaluation_Framework.zip
    rm -rf WSD_Evaluation_Framework
    wget http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip
    unzip WSD_Evaluation_Framework

    python reformat_semcor.py
fi

if [ ! -e "../data/models/GR/model.pt" ]; then
    cd ../data/models
    mkdir -p GR
    wget https://zenodo.org/records/10530146/files/best_model.ckpt -O GR/model.pt
    cd -
fi

# accelerate launch --num-processes 8 finetune_GR.py --dataset FiEnRu
# accelerate launch --num-processes 8 finetune_GR.py --dataset Fi --sg
# accelerate launch --num-processes 8 finetune_GR.py --dataset Ru

# We trained the models on 8xA100 80GB VRAM (~6 hours)
# You may need to tweek training parameters for your hardware:
# If you have 1xA100 80GB
accelerate launch --num-processes 1 finetune_GR.py --dataset FiEnRu --ga 8
# If you have 1xA100 40GB or 1xV100
# accelerate launch --num-processes 1 finetune_GR.py --dataset FiEnRu --batch 4 --ga 16

# refer to test_model.sh for evaluation