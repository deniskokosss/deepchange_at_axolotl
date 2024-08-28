#!/bin/bash

# bash repro.sh -c -> do not clear intermediate results before running the script
# bash repro.sh -d -> download embeddings instead of building them from scratch

set -eux  # exit on the first error (-e) or unset variable (-u), print commands (-x)

# remove the commited results, otherwise we will see them even if they are not reproduced!
rm -rf ../results/*/*tsv

# Define the options and their descriptions

opts=":d:c"
optstring="download embeddings instead of building them:use cache"

# Parse the options
REMOVE_CACHE=1
SKIP_EMBEDING=0
while getopts "$optstring" opt
do
    case $opt in
        c)
            REMOVE_CACHE=0
            ;;
        d)
            SKIP_EMBEDING=1
            ;;
        \?)
             "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
    esac
done

HOME=$(pwd)/..


if [ $REMOVE_CACHE != 0 ]; then
    echo "Removing cache"
    cd ..
    xargs rm -f < code/cached_files.txt
    cd -
fi

if [ ! -e "../axolotl24_shared_task" ]; then
    cd $HOME
    git clone https://github.com/ltgoslo/axolotl24_shared_task.git
    cd axolotl24_shared_task
    git reset --hard 6ccddbb6e536993ccb60cadbdb37fc95b68cbd9a
    # prepare german dataset
    cd $HOME/axolotl24_shared_task/data/german
    wget https://zenodo.org/records/8197553/files/dwug_de_sense.zip
    unzip dwug_de_sense.zip
    python surprise.py --dwug_path dwug_de_sense/
    cd $HOME/code
fi


echo $(pwd)

# -------Generate embeddings----------
if [ $SKIP_EMBEDING != 0 ]; then
    echo "Downloading embeddings"
    if [ ! -e glossreader_embs.zip ]; then 
        wget https://zenodo.org/records/11086527/files/glossreader_embs.zip
        unzip glossreader_embs.zip -d ../data/embedings/downloaded
        cp ../data/embedings/downloaded/glossreader.json ../data/embedings/GR.json
        cp ../data/embedings/downloaded/glossreader_fienru_v3.json ../data/embedings/GR_FiEnRu.json
        rm -r ../data/embedings/downloaded
    fi
else
    echo "Building embeddings"
    if [ ! -e "../data/models/GR_FiEnRu/model.safetensors" ]; then
        cd $HOME/data/models
        wget https://zenodo.org/records/13256679/files/GR_FiEnRu.tar.gz
        tar -xvzf GR_FiEnRu.tar.gz
        cd -
    fi
    if [ ! -e "../data/models/GR/model.pt" ]; then
        cd $HOME/data/models
        mkdir -p GR
        wget https://zenodo.org/records/10530146/files/best_model.ckpt -O GR/model.pt
        cd -
    fi

    if [ ! -e "../data/embedings/GR_FiEnRu.json" ]; then
        echo "Vectorizing FiEnRu"
        python gr_vectorize.py \
            --model ../data/models/GR_FiEnRu/model.safetensors \
            --out_file ../data/embedings/GR_FiEnRu.json
    fi
    if [ ! -e "../data/embedings/GR.json" ]; then
        echo "Vectorizing GR"
        python gr_vectorize.py \
            --model ../data/models/GR/model.pt \
            --out_file ../data/embedings/GR.json
    fi
fi

# We use our version russian dataset, where target word positions are added
for dataset in  "../axolotl24_shared_task/data/finnish/axolotl.test.fi.gold.tsv" \
                "../data/add_index/axolotl.test.ru.gold.tsv" \
                "../axolotl24_shared_task/data/german/axolotl.test.surprise.gold.tsv"; do
    fname=$(basename "$dataset")
    # # --------- WSD: GR FiEnRu --------- 
    out_file="../data/predictions/wsd_preds/GR_FiEnRu_$fname"
    echo $out_file

    python gr_wsd.py \
        --vectors_file ../data/embedings/GR_FiEnRu.json \
        --dataset $dataset \
        --pred $out_file

    # --------- WSI: Agglomerative --------- 
    # we currently use precomputed predictions
    python ../axolotl24_shared_task/code/evaluation/scorer_track1.py \
        --gold $dataset \
        --pred ../data/predictions/wsi_preds/$fname \
        -o ../results/WSI_agglomerative/$fname

    # --------- WSI: Agglomerative GR FiEnRu ---------
    mkdir -p ../results/WSI_agglomerative_improved 
    python Agglomerative.py \
        -e ../data/embedings/GR_FiEnRu.json \
        -d $dataset \
        -p ../data/predictions/wsi_preds/GR_FiEnRu_$fname
    
    python ../axolotl24_shared_task/code/evaluation/scorer_track1.py \
        --gold $dataset \
        --pred ../data/predictions/wsi_preds/GR_FiEnRu_$fname \
        -o ../results/WSI_agglomerative_improved/$fname



    #  SCM: Outlier2Cluster ru
    if [ ! -e ../data/models/NSD/NSD_russian.pkl ]; then
        python NSD_train.py \
            -e ../data/embedings/GR_FiEnRu.json,../data/embedings/GR.json \
            -d ../axolotl24_shared_task/data/russian/axolotl.dev.ru.tsv \
            -m ../data/models/NSD/NSD_russian.pkl
    fi
    python outlier2cluster.py \
        -e ../data/embedings/GR_FiEnRu.json,../data/embedings/GR.json \
        -d $dataset -m ../data/models/NSD/NSD_russian.pkl \
        --wsd ../data/predictions/wsd_preds/GR_FiEnRu_$fname \
        --wsi ../data/predictions/wsi_preds/$fname -t 0.65 \
        -p ../data/predictions/outlier2cluster/ru_$fname

    mkdir -p ../results/outlier2cluster_ru
    python ../axolotl24_shared_task/code/evaluation/scorer_track1.py \
        --gold $dataset \
        --pred ../data/predictions/outlier2cluster/ru_$fname \
        -o ../results/outlier2cluster_ru/$fname

    #  SCM: Outlier2Cluster ru improved
    python outlier2cluster.py \
        -e ../data/embedings/GR_FiEnRu.json,../data/embedings/GR.json \
        -d $dataset -m ../data/models/NSD/NSD_russian.pkl \
        --wsd ../data/predictions/wsd_preds/GR_FiEnRu_$fname \
        --wsi ../data/predictions/wsi_preds/GR_FiEnRu_$fname -t 0.65 \
        -p ../data/predictions/outlier2cluster/ru_improved_$fname

    mkdir -p ../results/outlier2cluster_ru_improved
    python ../axolotl24_shared_task/code/evaluation/scorer_track1.py \
        --gold $dataset \
        --pred ../data/predictions/outlier2cluster/ru_improved_$fname \
        -o ../results/outlier2cluster_ru_improved/$fname


    #  SCM: cluster2sense
    python cluster2sense.py \
        --wsd ../data/predictions/wsd_preds/GR_FiEnRu_$fname \
        --wsi ../data/predictions/wsi_preds/$fname \
        -d $dataset \
        -p ../data/predictions/cluster2sense/$fname

    mkdir -p ../results/cluster2sense
    python ../axolotl24_shared_task/code/evaluation/scorer_track1.py \
        --gold $dataset \
        --pred ../data/predictions/cluster2sense/$fname \
        -o ../results/cluster2sense/$fname

    #  SCM: cluster2sense_improved
    python cluster2sense.py \
        --wsd ../data/predictions/wsd_preds/GR_FiEnRu_$fname \
        --wsi ../data/predictions/wsi_preds/GR_FiEnRu_$fname \
        -d $dataset \
        -p ../data/predictions/cluster2sense/improved_$fname

    mkdir -p ../results/cluster2sense_improved
    python ../axolotl24_shared_task/code/evaluation/scorer_track1.py \
        --gold $dataset \
        --pred ../data/predictions/cluster2sense/improved_$fname \
        -o ../results/cluster2sense_improved/$fname

done

python build_report.py | tee ../results/table.txt
