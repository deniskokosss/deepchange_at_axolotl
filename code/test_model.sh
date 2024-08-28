set -eux

python gr_vectorize.py \
    --model ../data/models/glossreader_retrain7_FiEnRu/checkpoint-22900/model.safetensors \
    --out_file ../data/embedings/test.json

for dataset in  "../axolotl24_shared_task/data/finnish/axolotl.test.fi.gold.tsv" \
                "../data/add_index/axolotl.test.ru.gold.tsv" \
                "../axolotl24_shared_task/data/german/axolotl.test.surprise.gold.tsv"; do
    fname=$(basename "$dataset")    
    out_file="../data/predictions/wsd_preds/test_$fname"
    echo $out_file

    python gr_wsd.py \
        --vectors_file ../data/embedings/test.json \
        --dataset $dataset \
        --pred $out_file

    mkdir -p ../results/WSD_test
    python ../axolotl24_shared_task/code/evaluation/scorer_track1.py \
        --gold $dataset \
        --pred $out_file \
        -o ../results/WSD_test/$fname

done