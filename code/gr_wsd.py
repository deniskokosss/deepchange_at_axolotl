
import json
import pandas as pd

import numpy as np
from types import SimpleNamespace
from tqdm import tqdm

import seaborn as sns
import subprocess

from torch.nn.functional import softmax
import torch
import argparse

def to_args(s):
    return [t for t in s.split() if t]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_file', type=str, default="../data/embedings/GR_FiEnRu.json")
    parser.add_argument('--dataset', type=str, default="../data/test/axolotl.test.surprise.tsv")
    parser.add_argument('--senses', type=str, default='old') # other inference types are experimental
    parser.add_argument('--pred', type=str, default='../data/predictions/wsd_preds/GR_FiEnRu_axolotl.test.surprise.tsv')
    args = parser.parse_args()
    vectors_file = args.vectors_file
    dataset = args.dataset
    senses = args.senses
    args.sense_inventory = args.senses


    df = pd.read_csv(dataset, sep='\t')
    with open(args.vectors_file, 'r') as f:
        vectors = json.load(f)
    res = []
    for word, df_word in tqdm(df.groupby('word'), total=df.word.nunique()):
        if (args.sense_inventory == "old"):
            sense_ids = df_word.loc[df_word['period'] == 'old', 'sense_id'].unique()
        elif (args.sense_inventory == "oldthr"):
            sense_ids = df_word.loc[df_word['period'] == 'old', 'sense_id'].unique()
        elif (args.sense_inventory == "oldunk") or (args.sense_inventory == "old+mono"):
            sense_ids = np.concatenate([df_word.loc[df_word['period'] == 'old', 'sense_id'].unique(), ['unk']])
        elif args.sense_inventory == "new":
            sense_ids = df_word.loc[df_word['period'] == 'new', 'sense_id'].unique()
        elif args.sense_inventory == "oldnew":
            sense_ids = df_word.loc[:, 'sense_id'].unique()
            
        sense_vectors = np.array([vectors['glosses'][t] for t in sense_ids])
        sense_vectors = sense_vectors[:, :] #  gloss vectors are stored with extra dimension

        df_new = df_word.loc[df_word['period'] == 'new', :].copy()
        contexts_ids = df_new['usage_id'].unique()
        context_vectors = np.array([vectors['contexts'][t] for t in contexts_ids])

        # print((context_vectors @ sense_vectors.T)[:, -1])
        most_similar = (context_vectors @ sense_vectors.T).argmax(axis=1)
        if args.sense_inventory == "oldthr":
            most_similar = torch.cat([
                0.06 * torch.ones(context_vectors.shape[0], 1),
                softmax(torch.as_tensor(context_vectors @ sense_vectors.T), dim=-1)
            ], dim=-1).numpy()
            print(most_similar)
            most_similar = most_similar.argmax(axis=1)
            sense_ids = np.concatenate([['unk'], sense_ids])
        context_to_sense_preds = {
            contexts_ids[i]: sense_ids[pred]
            for i,pred in enumerate(most_similar)
        }
        df_new['sense_id'] = "-1"
        if args.sense_inventory == "old+mono" and len(sense_ids) == 2  :
            df_new['sense_id'] = sense_ids[0]
        else:
            df_new['sense_id'] = df_new['usage_id'].replace(context_to_sense_preds)
        res.append(pd.concat([df_word.loc[df_word['period'] == 'old', :], df_new]))

    preds = pd.concat(res).sort_index()
    preds.to_csv(args.pred, sep='\t', index=False)
    if 'dev' in dataset:
        p1 = subprocess.run(to_args(
            f"python3 evaluation/scorer_track1.py \
            --gold {dataset} \
            --pred {args.pred} \
            --output ../data/results/report{pred.split('/')[-1]}"
        ), cwd='/home/jovyan/kokosinskiy/projects/axolotl24_shared_task/code')