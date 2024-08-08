import json
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances

def norm_dist(vec1, vec2, d):
    vec1 = vec1 / np.linalg.norm(vec1, ord=d)
    vec2 = vec2 / np.linalg.norm(vec2, ord=d)

    return np.linalg.norm(vec1 - vec2, ord=d)


def norm_l1(vec1, vec2):
    return norm_dist(vec1, vec2, d=1)


def norm_l2(vec1, vec2):
    return norm_dist(vec1, vec2, d=2)


def create_dataset(df, embs_list):
    result = pd.DataFrame()
    for word, i in df.groupby('word')['usage_id']:
        cur_df = df[df['usage_id'].isin(i)]
        features = get_features(cur_df, embs_list)
        result = pd.concat([result, features], ignore_index=True)
    return result


def get_features(df, embs_list):
    result = pd.DataFrame()
    result['usage_id'] = df[df.period=='new']['usage_id']
    for i, embs in enumerate(embs_list):
        senses = list(df[df.period == 'old']['sense_id'].unique())
        usages = df[df.period == 'new']['usage_id'].values
        old_glosses = np.array([embs['glosses'][i] for i in senses])
        new_embs = np.array([embs['contexts'][i] for i in usages])
        dot_products = new_embs @ old_glosses.T
        chosen_glosses = dot_products.argmax(axis=1)
        for dist in [distance.cosine, distance.cityblock, distance.euclidean, norm_l1, norm_l2]:
            cur_dists = pairwise_distances(new_embs, old_glosses, metric=dist)
            result[f'{dist.__name__}_{i}'] = cur_dists[np.arange(len(chosen_glosses)), chosen_glosses]
            
    result['n_old_usages'] = len(df[df.period == 'old'])
    result['n_new_usages'] = len(df[df.period == 'new'])
    result['n_old_senses'] = len(senses)
    
    new_senses = df[df.period == 'new'].sense_id.values
    old_senses = df[df.period == 'old'].sense_id.unique()
    is_outlier = []
    for sense in new_senses:
        is_outlier.append(sense not in old_senses)
    result['is_outlier'] = is_outlier

    return result

def predict(dataset, model, tresh=0.5):
    X = dataset.drop(columns=['is_outlier', 'usage_id']).astype(float).values
    preds = model.predict_proba(X)[:, 1] > tresh
    outliers = pd.DataFrame({'usage_id': dataset.usage_id, 'is_outlier': preds})
    return outliers

def prep_for_submit(wsi_preds, wsd_preds, outliers):
    wsd_preds = wsd_preds.merge(outliers, on='usage_id', how='left')
    wsd_preds['is_outlier'] = wsd_preds['is_outlier'].fillna(False)
    if wsi_preds is not None:
        wsi_preds['wsi_preds'] = wsi_preds['sense_id']
        wsd_preds = wsd_preds.merge(wsi_preds[['usage_id', 'wsi_preds']], on='usage_id')
        wsd_preds['sense_id'] = np.where(wsd_preds['is_outlier'], wsd_preds['wsi_preds'], wsd_preds['sense_id'])
        wsd_preds = wsd_preds.drop(columns=['wsi_preds', 'is_outlier'])
    else:
        wsd_preds['sense_id'] = np.where(wsd_preds['is_outlier'], "0", wsd_preds['sense_id'])
        wsd_preds = wsd_preds.drop(columns=['is_outlier'])
    return wsd_preds

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--embeds", "-e", help="Path to the list of json files with GlossReader embeddings, seperated with ','", required=True)
    arg("--dataset", "-d", help="Path to the dataset TSV file", required=True)
    arg("--model", "-m", help="Path to the trained NSD model .pkl file", required=True)
    arg("--wsd", help="Path to the input WSD predictions TSV file", required=True)
    arg("--wsi", help="Path to the input WSI predictions TSV file. Pass 'none' if not needed.", required=True)
    arg("--tresh", "-t", help="Treshold for NSD model", required=True)
    arg("--predict", "-p", help="Path for the output predictions TSV file", required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.dataset, sep='\t')
    wsd_preds = pd.read_csv(args.wsd, sep='\t')
    wsi_preds = pd.read_csv(args.wsi, sep='\t') if args.wsi != 'none' else None

    embs_list = []
    for embeds_path in args.embeds.split(','):
        with open(embeds_path, 'r') as f:
            embs_list.append(json.load(f))

    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    dataset = create_dataset(df, embs_list)
    outliers = predict(dataset, model, float(args.tresh))
    preds = prep_for_submit(wsi_preds, wsd_preds, outliers)
    preds.to_csv(args.predict, sep='\t', index=False)

if __name__ == '__main__':
    main()
