import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances


def agglomerative(points, labels, affinity, linkage, n_clusters):
    is_new = labels < 0
    all_dists = pairwise_distances(points, metric=affinity)
    while len(np.unique(labels)) > n_clusters:
        l1 = l2 = 0
        m = np.inf
        for i in np.unique(labels[labels < 0]):
            for j in np.unique(labels):
                if i != j:
                    dists = all_dists[labels == i][:, labels == j]
                    if linkage == 'single':
                        ans = np.min(dists)
                    elif linkage == 'complete':
                        ans = np.max(dists)
                    elif linkage == 'average':
                        ans = np.mean(dists)
                    if ans < m:
                        l1 = i
                        l2 = j
                        m = ans
        a = max(l1, l2)
        b = min(l1, l2)
        labels[labels == b] = a
    return labels

def map_labels(pred_labels, gold_labels):
    mapping = {}
    for l1, l2 in zip(gold_labels[gold_labels['period'] == 'old']['sense_id'].values, pred_labels[pred_labels['period'] == 'old']['sense_id'].values):
        mapping[l2] = l1
    mapped = list(map(lambda x: mapping[x] if x in mapping else str(x), pred_labels['sense_id']))
    return mapped

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--embeds", "-e", help="Path to the json file with GlossReader embeddings", required=True)
    arg("--dataset", "-d", help="Path to dataset TSV file", required=True)
    arg("--predict", "-p", help="Path for the output predictions TSV file", required=True)
    arg("--k", "-k", help="Hyperparameter k", required=True)
    arg("--linkage", "-l", help="Hyperparameter linkage", required=True)
    args = parser.parse_args()


    with open(args.embeds, 'r') as f:
        embs = json.load(f)['contexts']
    df = pd.read_csv(args.dataset, sep='\t')
    df['sense_id'] = df['sense_id'].fillna("-1")
    with tqdm(df.groupby('word')['usage_id']) as t:
        for name, i in t:
            cur_df = df[df['usage_id'].isin(i)]
            cur_embs = np.array([embs[id] if id in embs else embs[list(embs)[0]] for id in i.values])
            n_clusters = cur_df[cur_df['period'] != 'new']['sense_id'].nunique() + int(args.k)
            _, labels = np.unique(cur_df['sense_id'], return_inverse=True)
            labels[cur_df['period'] == 'new'] = -np.arange(1, len(labels[cur_df['period'] == 'new']) + 1)
            pred_labels = agglomerative(cur_embs, labels, distance.cosine, args.linkage, n_clusters)
            translated_labels = map_labels(pd.DataFrame({'sense_id': pred_labels, 'period': cur_df.period}), cur_df)
            df.loc[df['usage_id'].isin(i), 'predicted_sense_id'] = translated_labels
    df['sense_id'] = df['predicted_sense_id']
    df = df.drop(columns=['predicted_sense_id'])
    df.to_csv(args.predict, sep = '\t', index=False)


if __name__ == '__main__':
    main()
