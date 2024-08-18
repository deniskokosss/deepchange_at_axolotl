import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score

def agglomerative(points):
    all_labels = []
    dists = pairwise_distances(points, metric=distance.cosine)
    for n_clusters in range(1, 10):
        if n_clusters <= len(points):
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average').fit(dists)
            all_labels.append(clustering.labels_)
    result = None
    score = None
    for labels in all_labels:
        try:
            cur_score = calinski_harabasz_score(points, labels)
        except:
            cur_score = -100
        if score is None or cur_score > score:
            result = labels
            score = cur_score
    return result

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--embeds", "-e", help="Path to the json file with GlossReader embeddings", required=True)
    arg("--dataset", "-d", help="Path to dataset TSV file", required=True)
    arg("--predict", "-p", help="Path for the output predictions TSV file", required=True)
    args = parser.parse_args()


    with open(args.embeds, 'r') as f:
        embs = json.load(f)['contexts']
    df = pd.read_csv(args.dataset, sep='\t')
    with tqdm(df.groupby('word')['usage_id']) as t:
        for name, i in t:
            cur_df = df[(df['usage_id'].isin(i)) & (df['period'] == 'new')]
            cur_embs = np.array([embs[id] if id in embs else embs[list(embs)[0]] for id in cur_df['usage_id'].values])
            pred_labels = agglomerative(cur_embs)
            df.loc[(df['usage_id'].isin(i)) & (df['period'] == 'new'), 'sense_id'] = pred_labels
    df.to_csv(args.predict, sep = '\t', index=False)


if __name__ == '__main__':
    main()
