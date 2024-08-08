import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse

def jaccard_index(x, y):
    return len(set(x).intersection(set(y))) / len(set(x).union(set(y)))

def get_clusters(df):
    res = {}
    for name, df0 in df.groupby('sense_id'):
        res[name] = df0['usage_id'].unique()
    return res

def compute_jaccard(word):
    labels_word = labels[labels.word == word]
    clusters_word = clusters[clusters.word == word]
    labels_word
    clusters1 = get_clusters(labels_word)
    clusters2 = get_clusters(clusters_word)
    dists = np.zeros((len(clusters1), len(clusters2)))
    for i, (label, cluster_labels) in enumerate(clusters1.items()):
        for j, (cluster, examples) in enumerate(clusters2.items()):
            dists[i,j] = jaccard_index(cluster_labels, examples)

    return dists, list(clusters1.keys()), list(clusters2.keys())



def get_labels_map(dists, labels1, labels2):
    res = {}
    for i, label1 in enumerate(labels1):
        winner = np.argmax(dists[i])
        cond1 = len(np.where(dists[i] == dists[i, winner])) < 2 # label is a clear winner 
        cond2 = len(np.where(dists[:, winner] == dists[i, winner])) < 2 # cluster is a clear winner
        # cond3 = dists[i, winner] >= 0.5 # intersection is large enough
        if cond1 and cond2:
            res[labels2[winner]] = labels1[i]
    return res
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--dataset", "-d", help="Path to dataset TSV file", required=True)
    arg("--wsd", help="WSD predictions", required=True)
    arg("--wsi", help="WSI predictions", required=True)
    arg("--predict", "-p", help="Path for the output predictions TSV file", required=True)
    args = parser.parse_args()

    clusters = pd.read_csv(args.wsi, sep='\t')
    clusters = clusters[clusters['period'] == 'new']
    labels =  pd.read_csv(args.wsd, sep='\t')
    labels =  labels[labels['period'] == 'new']
    old = pd.read_csv(args.dataset, sep='\t')
    old = old[old['period'] == 'old']

    res = []
    clusters_c = clusters.copy()
    for i,word in enumerate(clusters_c.word.unique()):
        dists, labels1, labels2 = compute_jaccard(word)
        mapping = get_labels_map(dists, labels1, labels2)
        clusters_c.loc[clusters_c.word == word, 'sense_id'] = clusters_c.loc[clusters_c.word == word, 'sense_id'].replace(mapping)

    df = pd.concat([old, clusters_c]).reset_index(drop=True)
    # df['sorter'] = df['usage_id'].apply(lambda x: int(x.split('_')[-1]))
    # df = df.sort_values(by='sorter').drop(columns='sorter').reset_index(drop=True)
    df.to_csv(args.predict, sep='\t', index=False)