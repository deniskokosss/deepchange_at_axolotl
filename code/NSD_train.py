import json
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

def train(dataset, model_path):
    X, y = dataset.drop(columns=['is_outlier', 'usage_id']).astype(float).values, dataset['is_outlier'].astype(int)
    clf = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(random_state=42))])
    clf.fit(X, y)
    with open(model_path,'wb') as f:
        pickle.dump(clf,f)


def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--embeds", "-e", help="Path to the list of json files with GlossReader embeddings, seperated with ','", required=True)
    arg("--dataset", "-d", help="Path to train dataset TSV file", required=True)
    arg("--model", "-m", help="Path to the output NSD model .pkl file", required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.dataset, sep='\t')
    embs_list = []
    for embeds_path in args.embeds.split(','):
        with open(embeds_path, 'r') as f:
            embs_list.append(json.load(f))

    dataset = create_dataset(df, embs_list)
    train(dataset, args.model)


if __name__ == '__main__':
    main()
