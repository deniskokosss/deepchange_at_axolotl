import pandas as pd
import os

datasets = {
    'axolotl.test.fi.gold.tsv': 'fi',
    'axolotl.test.ru.gold.tsv': 'ru',
    'axolotl.test.surprise.gold.tsv': 'de',
}
res = []
for method in os.listdir("../results"):
    if os.path.isdir(f"../results/{method}") and len(os.listdir(f"../results/{method}")) != 0:
        m = {'method': method}
        for path,name in datasets.items():
            with open(f"../results/{method}/{path}") as f:
                for line in f.readlines():
                    metric,value = line.split(': ')
                    m[f'{metric}_{name}'] = float(value)
        res.append(m)
    
res = pd.DataFrame(res).set_index('method')
res = res[['ARI_fi', "ARI_ru", 'ARI_de', 'F1_fi', "F1_ru", 'F1_de']].sort_index()
print(res.round(3))

target_metrics = pd.read_csv('target_metrics.tsv',sep=' +', engine='python').set_index('method')
print('\nAre results from the paper reproduced?')
print( ((res-target_metrics).abs() < 0.002) )

target_metrics = pd.read_csv('target_metrics_upd.tsv',sep=' +', engine='python').set_index('method')
print('\nAre results from README reproduced?')
print( ((res-target_metrics).abs() < 0.002) )
