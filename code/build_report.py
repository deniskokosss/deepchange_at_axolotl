import pandas as pd
import os

datasets = {
    'axolotl.test.fi.gold.tsv': 'fi',
    'axolotl.test.ru.gold.tsv': 'ru',
    'axolotl.test.surprise.gold.tsv': 'de',
}
res = []
for method in os.listdir("../results"):
    if os.path.isdir(f"../results/{method}"):
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
