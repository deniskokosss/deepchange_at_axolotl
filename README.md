# [AXOLOTL24 subtask 1](https://github.com/ltgoslo/axolotl24_shared_task/tree/main) solution by DeepChange team

This repository contains the code to reproduce the winning solution for [the first subtask of the AXOLOTL24 shared task on Explainable Semantic Change Modeling](https://github.com/ltgoslo/axolotl24_shared_task/tree/main). A detailed description of our approach is provided in our paper (link TBD):

[**Denis Kokosinskii, Mikhail Kuklin, and Nikolay Arefyev. 2024. Deep-change at AXOLOTL-24: Orchestrating WSD and WSI Models for Semantic Change Modeling.** _In Proceedings of the 5th Workshop on Computational Approaches to Historical Language Change, Bangkok. Association for Computational Linguistics._](https://aclanthology.org/2024.lchange-1.16/)

## Reproduction
**1. Prepare your environment**
```
# first install pytorch appropriate for your system (example for CUDA 11.8)
pip install pytorch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
# install the requirements
pip install -r requirements.txt
# OR you can use precise package versions to ensure reproduction
pip install -r requirements.lock
```
**2. Run the full reproduction script**

```
cd code
bash repro.sh
```
Note: you can use -c option to use cached results and -d option to download embeddings insted of generating them
**3. Results**
```
                    ARI_fi  ARI_ru  ARI_de  F1_fi  F1_ru  F1_de
AggloM               0.581   0.044   0.492  0.674  0.639  0.695
AggloM_FiEnRu        0.631   0.035   0.485  0.731  0.640  0.639
WSD_GR               0.589   0.041   0.386  0.692  0.721  0.694
WSD_GR_FiEnRu        0.645   0.048   0.521  0.753  0.750  0.745
WSI_agglomerative    0.209   0.259   0.316  0.055  0.152  0.042
cluster2sense        0.209   0.259   0.316  0.392  0.346  0.432
outlier2cluster_fi   0.646   0.047   0.480  0.753  0.747  0.745
outlier2cluster_ru   0.274   0.247   0.322  0.410  0.645  0.510
```
Note: the results for the Finnish dataset are slightly different from our results in the competition  