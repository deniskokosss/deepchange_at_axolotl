# [AXOLOTL24 subtask 1](https://github.com/ltgoslo/axolotl24_shared_task/tree/main) solution by DeepChange team

This repository contains the code to reproduce the winning solution for [the first subtask of the AXOLOTL24 shared task on Explainable Semantic Change Modeling](https://github.com/ltgoslo/axolotl24_shared_task/tree/main). A detailed description of our approach is provided in our paper (link TBD):

[**Denis Kokosinskii, Mikhail Kuklin, and Nikolay Arefyev. 2024. Deep-change at AXOLOTL-24: Orchestrating WSD and WSI Models for Semantic Change Modeling.** _In Proceedings of the 5th Workshop on Computational Approaches to Historical Language Change, Bangkok. Association for Computational Linguistics._](https://aclanthology.org/2024.lchange-1.16/)

## Reproduction
**1. Prepare your environment**
```
# first install pytorch appropriate for your system (example for CUDA 11.8)
pip install pytorch --index-url https://download.pytorch.org/whl/cu118
# install the requirements
pip install -r requirements.txt
```
**2. Run the full reproduction script**

```
cd code
bash repro.sh
```
**3. Results**
```
                    ARI_fi  ARI_ru  ARI_de  F1_fi  F1_ru  F1_de
WSD_GR_FiEnRu        0.649   0.048   0.521  0.756  0.750  0.745
WSD_GR               0.581   0.041   0.386  0.690  0.721  0.694
outlier2cluster_fi   0.649   0.047   0.480  0.756  0.747  0.745
outlier2cluster_ru   0.278   0.247   0.322  0.414  0.645  0.510
WSI_agglomerative    0.209   0.259   0.316  0.055  0.152  0.042
AggloM_FiEnRu        0.631   0.037   0.485  0.731  0.636  0.639
AggloM               0.581   0.026   0.492  0.674  0.643  0.695
cluster2sense        0.209   0.259   0.316  0.392  0.346  0.432
```
