# [AXOLOTL24 subtask 1](https://github.com/ltgoslo/axolotl24_shared_task/tree/main) solution by DeepChange team

This repository contains the code to reproduce the winning solution for [the first subtask of the AXOLOTL24 shared task on Explainable Semantic Change Modeling](https://github.com/ltgoslo/axolotl24_shared_task/tree/main). A detailed description of our approach is provided in our paper:

[**Denis Kokosinskii, Mikhail Kuklin, and Nikolay Arefyev. 2024. Deep-change at AXOLOTL-24: Orchestrating WSD and WSI Models for Semantic Change Modeling.** _In Proceedings of the 5th Workshop on Computational Approaches to Historical Language Change, Bangkok. Association for Computational Linguistics._](https://aclanthology.org/2024.lchange-1.16/)

## Reproduction
**1. Prepare your environment**

If you want to exactly reproduce our environment, use Python 3.9.19.

(Optional) create a new environment. E.g. if you use conda:
```
conda create -n deepchange_axolotl python=3.9.19
conda activate deepchange_axolotl
```
Install dependencies:
```
# first install pytorch appropriate for your system (example for CUDA 11.8)
pip3 install torch==2.2.0 torchaudio==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
# install the requirements
pip install -r requirements.txt
# OR use precise package versions to ensure reproduction
pip install -r requirements.lock
```
**2. Run the full reproduction script**

```
cd code
bash repro.sh
```
Note: you can use -c option to use cached results and -d option to download embeddings insted of generating them.

**3. Results**
```                            
                              ARI_fi  ARI_ru  ARI_de  F1_fi  F1_ru  F1_de
AggloM                        0.581   0.044   0.492  0.674  0.639  0.695
AggloM_FiEnRu                 0.631   0.035   0.485  0.731  0.640  0.639
WSD_GR                        0.589   0.041   0.386  0.692  0.721  0.694
WSD_GR_FiEnRu                 0.645   0.048   0.521  0.753  0.750  0.745
WSD_GR_FiSG                   0.638   0.059   0.543  0.752  0.729  0.758
WSD_GR_Ru                     0.572   0.043   0.446  0.679  0.748  0.705
WSI_agglomerative             0.209   0.259   0.316  0.055  0.152  0.042
cluster2sense                 0.209   0.259   0.316  0.392  0.346  0.432
outlier2cluster_fi            0.646   0.047   0.480  0.753  0.747  0.745
outlier2cluster_ru            0.274   0.247   0.322  0.410  0.645  0.510
```

Note: the results for the Finnish dataset are slightly different from our results in the competition.

<details><summary>Exact reproduction of the results from the paper: run the reproduction with the precomputed embeddings</summary>

In the competition we used an intermetiadate checkpoint of the GR model for the Finnish dataset and the Russian datasets, which were later lost. However, the difference is minor.

To get the exact results from the paper run

```
bash repro.sh -d
```
Which results in:
```
                    ARI_fi  ARI_ru  ARI_de  F1_fi  F1_ru  F1_de
AggloM               0.581   0.044   0.492  0.674  0.643  0.695
AggloM_FiEnRu        0.631   0.035   0.485  0.731  0.636  0.639
WSD_GR               0.581*  0.041   0.386  0.690* 0.721  0.694
WSD_GR_FiEnRu        0.649*  0.048   0.521  0.756* 0.750  0.745
WSD_GR_FiSG          0.638   0.059   0.543  0.752  0.729  0.758
WSD_GR_Ru            0.568*  0.053*  0.464* 0.568* 0.750* 0.724*
WSI_agglomerative    0.209   0.259   0.316  0.055  0.152  0.042
cluster2sense        0.209   0.259   0.316  0.392  0.346  0.432
outlier2cluster_fi   0.649*  0.047   0.480  0.756* 0.747  0.745
outlier2cluster_ru   0.278   0.247   0.322  0.414  0.645  0.510
```
\* – our results from the leaderboard, which can be reproduced exactly only with the pre-computed embeddings from the lost GR checkpoints
</details>

**4. Improvements after publication**

Some methods were improved by using a better WSI model. The results are the following:
```
                             ARI_fi  ARI_ru  ARI_de  F1_fi  F1_ru  F1_de
method                                                                  
WSI_agglomerative             0.209   0.259   0.316  0.055  0.152  0.042
WSI_agglomerative_improved    0.265   0.304   0.445  0.055  0.152  0.042
cluster2sense                 0.209   0.259   0.316  0.392  0.346  0.432
cluster2sense_improved        0.265   0.304   0.445  0.398  0.289  0.465
outlier2cluster_ru            0.274   0.247   0.322  0.410  0.645  0.510
outlier2cluster_ru_improved   0.276   0.254   0.326  0.410  0.645  0.510
```

To reproduce them run:

```
bash improvements.sh
```

**5. Retrain the models**
Refer to code/finetune_GR.sh

