from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import typing as tp
from tqdm import tqdm
import json
torch.set_grad_enabled(False)

from safetensors.torch import load_model
import numpy as np
import argparse


class GlossReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        self.context_encoder = AutoModel.from_pretrained(
            "xlm-roberta-large", device_map='auto'
        )
        self.gloss_encoder = AutoModel.from_pretrained(
            "xlm-roberta-large", device_map='auto'
        )
        self.ood_prob = torch.nn.Parameter(torch.Tensor([500.0]))

    def vectorize_word_in_context(self, context: str, word_pos: tp.Tuple[int, int]):
        left_ctx_len = 1 + len(self.tokenizer(context[:word_pos[0]], add_special_tokens=False)['input_ids'])
        tgt_len = len(self.tokenizer(context[word_pos[0]:word_pos[1]], add_special_tokens=False)['input_ids'])
        input_tokens = self.tokenizer.encode(context, return_tensors='pt').to(self.context_encoder.embeddings.word_embeddings.weight.device)
        # print(input_tokens.shape)
        if len(input_tokens[0]) >= 512:
            new_start = max(0, left_ctx_len - 250)
            new_end = min(len(input_tokens[0]), left_ctx_len + 250)
            left_ctx_len -= new_start
            input_tokens = input_tokens[:, new_start:new_end]
        outputs = self.context_encoder(input_tokens)
    
        vector = outputs['last_hidden_state'][0, left_ctx_len:left_ctx_len + tgt_len, :].mean(axis=0)
        return vector
    
    def vectorize_glosses(self, glosses: tp.List[str]):
        input_tokens = self.tokenizer.batch_encode_plus(glosses, return_tensors='pt', padding=True).to(self.context_encoder.embeddings.word_embeddings.weight.device)
        outputs = self.gloss_encoder(**input_tokens)
        vector = outputs['last_hidden_state'][:, 0, :]
        return vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='../data/models/GR_FiEnRu/model.safetensors')
    parser.add_argument('--datasets', type=str, help='comma-separated', 
                        default=
                        "../axolotl24_shared_task/data/finnish/axolotl.dev.fi.tsv,"
                        "../axolotl24_shared_task/data/finnish/axolotl.test.fi.gold.tsv,"
                        "../axolotl24_shared_task/data/german/axolotl.test.surprise.gold.tsv,"
                        "../data/add_index/axolotl.dev.ru.tsv,"
                        "../data/add_index/axolotl.test.ru.gold.tsv"
                        )
    parser.add_argument('--out_file', type=str, default="../data/embeddings/GR_FiEnRu.json")
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    print(args)
    
    glossreader = GlossReader()
    if args.model.endswith('.safetensors'):
        load_model(glossreader, args.model, strict=False)
    elif args.model.endswith('GR/model.pt'):
        cpt = torch.load(args.model, map_location='cpu')
        state_dict = {
            k.replace("gloss_encoder.gloss_encoder.", 'gloss_encoder.'): v 
            for k,v in cpt.items() if "gloss_encoder" in k}
        state_dict |= {
            k.replace("context_encoder.context_encoder.", 'context_encoder.'): v 
            for k,v in cpt.items() if "context_encoder" in k
        }
        state_dict['ood_prob'] = torch.Tensor([0.0])
        glossreader.load_state_dict(state_dict)

    parts = []
    for dataset in args.datasets.split(','):
        print("Loaded: " + dataset)
        df = (pd.read_csv(dataset, sep='\t'))
        df.loc[df['sense_id'].isna(), 'sense_id'] = 'unk'
        df.loc[df.example.isna(), 'example'] = 'unk'
        df['indices_target_token'] = df['indices_target_token'].str.replace('-', ':')
        df.loc[df['indices_target_token'].isna(), 'indices_target_token'] = df.loc[df['indices_target_token'].isna(), 'example'].apply(lambda x: f"0:{len(x)}")
        parts.append(df)
    
    df = pd.concat(parts)
    assert df['usage_id'].nunique() == df.shape[0]

    cache = {'contexts': {}, "glosses": {}}
    if not args.no_cuda and torch.cuda.is_available():
        glossreader = glossreader.to('cuda:0')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        assert row['usage_id'] not in cache['contexts'].keys()
        target_pos = list(map(int, row['indices_target_token'].split(';')[0].split(":")[:2]))
        if row['example'] != 'unk':
            cache['contexts'][row['usage_id']] = glossreader.vectorize_word_in_context(row['example'], target_pos).tolist()
        if row['sense_id'] != 'unk' and row['sense_id'] not in cache['glosses'].keys():
            cache['glosses'][row['sense_id']] = glossreader.vectorize_glosses((row['gloss'], ))[0].tolist()

    with open(args.out_file, "w") as f:
        json.dump(cache, f)