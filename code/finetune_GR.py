from transformers import AutoModel, AutoTokenizer, PreTrainedModel
import torch
import os
import pandas as pd
import typing as tp
from tqdm import tqdm
import json
import numpy as np

from transformers import EvalPrediction, Trainer, TrainingArguments
from datasets import Dataset
from argparse import ArgumentParser

from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score

class GlossReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        self.context_encoder = AutoModel.from_pretrained(
            "xlm-roberta-large", 
        )
        self.gloss_encoder = AutoModel.from_pretrained(
            "xlm-roberta-large",
        )
        self.ood_prob = torch.nn.Parameter(torch.Tensor([500.0]))

    def vectorize_word_in_context(self, context: str, word_pos: tp.Tuple[int, int]):
        left_ctx_len = 1 + len(self.tokenizer(context[:word_pos[0]], add_special_tokens=False)['input_ids'])
        tgt_len = len(self.tokenizer(context[word_pos[0]:word_pos[1]], add_special_tokens=False)['input_ids'])
        input_tokens = self.tokenizer.encode(context, return_tensors='pt').to(self.context_encoder.embeddings.word_embeddings.weight.device)
        outputs = self.context_encoder(input_tokens)
    
        vector = outputs['last_hidden_state'][0, left_ctx_len:left_ctx_len + tgt_len, :].mean(axis=0)
        return vector
    
    def vectorize_glosses(self, glosses: tp.List[str]):
        input_tokens = self.tokenizer.batch_encode_plus(glosses, return_tensors='pt', padding=True).to(self.context_encoder.embeddings.word_embeddings.weight.device)
        outputs = self.gloss_encoder(**input_tokens)
        vector = outputs['last_hidden_state'][:, 0, :]
        return vector
    
    def prepare_inputs(self, context: str, word_pos: tp.Tuple[int, int], glosses: tp.List[str], labels=None):
        left_ctx_len = 1 + len(self.tokenizer(context[:word_pos[0]], add_special_tokens=False)['input_ids'])
        tgt_len = len(self.tokenizer(context[word_pos[0]:word_pos[1]], add_special_tokens=False)['input_ids'])
        input_tokens = self.tokenizer.encode(context, return_tensors='pt')[0]
        input_pos_tokens = torch.as_tensor([left_ctx_len, left_ctx_len + tgt_len])
        
        gloss_tokens = self.tokenizer.batch_encode_plus(glosses, return_tensors='pt', padding=True)
        
        return {
            'input_ids': input_tokens,
            'input_pos_ids': input_pos_tokens,
            'gloss_input_ids': gloss_tokens['input_ids'],
            'gloss_attention_mask': gloss_tokens['attention_mask'],
            'labels': labels
        }
    
    def forward(self, input_ids, attention_mask, input_pos_ids, gloss_input_ids, gloss_attention_mask, labels=None):
        outputs = self.context_encoder(input_ids, attention_mask=attention_mask)
        word_vectors = []
        for (start, end), outs in zip(input_pos_ids, outputs['last_hidden_state']):
            word_vectors.append(outs[start:end, :].mean(axis=0))
        word_vectors = torch.stack(word_vectors, dim=0) # bs, hidden
        
        gloss_bs = gloss_input_ids.shape[0] * gloss_input_ids.shape[1]
        outputs = self.gloss_encoder(input_ids=gloss_input_ids.view(gloss_bs, -1), attention_mask=gloss_attention_mask.view(gloss_bs, -1)) 
        gloss_vectors = outputs['last_hidden_state'][:, 0, :].view(gloss_input_ids.shape[0], gloss_input_ids.shape[1], -1) # bs, num_glosses, hidden

        # print(f"{gloss_vectors.shape=}")
        logits = torch.matmul(gloss_vectors, word_vectors.unsqueeze(-1)).squeeze(-1)
        # similarities = torch.softmax(torch.matmul(gloss_vectors, word_vectors.unsqueeze(-1)).squeeze(-1), dim=1)
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            
            # print(f"{similarities=}, {labels=}, {loss=}")
        outputs = {
            'logits': logits,
            'loss': loss,
            # 'accuracy': float(similarities.argmax(dim=0) == np.argmax(labels))
        }
        return outputs


# sense of the word is unknown -> sanan merkitystä ei tunneta
def one_hot(label, classes):
    vec = [0.0] * len(classes)
    vec[classes.index(label)] = 1.0
    return vec


def df2ds(df, special_gloss=False):
    ds = []
    for word, df_word in df.groupby('word'):
        sense_ids_old = df_word.loc[df_word['period'] == 'old', 'sense_id'].unique()
        sense_ids_gained = np.setdiff1d(df_word.loc[df_word['period'] == 'new', 'sense_id'].unique(), sense_ids_old)
        if special_gloss:
            gloss_map = {k: df_word.loc[df_word.sense_id == k, 'gloss'].iloc[0] for k in sense_ids_old} | {k: "sanan merkitystä ei tunneta" for k in sense_ids_gained}
        else:
            gloss_map = {k: df_word.loc[df_word.sense_id == k, 'gloss'].iloc[0] for k in list(sense_ids_old) + list(sense_ids_gained)}
        glosses = list(set(gloss_map.values()))
        if len(gloss_map) > 1:
            for _, s in df_word.iterrows():
                sample = {
                    "context": s['example'],
                    "word_pos": list(map(int, s['indices_target_token'].split(';')[0].split(":")[:2])),
                    "glosses": glosses,
                    "labels": one_hot(gloss_map[s['sense_id']], glosses)
                }
                assert sample['word_pos'][0] < sample['word_pos'][1]
                assert (sample['glosses'][np.argmax(sample['labels'])] == "sanan merkitystä ei tunneta") or sample['glosses'][np.argmax(sample['labels'])] == s['gloss']
                ds.append(glossreader.prepare_inputs(**sample))
    return ds


def log_similarities(pred: EvalPrediction):
    logits = np.where(pred.predictions == -100, np.nan, pred.predictions)
    labels = np.where(pred.label_ids == -100, np.nan, pred.label_ids)
    predicted_classes = np.nanargmax(logits, -1)
    labels_classes = np.nanargmax(labels, -1)
    # print(pd.Series(predicted_classes).value_counts())
    # print(pd.Series(labels_classes).value_counts())
    # print(logits.shape)
    
    torch.cuda.empty_cache()
    return {
        "accuracy": accuracy_score(predicted_classes, labels_classes),
        "f1_score": f1_score(predicted_classes, labels_classes, average='micro'),
        "logits_mean": np.nanmean(logits),
        "logits_min": np.nanmin(logits)
    }
    

def collate_fn(batch):
    
    # input_ids bs, seq_len
    # input_pos_ids bs, 2
    # gloss_input_ids bs, num_glosses, seq_len
    # gloss_attention_mask bs, num_glosses, seq_len
    # labels bs, num_glosses
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    attention_mask = [torch.ones_like(t) for t in input_ids]
    input_ids = pad_sequence(input_ids, padding_value=1).T
    attention_mask = pad_sequence(attention_mask, padding_value=0).T
    
    input_pos_ids = torch.as_tensor([b['input_pos_ids'] for b in batch])

    bs = len(batch)
    max_num_glosses = max([len(b['gloss_input_ids']) for b in batch])
    max_length = max([len(t) for b in batch for t in b['gloss_input_ids']])
    template = torch.ones(max_num_glosses, max_length, dtype=torch.int64)
    gloss_input_ids = []
    for b in batch:
        templ_ = template.detach().clone()
        templ_[:len(b['gloss_input_ids']), :len(b['gloss_input_ids'][0])] = torch.tensor(b['gloss_input_ids'])
        gloss_input_ids.append(templ_)
    gloss_input_ids = torch.stack(gloss_input_ids)
    gloss_input_ids[:, :, 0] = 0

    template = torch.zeros(max_num_glosses, max_length, dtype=torch.int64)
    gloss_attention_mask = []
    for b in batch:
        templ_ = template.detach().clone()
        templ_[:len(b['gloss_attention_mask']), :len(b['gloss_attention_mask'][0])] = torch.tensor(b['gloss_attention_mask'])
        gloss_attention_mask.append(templ_)
    gloss_attention_mask = torch.stack(gloss_attention_mask)
    gloss_attention_mask[:, :, 0] = 1
   
    labels = [torch.tensor(b['labels'], dtype=torch.float32) for b in batch]
    labels = pad_sequence(labels, padding_value=0.0).T

    collated = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'input_pos_ids': input_pos_ids,
        'gloss_input_ids': gloss_input_ids,
        'gloss_attention_mask': gloss_attention_mask,
        'labels': labels,
    }
    return collated


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datasets', type=str, required=True, help='fienru , fi or ru', default='fienru')
    parser.add_argument('--sg', action='store_true', default=False, help='Use Special Gloss instead of gained senses')
    parser.add_argument('--batch', type=int, default=8, help='per_device_train_batch_size')
    parser.add_argument('--ga', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--run_name', type=str, default="glossreader_retrain0", help='Random seed')

    args = parser.parse_args()

    train_sets = []
    valid_sets = {}
    np.random.seed(args.seed)
    if 'ru' in args.datasets.lower():
        df = pd.read_csv("../data/add_index/axolotl.train.ru.tsv", sep='\t')
        df['indices_target_token'] = df['indices_target_token'].str.replace('-', ':')
        df = df[df['indices_target_token'].notna()]
        valid_words = np.random.choice(df['word'].unique(), int(df['word'].nunique() * 0.1), replace=False)
        df_valid = df[df['word'].isin(valid_words)]
        df_train = df[~df['word'].isin(valid_words)]
        train_sets.append(df_train)
        valid_sets['ru'] = df_valid
    if 'en' in args.datasets.lower():
        df_sc = pd.read_csv("../data/english/semcor.train.en.tsv", sep='\t')
        train_sets.append(df_sc)
    if 'fi' in args.datasets.lower():
        df = pd.read_csv("../axolotl24_shared_task/data/finnish/axolotl.train.fi.tsv", sep='\t')
        valid_words = np.random.choice(df['word'].unique(), int(df['word'].nunique() * 0.1), replace=False)
        df_valid = df[df['word'].isin(valid_words)]
        df_train = df[~df['word'].isin(valid_words)]
        train_sets.append(df_train)
        valid_sets['fi'] = df_valid


    glossreader = GlossReader()
    cpt = torch.load("../data/models/GR/model.pt", map_location='cpu')
    state_dict = {
        k.replace("gloss_encoder.gloss_encoder.", 'gloss_encoder.'): v 
        for k,v in cpt.items() if "gloss_encoder" in k}
    state_dict |= {
        k.replace("context_encoder.context_encoder.", 'context_encoder.'): v 
        for k,v in cpt.items() if "context_encoder" in k
    }
    state_dict['ood_prob'] = torch.Tensor([500.0])
    glossreader.load_state_dict(state_dict)
    train_dataset = Dataset.from_list(df2ds(pd.concat(train_sets), args.sg))
    valid_datasets = {k: Dataset.from_list(df2ds(v)) for k,v in valid_sets.items()}

    
    trainer = Trainer(model=glossreader, 
                  train_dataset=train_dataset, eval_dataset=valid_datasets,
                  data_collator=collate_fn,
                  compute_metrics=log_similarities,
                  args=TrainingArguments(
                        output_dir=f'../data/models/{args.run_name}_{args.datasets}' + ('_SG' if args.sg else ''),
                        eval_delay=0,
                        report_to='tensorboard',
                        learning_rate=3e-05,
                        per_device_train_batch_size=args.batch,
                        gradient_accumulation_steps=args.ga,
                        warmup_ratio=0.05,
                        logging_steps=10,
                        logging_first_step=True,
                        save_strategy='steps',
                        save_steps=100,
                        evaluation_strategy='steps',
                        eval_steps=100,
                        seed=args.seed, data_seed=args.seed
                  ))
    print(f"World size: {trainer.accelerator.num_processes}")
    trainer.train()