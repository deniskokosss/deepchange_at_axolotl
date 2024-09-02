'''
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import os
import re
import subprocess
import random

import torch
from tqdm.auto import tqdm
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
import seaborn as sns
import pandas as pd

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

pos_converter = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}

def generate_key(lemma, pos):
	if pos in pos_converter.keys():
		pos = pos_converter[pos]
	key = '{}+{}'.format(lemma, pos)
	return key

def load_pretrained_model(name):
    if name == 'roberta-base':
        model = RobertaModel.from_pretrained('roberta-base')
        hdim = 768
    elif name == 'roberta-large':
        model = RobertaModel.from_pretrained('roberta-large')
        hdim = 1024
    elif name == 'bert-large':
        model = BertModel.from_pretrained('bert-large-uncased')
        hdim = 1024
    else: #bert base
        model = BertModel.from_pretrained('bert-base-uncased')
        hdim = 768
    return model, hdim

def load_tokenizer(name):
	if name == 'roberta-base':
		tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
	elif name == 'roberta-large':
		tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
	elif name == 'bert-large':
		tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
	else: #bert base
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	return tokenizer

def load_wn_senses(path):
	wn_senses = {}
	with open(path, 'r', encoding="utf8") as f:
		for line in f:
			line = line.strip().split('\t')
			lemma = line[0]
			pos = line[1]
			senses = line[2:]

			key = generate_key(lemma, pos)
			wn_senses[key] = senses
	return wn_senses

def get_label_space(data):
	#get set of labels from dataset
	labels = set()
	
	for sent in data:
		for _, _, _, _, label in sent:
			if label != -1:
				labels.add(label)

	labels = list(labels)
	labels.sort()
	labels.append('n/a')

	label_map = {}
	for sent in data:
		for _, lemma, pos, _, label in sent:
			if label != -1:
				key = generate_key(lemma, pos)
				label_idx = labels.index(label)
				if key not in label_map: label_map[key] = set()
				label_map[key].add(label_idx)

	return labels, label_map

def process_encoder_outputs(output, mask, as_tensor=False):
	combined_outputs = []
	position = -1
	avg_arr = []
	for idx, rep in zip(mask, torch.split(output, 1, dim=0)):
		#ignore unlabeled words
		if idx == -1: continue
		#average representations for units in same example
		elif position < idx: 
			position=idx
			if len(avg_arr) > 0: combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
			avg_arr = [rep]
		else:
			assert position == idx 
			avg_arr.append(rep)
	#get last example from avg_arr
	if len(avg_arr) > 0: combined_outputs.append(torch.mean(torch.stack(avg_arr, dim=-1), dim=-1))
	if as_tensor: return torch.cat(combined_outputs, dim=0)
	else: return combined_outputs

#run WSD Evaluation Framework scorer within python
def evaluate_output(scorer_path, gold_filepath, out_filepath):
	eval_cmd = ['java','-cp', scorer_path, 'Scorer', gold_filepath, out_filepath]
	output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE ).communicate()[0]
	output = [x.decode("utf-8") for x in output.splitlines()]
	p,r,f1 =  [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]
	return p, r, f1

def load_data(datapath, name):
	text_path = os.path.join(datapath, '{}.data.xml'.format(name))
	gold_path = os.path.join(datapath, '{}.gold.key.txt'.format(name))

	#load gold labels 
	gold_labels = {}
	with open(gold_path, 'r', encoding="utf8") as f:
		for line in f:
			line = line.strip().split(' ')
			instance = line[0]
			#this means we are ignoring other senses if labeled with more than one 
			#(happens at least in SemCor data)
			key = line[1]
			gold_labels[instance] = key

	#load train examples + annotate sense instances with gold labels
	sentences = []
	s = []
	with open(text_path, 'r', encoding="utf8") as f:
		for line in f:
			line = line.strip()
			if line == '</sentence>':
				sentences.append(s)
				s=[]
				
			elif line.startswith('<instance') or line.startswith('<wf'):
				word = re.search('>(.+?)<', line).group(1)
				lemma = re.search('lemma="(.+?)"', line).group(1) 
				pos =  re.search('pos="(.+?)"', line).group(1)

				#clean up data
				word = re.sub('&apos;', '\'', word)
				lemma = re.sub('&apos;', '\'', lemma)

				sense_inst = -1
				sense_label = -1
				if line.startswith('<instance'):
					sense_inst = re.search('instance id="(.+?)"', line).group(1)
					#annotate sense instance with gold label
					sense_label = gold_labels[sense_inst]
				s.append((word, lemma, pos, sense_inst, sense_label))

	return sentences

#normalize ids list, masks to whatever the passed in length is
def normalize_length(ids, attn_mask, o_mask, max_len, pad_id):
	if max_len == -1:
		return ids, attn_mask, o_mask
	else:
		if len(ids) < max_len:
			while len(ids) < max_len:
				ids.append(torch.tensor([[pad_id]]))
				attn_mask.append(0)
				o_mask.append(-1)
		else:
			ids = ids[:max_len-1]+[ids[-1]]
			attn_mask = attn_mask[:max_len]
			o_mask = o_mask[:max_len]

		assert len(ids) == max_len
		assert len(attn_mask) == max_len
		assert len(o_mask) == max_len

		return ids, attn_mask, o_mask

#filters down training dataset to (up to) k examples per sense 
#for few-shot learning of the model
def filter_k_examples(data, k):
	#shuffle data so we don't only get examples for (common) senses from beginning
	random.shuffle(data)
	#track number of times sense from data is used
	sense_dict = {}
	#store filtered data 
	filtered_data = []

	example_count = 0
	for sent in data:
		filtered_sent = []
		for form, lemma, pos, inst, sense in sent:
			#treat unlabeled words normally
			if sense == -1:
				x  = (form, lemma, pos, inst, sense)
			elif sense in sense_dict:
				if sense_dict[sense] < k: 
					#increment sense count and add example to filtered data
					sense_dict[sense] += 1
					x = (form, lemma, pos, inst, sense)
					example_count += 1
				else: #if the data already has k examples of this sense
					#add example with no instance or sense label to data
					x = (form, lemma, pos, -1, -1)
			else:
				#add labeled example to filtered data and sense dict
				sense_dict[sense] = 1
				x = (form, lemma, pos, inst, sense)
				example_count += 1
			filtered_sent.append(x)
		filtered_data.append(filtered_sent)

	print("k={}, training on {} sense examples...".format(k, example_count))

	return filtered_data

def preprocess(tokenizer, text_data, label_space, label_map, bsz=1, max_len=-1):
	if max_len == -1: 
		assert bsz==1 #otherwise need max_len for padding

	input_ids = []
	input_masks = []
	bert_masks = []
	output_masks = []
	instances = []
	label_indexes = []

	#tensorize data
	for sent in tqdm(text_data):
		sent_ids = [torch.tensor([tokenizer.encode(tokenizer.cls_token)])] #cls token aka sos token, returns a list with index
		b_masks = [-1]
		o_masks = []
		sent_insts = []
		sent_labels = []

		ex_count = 0 #DEBUGGING
		for idx, (word, lemma, pos, inst, label) in enumerate(sent):
			word_ids = [torch.tensor([[x]]) for x in tokenizer.encode(word.lower())]
			sent_ids.extend(word_ids)

			if inst != -1:
				ex_count += 1 #DEBUGGING
				b_masks.extend([idx]*len(word_ids))

				sent_insts.append(inst)
				if label in label_space:
					sent_labels.append(torch.tensor([label_space.index(label)]))
				else:
					sent_labels.append(torch.tensor([label_space.index('n/a')]))

				#adding appropriate label space for sense-labeled word (we only use this for wsd task)
				key = generate_key(lemma, pos)
				if key in label_map:
					l_space = label_map[key]
					mask = torch.zeros(len(label_space))
					for l in l_space: mask[l] = 1
					o_masks.append(mask)
				else:
					o_masks.append(torch.ones(len(label_space))) #let this predict whatever -- should not use this (default to backoff for unseen forms)

			else:
				b_masks.extend([-1]*len(word_ids))

			#break if we reach max len so we don't keep overflowing examples
			if max_len != -1 and len(sent_ids) >= (max_len-1):
				break

		#pad inputs + add eos token
		sent_ids.append(torch.tensor([tokenizer.encode(tokenizer.sep_token)])) #aka eos token
		input_mask = [1]*len(sent_ids)
		b_masks.append(-1)
		sent_ids, input_mask, b_masks = normalize_length(sent_ids, input_mask, b_masks, max_len, pad_id=tokenizer.encode(tokenizer.pad_token)[0])

		#not including examples sentences with no annotated sense data	
		if len(sent_insts) > 0:
			input_ids.append(torch.cat(sent_ids, dim=-1))
			input_masks.append(torch.tensor(input_mask).unsqueeze(dim=0))
			bert_masks.append(torch.tensor(b_masks).unsqueeze(dim=0))
			output_masks.append(torch.stack(o_masks, dim=0))
			instances.append(sent_insts)
			label_indexes.append(torch.cat(sent_labels, dim=0))

	#batch data now that we pad it
	data = list(zip(input_ids, input_masks, bert_masks, output_masks, instances, label_indexes))
	if bsz > 1:
		print('Batching data with bsz={}...'.format(bsz))
		batched_data = []
		for idx in range(0, len(data), bsz):
			if idx+bsz <=len(data): b = data[idx:idx+bsz]
			else: b = data[idx:]
			input_ids = torch.cat([x for x,_,_,_,_,_ in b], dim=0)
			input_mask = torch.cat([x for _,x,_,_,_,_ in b], dim=0)
			bert_mask = torch.cat([x for _,_,x,_,_,_ in b], dim=0)
			output_mask = torch.cat([x for _,_,_,x,_,_ in b], dim=0)
			instances = []
			for _,_,_,_,x,_ in b: instances.extend(x)
			labels = torch.cat([x for _,_,_,_,_,x in b], dim=0)
			batched_data.append((input_ids, input_mask, bert_mask, output_mask, instances, labels))
		return batched_data
	else:  return data


# creates a sense label/ gloss dictionary for training/using the gloss encoder
def load_and_preprocess_glosses(data, tokenizer, wn_senses, max_len=-1):
    sense_glosses = {}
    sense_weights = {}

    gloss_lengths = []

    for sent in data:
        for _, lemma, pos, _, label in sent:
            if label == -1:
                continue  # ignore unlabeled words
            else:
                key = generate_key(lemma, pos)
                if key not in sense_glosses:
                    # get all sensekeys for the lemma/pos pair
                    sensekey_arr = wn_senses[key]
                    # get glosses for all candidate senses
                    gloss_arr = [wn.lemma_from_key(s).synset().definition() for s in sensekey_arr]

                    # preprocess glosses into tensors
                    # gloss_ids, gloss_masks = tokenize_glosses(gloss_arr, tokenizer, max_len)
                    # gloss_ids = torch.cat(gloss_ids, dim=0)
                    # gloss_masks = torch.stack(gloss_masks, dim=0)
                    sense_glosses[key] = (gloss_arr, sensekey_arr)

                    # intialize weights for balancing senses
                #     sense_weights[key] = [0] * len(gloss_arr)
                #     w_idx = sensekey_arr.index(label)
                #     sense_weights[key][w_idx] += 1
                # else:
                #     # update sense weight counts
                #     w_idx = sense_glosses[key][2].index(label)
                #     sense_weights[key][w_idx] += 1

                # make sure that gold label is retrieved synset
                # assert label in sense_glosses[key][2]

    # normalize weights
    for key in sense_weights:
        total_w = sum(sense_weights[key])
        sense_weights[key] = torch.FloatTensor([total_w / x if x != 0 else 0 for x in sense_weights[key]])

    return sense_glosses, sense_weights


if __name__ == '__main__':
    train_path = "WSD_Evaluation_Framework/Training_Corpora/SemCor"
    train_data = load_data(train_path, 'semcor')

    wn_path = os.path.join(train_path, '../../Data_Validation/candidatesWN30.txt')
    wn_senses = load_wn_senses(wn_path)
    train_gloss_dict, train_gloss_weights = load_and_preprocess_glosses(
        train_data, None, wn_senses, max_len=-1
    )
    label_space, label_map = get_label_space(train_data)
    print('num labels = {} + 1 unknown label'.format(len(label_space)-1))

    usage_id = 1000
    semcor = []
    for sent in train_data:
        print(sent)
        for wordform, lemma, pos, _, label in sent:
            res = {}
            if isinstance(label, str):
                usage_id += 1
                res['usage_id'] = str(usage_id)
                res['word'] = lemma
                res['orth'] = wordform
                res['sense_id'] = label
                senses = train_gloss_dict[generate_key(lemma,  pos)]
                res['gloss'] = senses[0][senses[1].index(label)]
                res['example'] = ' '.join([t for t,_,_,_,_ in sent])
                pos = res['example'].index(wordform), res['example'].index(wordform) + len(wordform)
                res['indices_target_token'] = '{}:{}'.format(pos[0], pos[1])
                assert res['example'][pos[0]:pos[1]] == wordform
                res['date'] = '2024'
                res['period'] = 'old'
                res['lang'] = 'en'
                res['part'] = 'train'
                semcor.append(res) 

    df = pd.DataFrame(semcor)
    df.to_csv('../data/english/semcor.train.en.tsv', index=False, sep='\t')