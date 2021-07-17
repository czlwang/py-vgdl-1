import os
import sys
import numpy as np 
import random
import statistics
from statistics import mode
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.functional as F 

import transformers

from data_loader import *
from pred_model import *

if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'

lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

def decode_model(model_out):
	types = []
	for i in range(len(model_out)):
		prob_dist = model_out[i]
		t = torch.argmax(prob_dist)
		t = int(t.cpu())
		types.append(t)
	return mode(types)

def eval_model(test_data, modelname, base_path, num_labels, dropout = 0.25, lm = None, bert_layer = -1, hidden_size = 200):
	classifier = MLP(num_labels, dropout, lm, hidden_size)

	saved_model = os.path.join(base_path, 'rule_rep', 'checkpoints', modelname)
	classifier.load_state_dict(torch.load(saved_model))
	classifier = classifier.to(device)

	classifier.eval()

	total_examples = 0
	total_correct = 0

	print('Beginning evaluation')
	for rule_dict, labels in tqdm(test_data):
		input_ids = []
		attention_mask = []
		input_ids.append(torch.tensor(rule_dict['input_ids']).long().to(device))
		attention_mask.append(torch.tensor(rule_dict['attention_mask']).long().to(device))
		labels = torch.tensor(labels).long().to(device)

		input_ids = torch.stack(input_ids).to(device)
		attention_mask = torch.stack(attention_mask).to(device)

		outputs = classifier.forward(input_ids, attention_mask, bert_layer)

		pred = decode_model(outputs)
		if pred == labels[0]:
			total_correct = total_correct + 1
		total_examples += 1
	return ('Output Score: {}').format(total_correct/total_examples)

def test_eval(base_path, modelname, games, num_labels, lm = None, dropout = 0.25, bert_layer = -1, hidden_size = 200):
	print('Beginning data loading from games {}'.format(games))
	game_path = os.path.join(base_path, 'vgdl', 'games')
	rule_dicts = load_vgdl(base_path, games)

	tokenized_rule_dict = {}
	if rule_type is not None:
		rules = []
		for rule_dict in rule_dicts:
			rules = rules + rule_dict[rule_type]
		tokenized_rules = tokenize_rules(rules)
	else:
		for rule_dict in rule_dicts:
			for rule_type in rule_dict:
				try:
					tokenized_rule_dict[rule_type] += tokenize_rules(rule_dict[rule_type])
				except KeyError:
					tokenized_rule_dict[rule_type] = tokenize_rules(rule_dict[rule_type])

	if tokenized_rule_dict:
		rule_dict = tokenized_rule_dict
		dataset = build_dataset(rule_dict)
	random.shuffle(dataset)
	print('Data loading complete and dataset built')

	print(eval_model(dataset, modelname, base_path, num_labels, dropout, lm, bert_layer, hidden_size))

# test_eval('/storage/vsub851/py-vgdl-1', 'mlp1.pt', ['zelda'], 4, lm = lm_pretrained, dropout = 0.25, bert_layer = -1)