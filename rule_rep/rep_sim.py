import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch import cuda
from sklearn.decomposition import PCA
import pandas as pd 

from data_loader import *

device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

bert = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)
bert.eval()

base_path = '/storage/vsub851/py-vgdl-1/vgdl/games'
games = ['aliens']
rule_dicts = load_vgdl(base_path, games)

tokenized_rule_dict = {}
for rule_dict in rule_dicts:
	for rule_type in rule_dict:
		try:
			tokenized_rule_dict[rule_type] += tokenize_rules(rule_dict[rule_type])
		except KeyError:
			tokenized_rule_dict[rule_type] = tokenize_rules(rule_dict[rule_type])

bert_layer = -1
rule_embed_dict = {}
for key in tokenized_rule_dict:
	rule_embed_dict[key] = []
	for rules in tokenized_rule_dict[key]:
		input_ids = torch.stack([torch.tensor(rules['input_ids']).long()]).to(device)
		attention_mask = torch.stack([torch.tensor(rules['attention_mask']).long()]).to(device)

		with torch.no_grad():
			bert_output = bert(input_ids, attention_mask, output_hidden_states = True)
			hidden_states = bert_output.hidden_states[bert_layer]
		rule_embed_dict[key].append((rules['rule'], hidden_states.cpu().numpy()))

rule_set1 = 'InteractionSet'
rule_set2 = 'SpriteSet'
rules1 = rule_embed_dict[rule_set1]
rules2 = rule_embed_dict[rule_set2]

pca = PCA(n_components = 2)
cls_reps1 = [] 
for rule, rule_rep in rules1:
	cls_rep = rule_rep[:, 0, :]
	# print(np.shape(cls_rep))
	cls_rep = np.squeeze(cls_rep)
	cls_reps1.append(cls_rep)

cls_reps1 = np.array(cls_reps1)
# print(np.shape(cls_reps1))

principalComponents1 = pca.fit_transform(cls_reps1)
principalDf1 = pd.DataFrame(data = principalComponents1, columns = ['principal component 1', 'principal component 2'])
plot = principalDf1.plot.scatter(x = 'principal component 1', y = 'principal component 2', title = 'PCA of InteractionSet')
fig = plot.get_figure()
fig.savefig('interaction_set.png')

cls_reps2 = [] 
for rule, rule_rep in rules2:
	cls_rep = rule_rep[:, 0, :]
	# print(np.shape(cls_rep))
	cls_rep = np.squeeze(cls_rep)
	cls_reps2.append(cls_rep)

cls_reps2 = np.array(cls_reps2)
# print(np.shape(cls_reps1))

principalComponents2 = pca.fit_transform(cls_reps2)
principalDf2 = pd.DataFrame(data = principalComponents2, columns = ['principal component 1', 'principal component 2'])
plot = principalDf2.plot.scatter(x = 'principal component 1', y = 'principal component 2', title = 'PCA of SpriteSet')
fig = plot.get_figure()
fig.savefig('sprite_set.png')
