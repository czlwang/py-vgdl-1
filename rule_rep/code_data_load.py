import os
import sys
import numpy as np 
import statistics

import fastBPE
import torch

import TransCoder.preprocessing.src.code_tokenizer as code_tokenizer
from TransCoder.XLM.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from TransCoder.XLM.src.model import build_model
from TransCoder.XLM.src.utils import AttrDict

base_path = '/storage/vsub851/py-vgdl-1/rule_rep'
games = ['aliens_py']

def data_load(base_path, games, sets = ['SpriteSet', 'InteractionSet', 'TerminationSet'], return_type = 'block'):
	rule_dict = {}
	for rule_type in sets:
		rule_dict[rule_type] = []
	rule_type = ''
	for game in games:
		game = game + '.txt'
		game_path = os.path.join(base_path, 'code_games', game)
		game_dict = {}
		if os.path.exists(game_path):
			f = open(game_path, 'r')
			file = f.readlines()
			for line in file:
				line = line.strip()
				if line in sets:
					rule_type = line
					game_dict[rule_type] = []
				try:
					if line not in sets:
						game_dict[rule_type].append(line)
				except KeyError:
					continue
			for rule_type in game_dict:
				rule_dict[rule_type].append(game_dict[rule_type])
		else:
			continue
	# print(rule_dict)
	new_rule_dict = {}
	if return_type == 'block':
		for rule_type in rule_dict:
			new_rule_dict[rule_type] = []
			for game in rule_dict[rule_type]:
				new_rule_dict[rule_type].append(' '.join(game))
	else:
		for rule_type in rule_dict:
			new_rule_dict[rule_type] = []
			for game in rule_dict[rule_type]:
				for rule in game:
					new_rule_dict[rule_type].append(rule)
	return new_rule_dict

def dataset(rule_dict, dataset_type = 'block'):
	label = 0
	dataset = []
	if dataset_type == 'block':
		for rule_type in rule_dict:
			for rules in rule_dict[rule_type]:
				dataset.append((rules, label))
			label = label + 1
	else:
		for rule_type in rule_dict:
			for rules in rule_dict[rule_type]:
				dataset.append(rules, label)
			label = label + 1
	return dataset

def encoder_load(base_path, model_name):
	model_path = os.path.join(base_path, model_name)
	reloaded = torch.load(model_path, map_location = 'cpu')
	reloaded['encoder'] = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in reloaded['encoder'].items()}
	reloaded_params = AttrDict(reloaded['params'])

	dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

	reloaded_params['reload_model'] = ','.join([model_path]*2)
	encoder, _ = build_model(reloaded_params, dico)

	encoder = encoder[0]
	encoder.load_state_dict(reloaded['encoder'])
	bpe_model = fastBPE.fastBPE(os.path.abspath('/storage/vsub851/py-vgdl-1/rule_rep/TransCoder/data/BPE_with_comments_codes'))
	assert len(reloaded['encoder'].keys()) == len(list(p for p, _ in encoder.state_dict().items()))
	return encoder, bpe_model, dico, reloaded_params

def test_data_encoder_load(base_path, games, sets = ['SpriteSet', 'InteractionSet', 'TerminationSet'], return_type = 'block', model_name = 'model_2.pth'):
	rule_dict = data_load(base_path = base_path, games = games, sets = sets, return_type = return_type)
	# print(rule_dict)

	dataset = dataset(rule_dict, dataset_type= return_type)
	# print(dataset)

	encoder, bpe_model, dico, reloaded_params = encoder_load(base_path = base_path, model_name = model_name)
	# print(dico)