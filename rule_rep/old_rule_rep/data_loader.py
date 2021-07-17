import os
import sys
import numpy as np

import transformers

tokenizer =  transformers.BertTokenizer.from_pretrained('bert-base-uncased')

base_path = '/storage/vsub851/py-vgdl-1/vgdl/games'
game = ['aliens']

def load_vgdl(input_path, games):
	rule_dicts = []
	for game in games:
		game = game + '.txt'
		game_path = os.path.join(input_path, game)
		f = open(game_path, 'r')
		file = f.readlines()
		rule_dict = {}
		set_type = ''
		for line in file:
			line = line.strip()
			if 'BasicGame' in line or line == '':
				continue
			elif 'Set' in line or 'LevelMapping' in line:
				set_type = line
				rule_dict[set_type] = []
			try:
				if 'Set' not in line:
					rule_dict[set_type].append(line)
			except KeyError:
				print('Issue with this set:', set_type)
		rule_dicts.append(rule_dict)
	return rule_dicts

def tokenize_rules(rules):
	tokenized_rules = []
	for rule in rules:
		rule_tk = tokenizer.encode_plus(rule, return_attention_mask = True)
		tokenized_rules.append({'rule': rule, 'input_ids': rule_tk['input_ids'], 'attention_mask': rule_tk['attention_mask']})
	return tokenized_rules

def build_dataset(rule_dict):
	dataset = []
	label = -1
	for key in rule_dict:
		if key == 'SpriteSet':
			label = 0
		elif key == 'InteractionSet':
			label = 1
		elif key == 'LevelMapping':
			label = 2
		else:
			label = 3
		for d in rule_dict[key]:
			input_ids = d['input_ids']
			labels = []
			for i in range(len(input_ids)):
				labels.append(label)
			dataset.append((d, labels))
	return dataset

def test_data_loader(input_path, game, rule_type = None):
	rule_dicts = load_vgdl(input_path, game)
	# print(rule_dict)

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
	# print(tokenized_rule_dict)

	if tokenized_rule_dict:
		rule_dict = tokenized_rule_dict
		dataset = build_dataset(rule_dict)
	# print(dataset)

# test_data_loader(base_path, game)