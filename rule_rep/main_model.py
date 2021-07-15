import os 
import sys
import numpy as np  
import random
import statistics
import argparse
import logging

import torch
import torch.nn as nn
import torch.functional as F 
import torch.cuda as cuda

import transformers

from data_loader import *
from train_model import *
from eval_model import *

seed = 0
random.seed(seed)
if cuda.is_available():
	device = 'cuda'
	cuda.manual_seed_all(seed)
else:
	device = 'cpu'

lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

def get_cmd_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = '/storage/vsub851/py-vgdl-1',
		help = 'Base path with the data and models')
	ap.add_argument('-m', '--model', action = 'store', type = str, dest = 'model', default = 'mlp1.pt',
		help = 'Pytorch model that predicts the heads that can be saved to after training or loaded in for evaluation')
	ap.add_argument('-g', '--games', action = 'store', type = list, dest = 'games', default = ['aliens', 'boulderdash', 'chase', 'frogs', 'missilecommand', 'portals', 'sokoban', 'survivezombies', 'zelda'], 
		help = 'Games that are given as input for model training and evaluation')
	ap.add_argument('-s', '--split', action = 'store', type = float, dest = 'split', default = 0.9, 
		help = 'Training/testing split for the dataset')
	ap.add_argument('-b', '--bert', action = 'store', type = int, dest = 'bert_layer', default = -1, 
		help = 'BERT layer to grab representations from')

	#Model Hyperparameters
	ap.add_argument('--lr', type=float, default=0.05, action = 'store', dest = 'lr')
	ap.add_argument('--epochs', type=int, default=10, action = 'store', dest = 'num_epochs')
	ap.add_argument('--dropout', type=float, default=0.25, action = 'store', dest=  'dropout')
	ap.add_argument('--hidden_size', type = int, default = 200, action = 'store', dest = 'hidden_size')

	return ap.parse_args()

def main():
	args = get_cmd_arguments()
	games = args.games
	base_path = args.base_path
	game_path = os.path.join(base_path, 'vgdl', 'games')
	split = args.split
	bert_layer = args.bert_layer

	print(('Data loading beginning from {} games').format(games))
	rule_dicts = load_vgdl(game_path, games)

	tokenized_rule_dict = {}
	for rule_dict in rule_dicts:
		for rule_type in rule_dict:
			try:
				tokenized_rule_dict[rule_type] += tokenize_rules(rule_dict[rule_type])
			except KeyError:
				tokenized_rule_dict[rule_type] = tokenize_rules(rule_dict[rule_type])

	dataset = build_dataset(tokenized_rule_dict)
	random.shuffle(dataset)
	print('Data loading complete')

	split_point = int(split * len(dataset))
	training_data = dataset[:split_point]
	testing_data = dataset[split_point:]
	print(('Training Dataset size {}, Testing Dataset size {}').format(len(training_data), len(testing_data)))

	train_model(train_data = training_data, modelname = args.model, base_path = base_path, num_labels = 4, lr = args.lr, dropout = args.dropout, 
		num_epochs = args.num_epochs, lm = lm_pretrained, bert_layer = bert_layer, hidden_size = args.hidden_size)

	print(eval_model(test_data = testing_data, modelname = args.model, base_path = base_path, num_labels = 4, dropout = args.dropout, lm = lm_pretrained, 
		bert_layer = bert_layer, hidden_size = args.hidden_size))

if __name__ == '__main__':
	main()