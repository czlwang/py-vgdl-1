import sys
import os
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

from data_loader import *
from pred_model import *

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	device = 'cpu'

base_path = '/storage/vsub851/py-vgdl-1'
game = 'aliens'
modelname = 'mlp1.pt'
lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

def train_model(train_data, modelname, base_path, num_labels, lr = 0.05, dropout = 0.25, num_epochs = 10, lm = None, bert_layer = -1):
	loss_fn = nn.CrossEntropyLoss().to(device)

	classifier = MLP(num_labels, dropout, lm)
	classifier = classifier.to(device)
	classifier.train()

	optimizer = optim.Adam(classifier.parameters(), lr = lr)

	print('Beginning training')
	for epoch in tqdm(range(num_epochs)):
		total_loss = 0
		classifier.train()
		for rule_dict, labels in train_data:
			input_ids = []
			attention_mask = []
			input_ids.append(torch.tensor(rule_dict['input_ids']).long().to(device))
			attention_mask.append(torch.tensor(rule_dict['attention_mask']).long().to(device))
			labels = torch.tensor(labels).long().to(device)

			input_ids = torch.stack(input_ids).to(device)
			attention_mask = torch.stack(attention_mask).to(device)

			outputs = classifier.forward(input_ids, attention_mask, bert_layer)

			loss = loss_fn(outputs, labels)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss = total_loss + loss.item()

		print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_data)))
	print('TRAINING IS FINISHED')

	save_path = os.path.join(base_path, 'rule_rep', 'checkpoints', modelname)
	torch.save(classifier.state_dict(), save_path)

def test_train(base_path, modelname, game, num_labels, lr = 0.05, dropout = 0.25, num_epochs = 10, lm = None, bert_layer = -1):
	print('Beginning data loading from game {}'.format(game))
	game_path = os.path.join(base_path, 'vgdl', 'games')
	rule_dict = load_vgdl(game_path, game)

	tokenized_rule_dict = {}
	for rule_type in rule_dict:
		tokenized_rule_dict[rule_type] = tokenize_rules(rule_dict[rule_type])
	# print(tokenized_rule_dict)
	dataset = build_dataset(tokenized_rule_dict)
	# for rule_dict, _ in dataset:
	# 	input_ids = rule_dict['input_ids']
	# 	print(len(input_ids))
	random.shuffle(dataset)
	print('Data loading complete and dataset built')

	train_model(dataset, modelname, base_path, num_labels, lr, dropout, num_epochs, lm, bert_layer)

test_train(base_path, modelname, game, num_labels = 4, lm = lm_pretrained)