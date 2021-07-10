import sys
import os
import numpy as np
import random
from tqdm import tqdm

import torch

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

base_path = '/Users/vsubramaniam/Documents/Infolab/Games/py-vgdl-1'
game = 'aliens'
modelname = 'mlp1.pt'

def train_model(train_data, modelname, base_path, num_labels, lr = 0.05, dropout = 0.25, num_epochs = 10, lm = None, bert_layer = -1):
	optimizer = optim.Adam(classifier.parameters(), lr = lr)

	loss_fn = nn.CrossEntropyLoss().to(device)

	classifier = MLP(num_labels, dropout, lm)
	classifier = classifier.to(device)
	classifier.train()

	print('Beginning training')
	for epoch in tqdm(range(num_epochs)):
		for rule_dict, label in train_data:
			label = torch.tensor(label)

			input_ids = torch.tensor(rule_dict['input_ids']).long().to(device)
			attention_mask = torch.tensor(rule_dict['attention_mask']).long().to(device)

			outputs = classifier.forward(input_ids, attention_mask, bert_layer)

			loss = loss_fn(outputs, labels_batch)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss = total_loss + loss.item()

		print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_corpus)))
	print('TRAINING IS FINISHED')

	save_path = os.path.join(base_path, 'rule_rep', 'outputs', modelname)
	torch.save(classifier.state_dict(), save_path)

def test_train(base_path, modelname, game, num_labels, lr = 0.05, dropout = 0.25, num_epochs = 10, lm = None, bert_layer = -1):
	game_path = os.path.join(base_path, 'vgdl', 'games')
	rule_dict = load_vgdl(game_path, game)

	tokenized_rule_dict = {}
	for rule_type in rule_dict:
		tokenized_rule_dict[rule_type] = tokenize_rules(rule_dict[rules])
	dataset = build_dataset(tokenized_rule_dict)
	dataset = random.shuffle(dataset)

	train_model(train_data, modelname, base_path, num_labels, lr, dropout, num_epochs, lm, bert_layer)

test_train(base_path, modelname, game)