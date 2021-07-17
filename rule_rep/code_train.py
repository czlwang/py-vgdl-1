import os
import sys
import numpy as np
import pandas as pd
import fastBPE
from tqdm import tqdm

import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

import transformers 

from code_data_load import *
from code_embed_model import *

import TransCoder.preprocessing.src.code_tokenizer as code_tokenizer
from TransCoder.XLM.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from TransCoder.XLM.src.model import build_model
from TransCoder.XLM.src.utils import AttrDict

if cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'

def train(train_data, base_path, modelname, encoder, bpe_model, reloaded_params, dico, num_labels, lr = 0.05, dropout = 0.33,  num_epochs = 10, hidden_size = 200):
	loss_fn = nn.CrossEntropyLoss().to(device)

	encoder = encoder.to(device)
	classifier = EncMLP(encoder = encoder, num_labels = num_labels, dropout = dropout, hidden_size = hidden_size)
	classifier = classifier.to(device)
	classifier.train()

	optimizer = optim.Adam(classifier.parameters(), lr = lr)
	lang1 = 'python'
	tokenizer = getattr(code_tokenizer, f'tokenize_{lang1}')
	lang1 = 'python_sa'
	lang1_id = reloaded_params.lang2id[lang1]

	print('Beginning Training')
	for epoch in tqdm(range(num_epochs)):
		total_loss = 0
		classifier.train()
		for code, label in train_data:
			tokens = [t for t in tokenizer(code)]
			tokens = bpe_model.apply(tokens)
			tokens = ['</s>'] + tokens + ['/s']
			input = ' '.join(tokens)
			len1 = len(input.split())
			labels = []
			for i in range(len1):
				labels.append(label)
			len1 = torch.LongTensor(1).fill_(len1).to(device)
			labels = torch.tensor(labels).long().to(device)

			x1 = torch.LongTensor([dico.index(w) for w in input.split()]).to(device)[:, None]
			langs1 = x1.clone().fill_(lang1_id)
			outputs = classifier.forward(x = x1, lengths = len1, langs = langs1)

			loss = loss_fn(outputs, labels)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss = total_loss + loss.item()
		print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_data)))
	print('Training is finished')

	save_path = os.path.join(base_path, 'checkpoints', modelname)
	torch.save(classifier.state_dict(), save_path)

def test_train(base_path, model_name, save_name, games, sets = ['InteractionSet', 'SpriteSet', 'TerminationSet'], return_type = 'block', lr = 0.05, dropout = 0.33, num_epochs = 10, hidden_size = 200):
	print('Data loading beginning')
	rule_dict = data_load(base_path = base_path, games = games, sets = sets, return_type = return_type)
	num_labels = len(sets)
	train_dataset = dataset(rule_dict = rule_dict, dataset_type = return_type)
	print('Data loading finished')

	print(('Encoder loading from model {}').format(model_name))
	encoder, bpe_model, dico, reloaded_params = encoder_load(base_path = base_path, model_name = model_name)

	train(train_data = train_dataset, base_path = base_path, modelname = save_name, encoder = encoder, bpe_model = bpe_model, reloaded_params = reloaded_params, dico = dico, 
		num_labels = num_labels, lr = lr, dropout = dropout, num_epochs = num_epochs, hidden_size = hidden_size)

test_train(base_path = '/storage/vsub851/py-vgdl-1/rule_rep', model_name = 'model_2.pth', save_name = 'mlp1.pt', games = ['aliens_py'])