import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

import transformers

if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'

lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

class MLP(nn.Module):
	def __init__(self, num_labels, dropout = 0.25, lm = None):
		self.lm = lm
		self.dropout = nn.Dropout(dropout)
		self.linear = nn.Linear(lm.config.hidden_size, num_labels)
		self.relu = nn.ReLU()
	def forward(self, input_ids = None, attention_mask = None, layer = -1):
		lm_output = self.lm(input_ids, attention_mask, output_hidden_states = True)
		hidden_states = lm_output.hidden_states[layer]
		hidden_states = self.dropout(hidden_states)
		output = self.linear(hidden_states)
		output = self.relu(output)
		return output