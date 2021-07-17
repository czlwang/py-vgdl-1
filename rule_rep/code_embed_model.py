import os
import sys
import numpy as np

import torch 
import torch.nn as nn
import torch.functional as F
import torch.cuda

import transformers
import statistics

class EncMLP(nn.Module):
	def __init__(self, encoder, num_labels, hidden_size, dropout = 0.25):
		super(EncMLP, self).__init__()
		self.encoder = encoder
		self.dropout = nn.Dropout(dropout)
		self.linear1 = nn.Linear(1024, hidden_size)
		self.linear2 = nn.Linear(hidden_size, hidden_size)
		self.linear3 = nn.Linear(hidden_size, num_labels)
	def forward(self, x, lengths, langs):
		encoding = self.encoder('fwd', x = x, lengths = lengths, langs = langs, causal = False)
		drop_enc = self.dropout(encoding)
		lin1 = self.linear1(drop_enc)
		lin2 = self.linear2(lin1)
		output = self.linear3(lin2)
		output = output.squeeze(1)
		return output