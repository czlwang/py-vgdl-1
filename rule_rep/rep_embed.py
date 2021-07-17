import os
import sys
import fastBPE
import torch

import TransCoder.preprocessing.src.code_tokenizer as code_tokenizer
from TransCoder.XLM.src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from TransCoder.XLM.src.model import build_model
from TransCoder.XLM.src.utils import AttrDict

def encode_py(model_path, input):
	reloaded = torch.load(model_path, map_location = 'cpu')
	reloaded['encoder'] = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in reloaded['encoder'].items()}
	reloaded_params = AttrDict(reloaded['params'])

	dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

	reloaded_params['reload_model'] = ','.join([model_path]*2)
	encoder, _ = build_model(reloaded_params, dico)

	encoder = encoder[0]
	encoder.load_state_dict(reloaded['encoder'])
	assert len(reloaded['encoder'].keys()) == len(list(p for p, _ in encoder.state_dict().items()))

	if torch.cuda.is_available(): 
		encoder.cuda()
		DEVICE = 'cuda'

	encoder.eval()
	bpe_model = fastBPE.fastBPE(os.path.abspath('/storage/vsub851/py-vgdl-1/rule_rep/TransCoder/data/BPE_with_comments_codes'))

	with torch.no_grad():
		lang1 = 'python'
		tokenizer = getattr(code_tokenizer, f'tokenize_{lang1}')
		lang1 = 'python_sa'
		lang1_id = reloaded_params.lang2id[lang1]

		tokens = [t for t in tokenizer(input)]
		tokens = bpe_model.apply(tokens)
		tokens = ['</s>'] + tokens + ['</s>']
		input = ' '.join(tokens)
		len1 = len(input.split())
		len1 = torch.LongTensor(1).fill_(len1).to(DEVICE)

		x1 = torch.LongTensor([dico.index(w) for w in input.split()]).to(DEVICE)[:, None]
		langs1 = x1.clone().fill_(lang1_id)
		enc1 = encoder('fwd', x = x1, lengths = len1, langs = langs1, causal = False)
	return enc1, len1

enc1, len1 = encode_py('/storage/vsub851/py-vgdl-1/rule_rep/model_2.pth', 'def Help(x): print(\'Hello World \')')
# print(enc1.size())
# print(len1)

def decode_py(model_path, enc1, len1, n = 1, beam_size = 1):
	reloaded = torch.load(model_path, map_location = 'cpu')
	assert 'decoder' in reloaded or ('decoder_0' in reloaded and 'decoder_1' in reloaded)
	if 'decoder' in reloaded:
		decoders_names = ['decoder']
	else:
		decoders_names = ['decoder_0', 'decoder_1']
	for decoder_name in decoders_names:
		reloaded[decoder_name] = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in reloaded[decoder_name].items()}

	reloaded_params = AttrDict(reloaded['params'])

	dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
	reloaded_params['reload_model'] = ','.join([model_path]*2)
	_, decoder = build_model(reloaded_params, dico)
	decoder=  decoder[0]
	decoder.load_state_dict(reloaded['decoder'])
	assert len(reloaded['decoder'].keys()) == len(list(p for p, _ in decoder.state_dict().items()))

	if torch.cuda.is_available():
		DEVICE = 'cuda'
		decoder.cuda()
	decoder.eval()

	bpe_model = fastBPE.fastBPE(os.path.abspath('/storage/vsub851/py-vgdl-1/rule_rep/TransCoder/data/BPE_with_comments_codes'))

	with torch.no_grad():
		lang2 = 'python'
		detokenizer = getattr(code_tokenizer, f'detokenize_{lang2}')
		lang2 += '_sa'

		lang2_id = reloaded_params.lang2id[lang2]
		enc1 = enc1.transpose(0, 1)

		if n > 1:
			enc1 = enc1.repeat(n, 1, 1)
			len1 = len1.expand(n)

		if beam_size == 1:
			x2, len2 = decoder.generate(enc1, len1, lang2_id, max_len = int(min(reloaded_params.max_len, 3*len1.max().item() + 10)), 
				sample_temperature = None)

		else:
			x2, len2 = decoder.generate_beam(enc1, len1, lang2_id, max_len = int(min(reloaded_params.max_len, 3*len1.max().item() + 10)), early_stopping = False, length_penalty = 1.0,
				beam_size = beam_size)

		tok = []
		results = []
		for i in range(x2.shape[1]):
			wid = [dico[x2[j, i].item()] for j in range(len(x2))][1:]
			wid = wid[:wid.index(EOS_WORD)] if EOS_WORD in wid else wid
			tok.append(' '.join(wid).replace('@@ ', ''))

		results = []
		for t in tok:
			results.append(detokenizer(t))
		return results
# print(decode_py('/storage/vsub851/py-vgdl-1/rule_rep/model_2.pth', enc1, len1))