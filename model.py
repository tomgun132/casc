from __future__ import division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

USE_CUDA = True
random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_rnn_wt(rnn):
	for names in rnn._all_weights:
		for name in names:
			if name.startswith('weight_'):
				wt = getattr(rnn, name)
				wt.data.uniform_(-0.1, 0.1)
			elif name.startswith('bias_'):
				# set forget bias to 1
				bias = getattr(rnn, name)
				bias.data.uniform_(-0.1,0.1)


def init_linear_wt(linear):
	linear.weight.data.uniform_(-0.1,0.1)
	if linear.bias is not None:
		linear.bias.data.uniform_(-0.1,0.1)

def init_wt_unif(wt):
	wt.data.uniform_(-0.1, 0.1)

	
class Reader(nn.Module):

	def __init__(self, v_size, emb_dim, h_dim, net_type='lstm'):
		super(Reader, self).__init__()
		self.embedding = nn.Linear(v_size,emb_dim)
		init_linear_wt(self.embedding)
		if net_type == 'lstm':
			self.rnn = nn.LSTM(emb_dim, h_dim, num_layers=1, batch_first=True, bidirectional=False)
		elif net_type == 'gru':
			self.rnn = nn.GRU(emb_dim, h_dim, num_layers=1, batch_first=True, bidirectional=False)
		init_rnn_wt(self.rnn)
		
	def forward(self, input):
		h_v = torch.tanh(self.embedding(input))
		_, hidden = self.rnn(h_v)
		return hidden, h_v
	
class Recaller(nn.Module):
	
	def __init__(self,h_dim, out_size, net_type='lstm'):
		super(Recaller, self).__init__()
		
		self.net_type = net_type
		self.h_dim = h_dim
		
		if self.net_type == 'lstm':
			self.rnn = nn.LSTM(h_dim, h_dim, num_layers=1, batch_first=True, bidirectional=False)
		elif self.net_type == 'gru':
			self.rnn = nn.GRU(h_dim, h_dim, num_layers=1, batch_first=True, bidirectional=False)
		init_rnn_wt(self.rnn)
		
		self.attn_network_hidden = AttnHidden(h_dim)
		self.attn_network_out = AttnOut()
		
		self.out_linear = nn.Linear(h_dim,h_dim)
		init_linear_wt(self.out_linear)
		self.update_hidden_linear1 = nn.Linear(h_dim,h_dim, bias=False)
		init_linear_wt(self.update_hidden_linear1)
		self.update_hidden_linear2 = nn.Linear(h_dim,h_dim, bias=False)
		init_linear_wt(self.update_hidden_linear2)
		self.projection_linear = nn.Linear(h_dim,out_size)
		init_linear_wt(self.projection_linear)
		self.lambda_c = nn.Parameter(torch.rand(1))
		# self.lambda_c_1 = nn.Parameter(torch.rand(1))
		# self.lambda_c_2 = nn.Parameter(torch.rand(1))
		
	def forward(self, cg, hd_t_1, h_v, x):
		b, seq_len, _ = x.size()
		output, hd_t = self.rnn(cg, hd_t_1)
		ho_t_ = torch.tanh(self.out_linear(output)) # b x 1 x hid
		ch_t, ah_t = self.attn_network_hidden(h_v, ho_t_)
		
		ho_t = torch.tanh(self.update_hidden_linear1(ch_t) + self.update_hidden_linear2(ho_t_))
		s_t_ = torch.sigmoid(self.projection_linear(ho_t))
		
		co_t, ao_t = self.attn_network_out(x, s_t_, ah_t)
		s_t_ = s_t_.view(co_t.size()) # b x V
		lambda_c = torch.sigmoid(self.lambda_c) 
		s_t = lambda_c * co_t + (1-lambda_c) * s_t_
		# s_t = self.lambda_c_1 * co_t + self.lambda_c_2 * s_t
		return hd_t, s_t, ao_t
	
	
class AttnHidden(nn.Module):
	
	def __init__(self, h_dim):
		super(AttnHidden, self).__init__()
		self.attn_linear = nn.Linear(h_dim * 2, h_dim, bias=False)
		self.v = nn.Linear(h_dim, 1, bias=False)
			
	def forward(self, h_v, h_o):
		"""
		* h_v : b x seq_len x h_dim
		* h_o : b x h_dim
		"""
		b, seq_len, h_dim = h_v.size()
		h_o_expanded = h_o.expand(b, seq_len, h_dim).contiguous()
		feat = self.attn_linear(torch.cat((h_v,h_o_expanded), 2))
		score = self.v(torch.tanh(feat))
		score = score.view(b, seq_len) #* b x seq_len
		attn = F.softmax(score, dim=1)
		
		attn = attn.unsqueeze(1) #* b x 1 x seq_len
		context = attn.bmm(h_v) #* b x 1 x hid
		# context = context.view(-1, h_dim)
		attn = attn.view(-1, seq_len)
		return context, attn
		
class AttnOut(nn.Module):

	def __init__(self):
		super(AttnOut, self).__init__()
		self.lambda_a = nn.Parameter(torch.rand(1))
		# self.lambda_a_1 = nn.Parameter(torch.rand(1))
		# self.lambda_a_2 = nn.Parameter(torch.rand(1))
		
	def forward(self, input, output, hidden_attn):
		b, seq_len, v_size = input.size()
		score = input.bmm(output.transpose(1,2)).view(b,seq_len) #* b x seq_len
		attn = F.softmax(score, dim=1)
		lambda_a = torch.sigmoid(self.lambda_a)
		attn_out = lambda_a * attn + (1-lambda_a) * hidden_attn
		# attn_out = self.lambda_a_1 * attn + self.lambda_a_2 * hidden_attn
		attn_out = attn_out.view(b,-1,seq_len)
		context = attn_out.bmm(input) #* b x 1 x v
		context = context.view(b,v_size) #* b x v
		attn_out = attn_out.view(b, seq_len) #* b x seq_len
		return context, attn_out
		
	
class Model(object):
	def __init__(self, v_size, emb_dim, h_dim, net_type = 'lstm', model_file_path=None):
		reader = Reader(v_size, emb_dim, h_dim, net_type)
		recaller = Recaller(h_dim, v_size, net_type)
		
		if USE_CUDA:
			reader = reader.cuda()
			recaller = recaller.cuda()
				
		self.reader = reader	
		self.recaller = recaller	
		
		if model_file_path is not None:
			state = torch.load(model_file_path, map_location= lambda storage, location: storage)
			self.reader.load_state_dict(state['reader_state_dict'])
			self.recaller.load_state_dict(state['recaller_state_dict'], strict=False)
		

