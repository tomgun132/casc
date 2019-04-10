from __future__ import division, print_function, unicode_literals

import os
import time

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

import data
from model import Model

USE_CUDA = True

class Train(object):
	def __init__(self, base_path, topic):
		i2w, w2i, sents, lines, w_pos, w_concept,  entities = data.load(base_path + "/docs/" + topic)
		if USE_CUDA:
			sents = sents.cuda()
		self.i2w = i2w
		self.w2i = w2i
		self.sents = sents
		self.lines = lines
		self.w_concept = w_concept
		
		train_dir = 'log/'
		self.model_dir = os.path.join(train_dir,'model')
		
	def save_model(self, running_avg_loss, iter):
		state = {
			'iter' : iter,
			'reader_state_dict' : self.model.reader.state_dict(),
			'recaller_state_dict' : self.model.recaller.state_dict(),
			'optimizer' : self.optimizer.state_dict(),
			'current_loss' : running_avg_loss
		}
		
		model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
		torch.save(state, model_save_path)
		
	def setup_train(self, net_type, model_file_path=None):
		x_dim = len(self.w2i)
		y_dim = len(self.w2i)
		hidden_size = 500
		emb_size = 500
		self.net_type = net_type
		self.model = Model(x_dim, emb_size, hidden_size, net_type, model_file_path)
		
		self.params = list(self.model.reader.parameters()) + list(self.model.recaller.parameters())
		# initial_lr = 0.001
		self.optimizer = torch.optim.Adadelta(self.params) #edit later
		iter, loss = 0,0
		if model_file_path is not None:
			state = torch.load(model_file_path, map_location= lambda storage, location: storage)
			iter = state['iter']
			loss = state['current_loss']
		return iter, loss
		
	def mse(self, pred, label):
		loss = torch.mean((pred - label) ** 2, dim=1)
		return 0.5 * torch.sum(loss)
	
	def train(self, iter, num_summs,net_type='lstm', model_file_path=None):
		iter_, loss = self.setup_train( net_type, model_file_path=None)
		if iter_ > iter: iter=iter_
		
		for i in range(iter):
			self.optimizer.zero_grad()
			
			cg, input_emb = self.model.reader(self.sents)
			rec_hid = cg
			if self.net_type == 'lstm':
				cg, _ = cg
			cg = cg.view(1,1,-1)
			v_size = len(self.w2i)
			seq_lens = self.sents.size(1)
			
			recons = list()
			attns = list()
			for j in range(num_summs):
				rec_hid, rec_out, rec_attn = self.model.recaller(cg, rec_hid, input_emb, self.sents)
				rec_out = rec_out.squeeze()
				rec_attn = rec_attn.squeeze()
				recons.append(rec_out)
				attns.append(rec_attn)
			
			recons = torch.stack(recons,0) # num_summs x vsize
			attns = torch.stack(attns,0) # num_summs x seq_len
			out_score = torch.mm(attns.t(),recons)
			loss = self.mse(self.sents.squeeze(), out_score) + 0.001 * torch.sum(torch.norm(recons, p=1, dim=1))
			loss.backward()
			clip_grad_norm_(self.model.reader.parameters(), 10)
			clip_grad_norm_(self.model.recaller.parameters(), 10)
			self.optimizer.step()
			
			print('iteration: ',i)
			print('loss: ',loss)
			
		return recons.cpu().data.numpy(), attns.cpu().data.numpy()
