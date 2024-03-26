# this code for our DiG-In-GNN is based on PC-GNN's original code

import torch
import torch.nn as nn
from torch.nn import init
from model.neigh_gen_model import Discriminator

class DiGInLayer(nn.Module):
	def __init__(self, num_classes, inter1, lambda_1, nei_gen, device):
		"""
		Initialize the model
		:param num_classes: 2 for binary classification
		:param inter1: the inter-relation aggregator that output the final embedding
		"""
		super(DiGInLayer, self).__init__()
		self.device = device
		self.inter1 = inter1

		self.nei_gen = nei_gen

		self.xent = nn.CrossEntropyLoss()

		self.weight1 = nn.Parameter(torch.FloatTensor(128, inter1.embed_dim))
		# self.weight1_1 = nn.Parameter(torch.FloatTensor(64, 256))
		self.weight2 = nn.Parameter(torch.FloatTensor(num_classes, 128))
		init.xavier_uniform_(self.weight1)
		# init.xavier_uniform_(self.weight1_1)
		init.xavier_uniform_(self.weight2)
		self.lambda_1 = lambda_1
		self.epsilon = 0.1

	def forward(self, nodes, labels, train_flag=True, rl_train_flag = True, rl_has_trained = False):
		embeds1, label_scores, gen_feats, raw_feats, env_feats, env_raw_feats, rewards, label_lists, rate_list \
			= self.inter1(nodes, labels, train_flag, rl_train_flag, rl_has_trained, self.device)
		scores = self.weight1.mm(embeds1)
		# scores = self.weight1_1.mm(scores)
		scores = self.weight2.mm(scores)
		return scores.t(), label_scores, gen_feats, raw_feats, env_feats, env_raw_feats, rewards, label_lists, rate_list

	def to_prob(self, nodes, labels, train_flag=False, rl_train_flag = False, rl_has_trained = True):
		gnn_logits, label_logits, gen_feats, raw_feats, env_feats, env_raw_feats, rewards, label_lists, rate_list \
			= self.forward(nodes, labels, train_flag, rl_train_flag, rl_has_trained)
		gnn_scores = torch.softmax(gnn_logits, dim=1)
		label_scores = 0
		return gnn_scores, label_scores, rate_list

	def loss(self, nodes, labels, train_flag=True, rl_train_flag = True, rl_has_trained = False):
		gnn_scores, label_scores, gen_feats, raw_feats, env_feats, env_raw_feats, rewards, label_lists, rate_list \
			= self.forward(nodes, labels, train_flag, rl_train_flag, rl_has_trained)

		reward_loss = 0
		for idx in range(3):
			if len(rewards[idx]) != 0:
				reward_loss += self.xent(rewards[idx], label_lists[idx].squeeze())
		reward_loss /= 3

		gnn_loss = self.xent(gnn_scores, labels.squeeze())
		# loss of DiG-In-GNN
		final_loss = gnn_loss  + 0.5 * reward_loss
		# reward_loss: value prediction loss

		context_loss = []
		env_loss = []
		for i in range(len(self.nei_gen)):
			c_loss, e_loss = self.nei_gen[i].discriminator.forward(gen_feats[i],raw_feats[i], env_feats[i], env_raw_feats[i], labels)
			context_loss.append(c_loss)
			env_loss.append(e_loss)

		return final_loss, context_loss, env_loss
