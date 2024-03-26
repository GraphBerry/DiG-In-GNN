import random

import torch.nn as nn
import torch.nn.functional as F
from model.neigh_gen_layers import GraphConvolution
import torch
import numpy as np
import scipy.sparse as sp


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.relu(x)

class Gen(nn.Module):
    def __init__(self, latent_dim, dropout, num_pred, feat_shape):
        super(Gen, self).__init__()
        self.num_pred = num_pred
        self.feat_shape = feat_shape
        # Discriminator is used to judge the similarity between the generated guidance node features and the positive and negative samples
        self.discriminator = Discriminator(2 * feat_shape, 1)

        # The generator network layers
        self.fc1 = nn.Linear(latent_dim, 512).requires_grad_(True)
        self.fc2 = nn.Linear(512, 2048).requires_grad_(True)
        self.fc_flat = nn.Linear(2048, latent_dim).requires_grad_(True)
        self.bn0 = nn.BatchNorm1d(latent_dim).requires_grad_(False)
        # These normalization layers are optional, but self.bn0 is needed, otherwise it cannot be trained well
        # self.bn1 = nn.BatchNorm1d(256).requires_grad_(False)
        # self.bn2 = nn.BatchNorm1d(256).requires_grad_(False)
        self.dropout = dropout

    def forward(self, x):
        x = self.bn0(x)
        raw_feats = torch.tanh(x)
        x = (self.fc1(x))
        #x = self.bn1(x)
        x = (self.fc2(x))
        #x = self.bn2(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.tanh(self.fc_flat(x))
        return x, raw_feats

class Discriminator(nn.Module):
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        # context-level
        self.f_k = nn.Bilinear(n_h, n_h, 1).requires_grad_(False)
        # local-level
        self.f_k_env = nn.Bilinear(n_h, n_h, 1).requires_grad_(False)
        for m in self.modules():
            self.weights_init(m)
        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def get_contrast_sample(self, gen_feats, raw_feats, env_feats, env_raw_feats, labels):
        classify_label_1 = []
        classify_label_0 = []
        for i in range(len(labels)):
            if labels[i] == torch.tensor(1):
                classify_label_1.append(i)
            else:
                classify_label_0.append(i)

        # positive_feats = torch.mean(gen_feats[classify_label_1], dim=0, keepdim=True)
        # negative_feats = torch.mean(gen_feats[classify_label_0], dim=0, keepdim=True)
        # positive_sample = []
        # negative_sample = []
        # for i in range(len(labels)):
        #     if labels[i] == torch.tensor(1):
        #         positive_sample.append((gen_feats[random.sample(classify_label_1, 1)] + positive_feats) / 2)
        #         negative_sample.append((gen_feats[random.sample(classify_label_0, 1)] + negative_feats) / 2)
        #     else:
        #         positive_sample.append((gen_feats[random.sample(classify_label_0, 1)] + negative_feats) / 2)
        #         negative_sample.append((gen_feats[random.sample(classify_label_1, 1)] + positive_feats) / 2)


        positive_feats = torch.mean(gen_feats[classify_label_1], dim=0, keepdim=True)
        negative_feats = torch.mean(gen_feats[classify_label_0], dim=0, keepdim=True)
        
        raw_positive_feats = torch.mean(raw_feats[classify_label_1], dim=0, keepdim=True)
        raw_negative_feats = torch.mean(raw_feats[classify_label_0], dim=0, keepdim=True)

        """context - level"""
        positive_sample = []
        negative_sample = []
        raw_positive_sample = []
        raw_negative_sample = []
        for i in range(len(labels)):
            if labels[i] == torch.tensor(1):
                positive_sample.append(positive_feats)
                negative_sample.append(negative_feats)
                raw_positive_sample.append(raw_positive_feats)
                raw_negative_sample.append(raw_negative_feats)
            else:
                positive_sample.append(negative_feats)
                negative_sample.append(positive_feats)
                raw_positive_sample.append(raw_negative_feats)
                raw_negative_sample.append(raw_positive_feats)
                
        positive_sample = torch.cat(positive_sample)
        negative_sample = torch.cat(negative_sample)
        raw_positive_sample = torch.cat(raw_positive_sample)
        raw_negative_sample = torch.cat(raw_negative_sample)

        """local-level"""
        # get the environment info
        env_contrast_sample = (2 * env_feats + env_raw_feats) / 3


        return positive_sample, negative_sample, raw_positive_sample, raw_negative_sample, env_contrast_sample

    def get_contrast_loss(self, gen_feats, raw_feats, env_feats, env_raw_feats, labels):
        """
        :param gen_feats: [batchsize, 64]
        :param labels: labels
        :return: loss
        """
        gen_loss = []
        positive_sample, negative_sample, raw_positive_sample, raw_negative_sample, env_contrast_sample \
            = self.get_contrast_sample(gen_feats, raw_feats, env_feats, env_raw_feats, labels)

        """context_level"""
        # Calculate the loss of positive samples
        gen_loss.append(2 * self.f_k(positive_sample, gen_feats) + self.f_k(raw_positive_sample, gen_feats))
        # Calculate the loss of negative samples
        gen_loss.append(2 * self.f_k(negative_sample, gen_feats) + self.f_k(raw_negative_sample, gen_feats))
        # # Calculate the loss of positive samples
        # gen_loss.append(self.f_k(positive_sample, gen_feats))
        # # Calculate the loss of negative samples
        # gen_loss.append(self.f_k(negative_sample, gen_feats))
        context_logits = torch.cat(tuple(gen_loss))

        """patch_level"""
        # Calculate the similarity with the environment
        env_logits = self.f_k_env(env_contrast_sample, gen_feats)


        return context_logits, env_logits

    def forward(self, gen_feats, raw_feats, env_feats, env_raw_feats, labels):
        """
        contrastive learning loss
        """
        # gen_loss = []
        #
        # for feats in gen_feats:
        #     gen_loss.append(self.get_contrast_loss(feats, labels))
        #
        # return gen_loss

        return self.get_contrast_loss(gen_feats, raw_feats, env_feats, env_raw_feats, labels)

