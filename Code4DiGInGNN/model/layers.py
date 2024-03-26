import copy

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from utils.utils import getVISUAL
from operator import itemgetter
from model.attention import att_inter_agg
import numpy as np
import math

from model.RL_choose import RL_choose_neighs_and_get_features

class InterAgg(nn.Module):
    def __init__(
        self,
        features,
        feature_dim,
        embed_dim,
        train_pos,
        adj_lists,
        intraggs,
        inter="GNN",
        cuda=True,
    ):
        """
        Initialize the inter-relation aggregator
        :param features: the input node features or embeddings for all nodes
        :param feature_dim: the input dimension
        :param embed_dim: the embed dimension
        :param train_pos: positive samples in training set
        :param adj_lists: a list of adjacency lists for each single-relation graph
        :param intraggs: the intra-relation aggregators used by each single-relation graph
        :param inter: NOT used in this version, the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
        :param cuda: whether to use GPU
        """
        super(InterAgg, self).__init__()

        self.features = features
        self.dropout = 0.6
        self.adj_lists = adj_lists
        
        self.intra_agg1 = intraggs[0]
        self.intra_agg2 = intraggs[1]
        self.intra_agg3 = intraggs[2]
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.inter = inter
        self.cuda = cuda
        self.intra_agg1.cuda = cuda
        self.intra_agg2.cuda = cuda
        self.intra_agg3.cuda = cuda
        self.train_pos = train_pos

        self.weight1 = nn.Parameter(
            torch.FloatTensor(
                self.feat_dim * 3 * len(intraggs) + 1 * self.feat_dim, self.embed_dim
            )
        )
        init.xavier_uniform_(self.weight1)
        self.bn1 = nn.BatchNorm1d(self.feat_dim * 3 * len(intraggs) + 1 * self.feat_dim)

        self.bn = nn.BatchNorm1d(96)
        self.weight_rl1 = weight(self.feat_dim)
        self.weight_rl2 = weight(self.feat_dim)
        self.weight_rl3 = weight(self.feat_dim)

    def forward(
        self, nodes, labels, train_flag=True, rl_train_flag=True, rl_has_trained=False, device=torch.device("cpu")
    ):
        """
        :param nodes: a list of batch node ids
        :param labels: a list of batch node labels
        :param train_flag: indicates whether in training or testing mode
        :return combined: the embeddings of a batch of input node features
        """

        to_neighs = []
        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

            # stuff from DiG-In-GNN, not used in ours
        pos_scores = 0
        center_scores = 0

        # get neighbor node id list for each batch node and relation
        r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
        r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

        r1_scores = 0
        r2_scores = 0
        r3_scores = 0
        r1_sample_num_list = []
        r2_sample_num_list = []
        r3_sample_num_list = []

        # intra-relation aggregation
        """
        Take relation 1 as an example
        r1_feats: generated intra-relation features
        r1_gen_feats: guidance nodes
        r1_raw_feats: original features
        label_list1: labels of nodes selected for training the value predictor, VP
        rate1: the rate of neighbor selection
        r1_env_feats: generated environment features
        r1_env_raw_feats: original environment features
		"""
        (
            agg_feats1,
            r1_feats,
            r1_scores,
            r1_gen_feats,
            r1_raw_feats,
            reward1,
            label_list1,
            rate1,
            r1_env_feats,
            r1_env_raw_feats,
        ) = self.intra_agg1.forward(
            nodes,
            labels,
            r1_list,
            center_scores,
            r1_scores,
            pos_scores,
            r1_sample_num_list,
            self.weight_rl1,
            self.bn,
            train_flag,
            rl_train_flag,
            rl_has_trained,
            device
        )
        (
            agg_feats2,
            r2_feats,
            r2_scores,
            r2_gen_feats,
            r2_raw_feats,
            reward2,
            label_list2,
            rate2,
            r2_env_feats,
            r2_env_raw_feats,
        ) = self.intra_agg2.forward(
            nodes,
            labels,
            r2_list,
            center_scores,
            r2_scores,
            pos_scores,
            r2_sample_num_list,
            self.weight_rl1,
            self.bn,
            train_flag,
            rl_train_flag,
            rl_has_trained,
            device
        )
        (
            agg_feats3,
            r3_feats,
            r3_scores,
            r3_gen_feats,
            r3_raw_feats,
            reward3,
            label_list3,
            rate3,
            r3_env_feats,
            r3_env_raw_feats,
        ) = self.intra_agg3.forward(
            nodes,
            labels,
            r3_list,
            center_scores,
            r3_scores,
            pos_scores,
            r3_sample_num_list,
            self.weight_rl1,
            self.bn,
            train_flag,
            rl_train_flag,
            rl_has_trained,
            device
        )

        gen_feats = []
        gen_feats.append(r1_gen_feats)
        gen_feats.append(r2_gen_feats)
        gen_feats.append(r3_gen_feats)

        env_feats = []
        env_feats.append(r1_env_feats)
        env_feats.append(r2_env_feats)
        env_feats.append(r3_env_feats)

        raw_feats = []
        raw_feats.append(r1_raw_feats)
        raw_feats.append(r2_raw_feats)
        raw_feats.append(r3_raw_feats)

        env_raw_feats = []
        env_raw_feats.append(r1_env_raw_feats)
        env_raw_feats.append(r2_env_raw_feats)
        env_raw_feats.append(r3_env_raw_feats)

        rewards = []
        rewards.append(reward1)
        rewards.append(reward2)
        rewards.append(reward3)

        label_lists = []
        label_lists.append(label_list1)
        label_lists.append(label_list2)
        label_lists.append(label_list3)

        # get features or embeddings for batch nodes
        if self.cuda and isinstance(nodes, list):
            index = torch.LongTensor(nodes).cuda()
        else:
            index = torch.LongTensor(nodes)
        self_feats = self.features(index)

        # cat_feats = torch.cat((self_feats, agg_feats1, agg_feats2, agg_feats3, r1_feats, r2_feats, r3_feats), dim=1)

        cat_feats = torch.cat((self_feats, r1_feats, r2_feats, r3_feats), dim=1)
        cat_feats = self.bn1(cat_feats)

        combined = F.relu(cat_feats.mm(self.weight1).t())


        try:
            label_lists = [label_list.to(device) for label_list in label_lists]
        except:
            pass
        
        return (
            combined,
            center_scores,
            gen_feats,
            raw_feats,
            env_feats,
            env_raw_feats,
            rewards,
            label_lists,
            [rate1, rate2, rate3],
        )


class IntraAgg(nn.Module):
    def __init__(
        self, features, feat_dim, embed_dim, train_pos, rho, gen, rl, relation, cuda=False
    ):
        """Initialize the intra-relation aggregator
        """
        super(IntraAgg, self).__init__()

        self.features = features
        self.cuda = cuda
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.train_pos = train_pos
        self.rho = rho
        self.relation = relation
        
        
        self.weight1 = nn.Parameter(
            torch.FloatTensor(3 * self.feat_dim, 3 * self.feat_dim)
        )
        self.weight2 = nn.Parameter(
            torch.FloatTensor(2 * self.feat_dim, 2 * self.feat_dim)
        )
        self.bn = nn.BatchNorm1d(3 * self.feat_dim)
        self.bn1 = nn.BatchNorm1d(3 * self.feat_dim)
        self.bn2 = nn.BatchNorm1d(2 * self.feat_dim)

        self.gen = gen 
        self.rl = rl  
        init.xavier_uniform_(self.weight1)
        init.xavier_uniform_(self.weight2)

    def forward(
        self,
        nodes,
        batch_labels,
        to_neighs_list,
        batch_scores,
        neigh_scores,
        pos_scores,
        sample_list,
        weight,
        bn,
        train_flag,
        rl_train_flag,
        rl_has_trained,
        device
    ):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        """

        if self.cuda:
            self_feats = self.features(torch.LongTensor(nodes).cuda())
        else:
            self_feats = self.features(torch.LongTensor(nodes))
        agg_feats = get_agg_feats(self.features, to_neighs_list, self.cuda)

        # context_level contrast
        cat_feats = torch.cat((self_feats, agg_feats), dim=1)
        gen_feats, raw_feats = self.gen(
            cat_feats
        ) 

        env_agg_feats = get_env_agg_feats(
            nodes, self.features, copy.deepcopy(to_neighs_list), self.cuda
        )
        if self.cuda:
            zero = torch.zeros(self_feats.shape[0], self_feats[0].shape[0]).cuda()
        else:
            zero = torch.zeros(self_feats.shape[0], self_feats[0].shape[0])
        env_gen_feats = torch.cat((zero, env_agg_feats), dim=1) 
        env_gen_feats, env_raw_feats = self.gen(env_gen_feats)


        reward = []
        label_list = []
        rate = []

        if train_flag and rl_train_flag is False:
            feats = torch.cat((agg_feats, gen_feats), dim=1)
            to_feats = F.relu(self.bn1(feats).mm(self.weight1))

        if train_flag is False and rl_train_flag is False:
            if rl_has_trained is False:
                feats = torch.cat((agg_feats, gen_feats), dim=1)
                to_feats = F.relu(self.bn1(feats).mm(self.weight1))
            else:
                (
                    rl_agg_feats,
                    observations1,
                    observations2,
                    label_list,
                    rate,
                ) = RL_choose_neighs_and_get_features(
                    batch_labels,
                    self.features,
                    gen_feats,
                    self_feats,
                    to_neighs_list,
                    self.rl,
                    self.cuda,
                )
                feats = torch.cat((rl_agg_feats, gen_feats), dim=1)
                to_feats = F.relu(self.bn1(feats).mm(self.weight1))
                self.rl.memory_clear() 

        if train_flag and rl_train_flag:
            (
                rl_agg_feats,
                observations1,
                observations2,
                label_list,
                rate,
            ) = RL_choose_neighs_and_get_features(
                batch_labels,
                self.features,
                gen_feats,
                self_feats,
                to_neighs_list,
                self.rl,
                self.cuda,
            )

            feats = torch.cat((rl_agg_feats, gen_feats), dim=1)
            to_feats = F.relu(self.bn1(feats).mm(self.weight1))


            observations = torch.cat((observations1, observations2), dim=1)
            observations = F.relu(self.bn1(observations).mm(self.weight1))

            reward = weight(observations)
            reward_rl = torch.softmax(nn.LeakyReLU(0.2)(reward), dim=1)


            self.rl.store_reward(reward_rl)

            self.rl.learn(label_list)
            label_list = torch.LongTensor(label_list)

            self.rl.memory_clear()
        elif train_flag and rl_has_trained: 
            (
                rl_agg_feats,
                observations1,
                observations2,
                label_list,
                rate,
            ) = RL_choose_neighs_and_get_features(
                batch_labels,
                self.features,
                gen_feats,
                self_feats,
                to_neighs_list,
                self.rl,
                self.cuda,
            )

            feats = torch.cat((rl_agg_feats, gen_feats), dim=1)
            to_feats = F.relu(self.bn1(feats).mm(self.weight1))
            self.rl.memory_clear()  

        return (
            agg_feats,
            to_feats,
            [],
            gen_feats,
            raw_feats,
            reward,
            label_list,
            rate,
            env_gen_feats,
            env_raw_feats,
        )


def get_agg_feats(features, samp_neighs, cuda):
    neighs = []
    for samp in samp_neighs:
        neighs.append(set(samp))
    unique_nodes_list = list(set.union(*neighs))
    unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

    # intra-relation aggregation only with sampled neighbors
    mask = Variable(torch.zeros(len(neighs), len(unique_nodes)))
    column_indices = [unique_nodes[n] for samp_neigh in neighs for n in samp_neigh]
    row_indices = [i for i in range(len(neighs)) for _ in range(len(neighs[i]))]
    mask[row_indices, column_indices] = 1
    if cuda:
        mask = mask.cuda()
    num_neigh = mask.sum(1, keepdim=True)
    mask = mask.div(num_neigh)  # mean aggregator
    if cuda:
        # self_feats = self.features(torch.LongTensor(nodes).cuda())
        embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())
    else:
        # self_feats = self.features(torch.LongTensor(nodes))
        embed_matrix = features(torch.LongTensor(unique_nodes_list))
    agg_feats = mask.mm(embed_matrix)

    return agg_feats


def get_env_agg_feats(nodes, features, samp_neighs, cuda):
    neighs = []
    for node, samp in zip(nodes, samp_neighs):
        if len(samp) != 1:
            s = samp.remove(node)
            neighs.append(set(samp))
        else:
            neighs.append(set(samp))
    unique_nodes_list = list(set.union(*neighs))
    unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

    # intra-relation aggregation only with sampled neighbors
    mask = Variable(torch.zeros(len(neighs), len(unique_nodes)))
    column_indices = [unique_nodes[n] for samp_neigh in neighs for n in samp_neigh]
    row_indices = [i for i in range(len(neighs)) for _ in range(len(neighs[i]))]
    mask[row_indices, column_indices] = 1
    if cuda:
        mask = mask.cuda()
    num_neigh = mask.sum(1, keepdim=True)
    mask = mask.div(num_neigh)  # mean aggregator
    if cuda:
        # self_feats = self.features(torch.LongTensor(nodes).cuda())
        embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())
    else:
        # self_feats = self.features(torch.LongTensor(nodes))
        embed_matrix = features(torch.LongTensor(unique_nodes_list))
    agg_feats = mask.mm(embed_matrix)

    return agg_feats


class weight(nn.Module):
    def __init__(self, feat_dim):
        super(weight, self).__init__()
        self.feat_dim = feat_dim
        self.reward_weight1 = nn.Parameter(
            torch.FloatTensor(3 * self.feat_dim, 2 * self.feat_dim)
        )
        self.reward_weight2 = nn.Parameter(torch.FloatTensor(2 * self.feat_dim, 2))
        # self.reward_weight3 = nn.Parameter(torch.FloatTensor(64, 2))
        init.xavier_uniform_(self.reward_weight1)
        init.xavier_uniform_(self.reward_weight2)
        # init.xavier_uniform_(self.reward_weight3)
        self.bn = nn.BatchNorm1d(3 * self.feat_dim)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, features):
        features = self.bn(features)
        features = features.mm(self.reward_weight1)
        # features = self.bn1(features)
        features = features.mm(self.reward_weight2)
        # l = features.mm(self.reward_weight3)
        l = features
        return l
