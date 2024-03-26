import random

import numpy as np
import torch
from torch.autograd import Variable


def RL_choose_neighs_and_get_features(
    labels, features, gen_feats, self_feats, neighs_list, rl, cuda
):
    """
    Reinforcement learning selects neighboring nodes and generates aggregation information
    :param features: features of all nodes
    :param gen_feats: guidance node feature
    :param self_feats: feature of the target nodes in the node batch
    :param neighs_list: list of neighbors
    :param rl: neighbor selector
    """
    """
    Steps:
    1. For each target node, obtain all its states [target node feature, neighbor node feature, generated feature]
       shape = [number of neighboring nodes of the target node, 128(32+32+64)]
    2. Input the state into the policy network to obtain the action - select or discard
       shape = [number of neighboring nodes of the target node, 1]
    3. Obtain the neighbor selection results of all nodes in the batch
       shape = list(batchsize)
    4. Aggregate information according to the neighbor node selection
       shape = [1, 32]
    5. All nodes' neighbor aggregation information
       shape = [batchsize, 32]
    """

    # step 1，2,3
    samp_neighs = []
    observations1 = []
    observations2 = []
    label_list = []
    rate = 0

    # statistics
    rate_b = 0
    rate_f = 0
    total_b = 0
    total_f = 0
    num_b = 0
    num_f = 0

    # the first node in neighs_list is the node itself, which needs to be removed
    for idx, neighs in enumerate(neighs_list):
        if len(neighs) == 1:
            samp_neighs.append(set(neighs))
            continue
        else:
            gen_feat = gen_feats[idx]  # guidance node

            gen_feat = gen_feat.expand(len(neighs) - 1, gen_feat.shape[0])
            if cuda:
                neighs_feature = features(torch.LongTensor(neighs[1:]).cuda())
            else:
                neighs_feature = features(torch.LongTensor(neighs[1:]))

            # get all states of the node and choose an action
            observation = torch.cat((neighs_feature, gen_feat), dim=1)
            # normalization
            observation = rl.policy.instancenorm(observation)

            label = labels[idx]
            # Here a sampling rate is set because in some relationships,
			# there are few neighbor nodes, 
			# so all of them are used for learning, 
			# while in other relationships there are too many neighbors, 
			# so only a subset can be selected for learning.
            if rl.sample_rate == 1:
                if label.item() == 1:
                    ll = np.ones(len(neighs) - 1, dtype=int)
                else:
                    ll = np.zeros(len(neighs) - 1, dtype=int)
                label_list.extend(ll)
                observations1.append(neighs_feature)
                observations2.append(gen_feat)
                action = rl.choose_action(observation, neighs)
            else:
                l = random.sample(
                    range(0, len(neighs) - 1),
                    int(max((len(neighs) - 1) * rl.sample_rate, 1)),
                )
                if label.item() == 1:
                    ll = np.ones(len(l), dtype=int)
                else:
                    ll = np.zeros(len(l), dtype=int)
                label_list.extend(ll)
                observations1.append(neighs_feature[l])
                observations2.append(gen_feat[l])
                action = rl.choose_action(observation, l)
            # At each action selection, the action policy is recorded.

            # The selected neighbors are added to a list.
            selected_neighs = []
            selected_neighs.append(neighs[0])
            for i in range(1, len(neighs)):
                if action[i - 1].item() == 1:
                    selected_neighs.append(neighs[i])
            samp_neighs.append(set(selected_neighs))
            rate += (len(selected_neighs) - 1) / (len(neighs) - 1)
            # Here the neighbor selection probability is calculated.
            if label.item() == 1:
                rate_f += (len(selected_neighs) - 1) / (len(neighs) - 1)
                total_f += len(neighs) - len(selected_neighs)
                num_f += 1
            else:
                rate_b += (len(selected_neighs) - 1) / (len(neighs) - 1)
                total_b += len(neighs) - len(selected_neighs)
                num_b += 1

    # Steps 4 and 5

    # Find the unique nodes in the neighbors
    unique_nodes_list = list(set.union(*samp_neighs))
    unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

    # Generate a mask: tensor[batchsize, number of unique nodes]
    mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
    column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
    row_indices = [
        i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))
    ]
    mask[row_indices, column_indices] = 1
    if cuda:
        mask = mask.cuda()

    # Calculate the average number of aggregated neighbors for each node
    num_neigh = mask.sum(1, keepdim=True)
    mask = mask.div(num_neigh)

    # Aggregate the selected neighbor features
    if cuda:
        embed_matrix = features(torch.LongTensor(unique_nodes_list).cuda())
    else:
        embed_matrix = features(
            torch.LongTensor(unique_nodes_list)
        )  # tensor[unique number of neighbor nodes, 32]

    agg_feats = mask.mm(embed_matrix)

    rl.rate = rate / len(observations1)
    rl.rate_f = rate_f / (num_f + 0.00000000001)
    rl.rate_b = rate_b / (num_b + 0.00000000001)
    observations1 = torch.cat(observations1)
    observations2 = torch.cat(observations2)
    """
    agg_feats: the aggregated features of the selected neighbor nodes
    label_list: to calculate the loss of VP
    """
    return (
        agg_feats,
        observations1,
        observations2,
        label_list,
        [rl.rate, rl.rate_f, rl.rate_b, total_f, total_b],
    )
