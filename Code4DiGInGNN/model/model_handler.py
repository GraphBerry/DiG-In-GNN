import time, datetime
import os
import random
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from RL_model.RL_model import Reinforce
from utils.utils import (
    test_dig,
    test_sage,
    load_data,
    pos_neg_split,
    normalize,
    pick_step,
    getVISUAL,
    writeRow,
    writeHeader,
)
from model.model import DiGInLayer
from model.layers import InterAgg, IntraAgg
from model.graphsage import *
from model.neigh_gen_model import Gen


'''
model training
'''


class ModelHandler(object):
    def __init__(self, config):
        args = argparse.Namespace(**config)
        # load graph, feature, and label
        [homo, relation1, relation2, relation3], feat_data, labels = load_data(
            args.data_name, prefix=args.data_dir
        )

        # train_test split
        np.random.seed(args.seed)
        random.seed(args.seed)
        if args.data_name == "yelp":
            # index：0-45954
            index = list(range(len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(
                index,
                labels,
                stratify=labels,
                train_size=args.train_ratio,
                random_state=2,
                shuffle=True,
            )
            idx_valid, idx_test, y_valid, y_test = train_test_split(
                idx_rest,
                y_rest,
                stratify=y_rest,
                test_size=args.test_ratio,
                random_state=2,
                shuffle=True,
            )

        elif args.data_name == "amazon":  # amazon
            # 0-3304 are unlabeled nodes
            index = list(range(3305, len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(
                index,
                labels[3305:],
                stratify=labels[3305:],
                train_size=args.train_ratio,
                random_state=2,
                shuffle=True,
            )
            idx_valid, idx_test, y_valid, y_test = train_test_split(
                idx_rest,
                y_rest,
                stratify=y_rest,
                test_size=args.test_ratio,
                random_state=2,
                shuffle=True,
            )
            
        elif args.data_name == "tfinance":
            index = list(range(len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(
                index,
                labels,
                stratify=labels,
                train_size=args.train_ratio,
                random_state=2,
                shuffle=True,
            )
            idx_valid, idx_test, y_valid, y_test = train_test_split(
                idx_rest,
                y_rest,
                stratify=y_rest,
                test_size=args.test_ratio,
                random_state=2,
                shuffle=True,
            )
            

        print(
            f"Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},"
            + f"valid num {len(y_valid)}, test num {len(y_test)}, test positive num {np.sum(y_test)}"
        )
        print(f"Classification threshold: {args.thres}")
        print(f"Feature dimension: {feat_data.shape[1]}")

        train_pos, train_neg = pos_neg_split(idx_train, y_train)
        feat_data = normalize(feat_data)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_id)

        # set input graph
        # if args.model == "SAGE" or args.model == "GCN":
        #     adj_lists = homo
        # else:
        #     adj_lists = [relation1, relation2, relation3]
        adj_lists = [relation1, relation2, relation3]

        print(
            f"Model: {args.model}, multi-relation aggregator: {args.multi_relation}, emb_size: {args.emb_size}."
        )

        self.args = args
        self.dataset = {
            "feat_data": feat_data,
            "labels": labels,
            "adj_lists": adj_lists,
            "homo": homo,
            "idx_train": idx_train,
            "idx_valid": idx_valid,
            "idx_test": idx_test,
            "y_train": y_train,
            "y_valid": y_valid,
            "y_test": y_test,
            "train_pos": train_pos,
            "train_neg": train_neg,
        }

    def train(self):
        args = self.args
        feat_data, adj_lists = self.dataset["feat_data"], self.dataset["adj_lists"]
        idx_train, y_train = self.dataset["idx_train"], self.dataset["y_train"]
        idx_valid, y_valid, idx_test, y_test = (
            self.dataset["idx_valid"],
            self.dataset["y_valid"],
            self.dataset["idx_test"],
            self.dataset["y_test"],
        )
        # initialize model input
        features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
        features.weight = nn.Parameter(
            torch.FloatTensor(feat_data), requires_grad=False
        )

        neigh_gen1 = Gen(2 * feat_data.shape[1], 0.5, 5, feat_data.shape[1])
        neigh_gen2 = Gen(2 * feat_data.shape[1], 0.5, 5, feat_data.shape[1])
        neigh_gen3 = Gen(2 * feat_data.shape[1], 0.5, 5, feat_data.shape[1])
        neigh_gen = [neigh_gen1, neigh_gen2, neigh_gen3]

        # if args.laod_gen:
        #     gen_path = os.path.join(
        #         args.gen_save_dir, "{}_{}.pkl".format(args.gen_num, args.data_name)
        #     )
        #     checkpoint = torch.load(gen_path)
        #     neigh_gen1.load_state_dict(checkpoint["gen1"])
        #     neigh_gen2.load_state_dict(checkpoint["gen2"])
        #     neigh_gen3.load_state_dict(checkpoint["gen3"])

        if args.cuda:
            features.cuda()

        # build one-layer models
        if args.model == "DiG-In-GNN":
            # three RL model for the three relation types in the multi-relation dataset
            RL1 = Reinforce(
                args.rl_lr1,
                args.ob_dim,
                args.action_dim,
                args.fc1_dim,
                args.fc1_dim,
                args.sample_rate1,
                args.save_dir,
                args.rl_tolerance1,
                gamma=0.99,
            )

            RL2 = Reinforce(
                args.rl_lr2,
                args.ob_dim,
                args.action_dim,
                args.fc1_dim,
                args.fc1_dim,
                args.sample_rate2,
                args.save_dir,
                args.rl_tolerance2,
                gamma=0.99,
            )

            RL3 = Reinforce(
                args.rl_lr3,
                args.ob_dim,
                args.action_dim,
                args.fc1_dim,
                args.fc1_dim,
                args.sample_rate3,
                args.save_dir,
                args.rl_tolerance3,
                gamma=0.99,
            )
            
            if args.cuda:
                for i in [RL1, RL2, RL3]:
                    i.policy.cuda()
            RL_model = [RL1, RL2, RL3]

            # three intra-relation aggregator for the three relation types in the multi-relation dataset
            # a inter-relation aggregator to aggregate the three relation types
            intra1 = IntraAgg(
                features,
                feat_data.shape[1],
                args.emb_size,
                self.dataset["train_pos"],
                args.rho,
                neigh_gen1,
                RL1,
                1,
                cuda=args.cuda,
            )
            intra2 = IntraAgg(
                features,
                feat_data.shape[1],
                args.emb_size,
                self.dataset["train_pos"],
                args.rho,
                neigh_gen2,
                RL2,
                2,
                cuda=args.cuda,
            )
            intra3 = IntraAgg(
                features,
                feat_data.shape[1],
                args.emb_size,
                self.dataset["train_pos"],
                args.rho,
                neigh_gen3,
                RL3,
                3,
                cuda=args.cuda,
            )
            inter1 = InterAgg(
                features,
                feat_data.shape[1],
                args.emb_size,
                self.dataset["train_pos"],
                adj_lists,
                [intra1, intra2, intra3],
                inter=args.multi_relation,
                cuda=args.cuda,
            )

        elif args.model == "SAGE":
            agg_sage = MeanAggregator(features, cuda=args.cuda)
            enc_sage = Encoder(
                features,
                feat_data.shape[1],
                args.emb_size,
                adj_lists,
                agg_sage,
                gcn=False,
                cuda=args.cuda,
            )
        elif args.model == "GCN":
            agg_gcn = GCNAggregator(features, cuda=args.cuda)
            enc_gcn = GCNEncoder(
                features,
                feat_data.shape[1],
                args.emb_size,
                adj_lists,
                agg_gcn,
                gcn=True,
                cuda=args.cuda,
            )

        if args.model == "DiG-In-GNN":
            if args.cuda:
                gnn_model = DiGInLayer(2, inter1, args.alpha, neigh_gen, torch.device("cuda:0"))
            else:
                gnn_model = DiGInLayer(2, inter1, args.alpha, neigh_gen, torch.device("cpu"))

        if args.cuda:
            gnn_model.cuda()
            # neigh_gen.cuda()
            for i in range(3):
                neigh_gen[i].cuda()

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, gnn_model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.00001)

        for i in range(3):
            for name, i in gnn_model.nei_gen[i].named_parameters():
                i.requires_grad = True
        optimizer_gen = []
        for i in range(3):
            optimizer_gen.append(
                torch.optim.Adam(
                    filter(lambda p: p.requires_grad, neigh_gen[i].parameters()),
                    lr=args.gen_lr,
                    weight_decay=args.gen_weight_decay,
                )
            )

        # BCE
        if args.cuda:
            b_xent = nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=torch.tensor([args.negsamp_ratio])
            ).cuda()
        else:
            b_xent = nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=torch.tensor([args.negsamp_ratio])
            )

        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime(
            "%Y-%m-%d %H-%M-%S"
        )
        dir_saver = args.save_dir + timestamp
        path_saver = os.path.join(
            dir_saver, "{}_{}.pkl".format(args.data_name, args.model)
        )
        f1_mac_best, auc_best, ep_best = 0, 0, -1

        # 训练模型
        for epoch in range(args.num_epochs):
            if epoch >= args.gnn_update_start:
                gen_train_flag = True
            else:
                gen_train_flag = False

            # balance the unbalanced fraud dataset by picking probability
            sampled_idx_train = pick_step(
                idx_train,
                y_train,
                self.dataset["homo"],
                size=len(self.dataset["train_pos"]) * args.pick,
                gen_train_flag=gen_train_flag,
                all_labels=self.dataset["labels"],
                b_pick=self.args.b_pick,
                f_pick=self.args.f_pick
            )

            random.shuffle(sampled_idx_train)

            num_batches = int(len(sampled_idx_train) / args.batch_size) + 1

            epoch_time = 0

            gen_loss_avg = [0, 0, 0]

            losses=[]
            # mini-batch training
            for batch in range(num_batches):
                start_time = time.time()
                i_start = batch * args.batch_size
                i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
                batch_nodes = sampled_idx_train[i_start:i_end]
                if batch == num_batches - 1:
                    batch_nodes = sampled_idx_train[-args.batch_size:]
                batch_label = self.dataset["labels"][np.array(batch_nodes)]
                if args.cuda:
                    batch_label = torch.Tensor(batch_label).cuda()

                """
                # Context-level label, length needs to be multiplied by 2, batchsize->2*batchsize, because there is one positive sample and one negative sample
                This label is not the label from the dataset, but indicates whether the contrastive sample is a positive or negative sample, with 1 for positive and 0 for negative.
                """
                if args.cuda:
                    batch_label_contrast = torch.unsqueeze(
                        torch.cat(
                            (
                                torch.ones(len(batch_label)),
                                torch.zeros(len(batch_label) * args.negsamp_ratio),
                            )
                        ),
                        1,
                    ).cuda()
                    # labels = Variable(torch.cuda.LongTensor(batch_label))
                    labels = batch_label.long()
                else:
                    batch_label_contrast = torch.unsqueeze(
                        torch.cat(
                            (
                                torch.ones(len(batch_label)),
                                torch.zeros(len(batch_label) * args.negsamp_ratio),
                            )
                        ),
                        1,
                    )
                    labels = Variable(torch.LongTensor(batch_label))

                optimizer.zero_grad()

                rl_train_flag = False
                train_flag = False
                rl_has_trained = False
                if epoch >= args.gnn_update_start:
                    train_flag = True
                if epoch >= args.rl_update_start and epoch <= args.rl_update_end:
                    rl_train_flag = True
                if epoch >= args.rl_update_start and args.rl_update_start > 0:
                    rl_has_trained = True

                loss, ctx_, env_ = gnn_model.loss(
                    batch_nodes,
                    labels,
                    train_flag=train_flag,
                    rl_train_flag=rl_train_flag,
                    rl_has_trained=rl_has_trained,
                )

                if epoch <= args.gen_update_end:
                    if epoch >= args.gnn_update_start:
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()

                    totalgen = 0
                    res = []
                    i = 0
                    for c_l, e_l, op in zip(ctx_, env_, optimizer_gen):
                        op.zero_grad()
                        con = b_xent(c_l, batch_label_contrast)
                        if args.cuda:
                            l = torch.unsqueeze(
                                (torch.ones(len(batch_label)).cuda() - labels), 1
                            )
                        else:
                            l = torch.unsqueeze(
                                (torch.ones(len(batch_label)) - labels), 1
                            )
                        env = b_xent(e_l, l)
                        tmpl = (3 * torch.mean(con) + torch.mean(env)) / 4
                        # Because PyTorch will delete intermediate variables during backpropagation, 
                        # the intermediate variables are kept here.
                        if i < 2:
                            tmpl.backward(retain_graph=True)
                            i += 1
                        else:
                            tmpl.backward()
                        op.step()
                        totalgen = totalgen + tmpl / 3
                        res.append(tmpl)
                    for i in range(len(gen_loss_avg)):
                        gen_loss_avg[i] += res[i].item() / num_batches

                    print(
                        "multi-scale contrastive learning loss：",
                        totalgen.item(),
                        res[0].item(),
                        res[1].item(),
                        res[2].item(),
                    )
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(float(loss.item()))
                    losses.append(loss.item())
                    optimizer.step()
                    scheduler.step()

                end_time = time.time()
                epoch_time += end_time - start_time
                loss += loss.item()

            print(f"Epoch: {epoch}, avg_loss: {sum(losses)/(len(losses) + 0.0000001)}, time: {epoch_time}s")

            if (
                epoch >= 20
                and epoch <= 100
                and epoch % 10 == 0
                and args.gen_update_end > 20
                and args.save_gen
            ):
                state = {
                    "gen1": neigh_gen1.state_dict(),
                    "gen2": neigh_gen2.state_dict(),
                    "gen3": neigh_gen3.state_dict(),
                }
                gen_path = os.path.join(
                    args.gen_save_dir, "{}_{}_{}.pkl".format(epoch, args.data_name, 1)
                )
                if not os.path.exists(args.gen_save_dir):
                    os.makedirs(args.gen_save_dir)
                print("saving gen")
                torch.save(state, gen_path)

            if epoch % args.valid_epochs == 0 and epoch >= args.gnn_update_start:
                valid_start_time = time.time()
                if args.model == "SAGE" or args.model == "GCN":
                    continue
                    # print("Valid at epoch {}".format(epoch))
                    # f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_sage(
                    #     idx_valid,
                    #     y_valid,
                    #     gnn_model,
                    #     args.batch_size,
                    #     args.thres,
                    #     rl_has_trained,
                    # )
                    # if auc_val > auc_best:
                    #     f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
                    #     if not os.path.exists(dir_saver):
                    #         os.makedirs(dir_saver)
                    #     print("  Saving model ...")
                    #     torch.save(gnn_model.state_dict(), path_saver)
                else:
                    print("\nValid at epoch {}".format(epoch))
                    f1_mac_val, f1_1_val, f1_0_val, auc_val, gmean_val = test_dig(
                        idx_valid,
                        y_valid,
                        gnn_model,
                        args.batch_size,
                        args.thres,
                        rl_has_trained,
                    )
                    # writeRow(
                    #     metric_file_name,
                    #     ["AUC", epoch, "{:.4f}".format((auc_val - 0.01) * 100)],
                    #     "a",
                    # )
                    # writeRow(
                    #     metric_file_name,
                    #     ["F1-macro", epoch, "{:.4f}".format((f1_mac_val - 0.01) * 100)],
                    #     "a",
                    # )
                    # writeRow(
                    #     metric_file_name,
                    #     ["GMean", epoch, "{:.4f}".format((gmean_val - 0.01) * 100)],
                    #     "a",
                    # )
                    if f1_mac_val > f1_mac_best:
                        f1_mac_best, auc_best, ep_best = f1_mac_val, auc_val, epoch
                        if not os.path.exists(dir_saver):
                            os.makedirs(dir_saver)
                        print("  Saving model ...")
                        torch.save(gnn_model.state_dict(), path_saver)
                valid_end_time = time.time()
                print("valid_time_cost: {}".format(valid_end_time - valid_start_time))

        print("Restore model from epoch {}".format(ep_best))
        print("Model path: {}".format(path_saver))
        gnn_model.load_state_dict(torch.load(path_saver))
        f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = test_dig(
            idx_test, y_test, gnn_model, args.batch_size, args.thres, rl_has_trained
        )
        return f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test
