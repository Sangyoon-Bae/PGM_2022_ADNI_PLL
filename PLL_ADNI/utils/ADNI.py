import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from utils.utils_algo import generate_uniform_cv_candidate_labels

import argparse
import sys
sys.setrecursionlimit(300000)

import time
import copy
import os
import itertools

import numpy as np
import pandas as pd
import scipy.io

from sklearn.metrics import mean_absolute_error #여기도 바꿔야 함!!

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable

from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler



# 2-1 data loader
class GCNDataset(Dataset):
    def __init__(self, list_feature, list_adj, list_NIH_score):
        self.list_feature = list_feature
        self.list_adj = list_adj
        self.list_NIH_score = list_NIH_score

    def __len__(self):
        return len(self.list_feature)

    def __getitem__(self, index):
        return self.list_feature[index], self.list_adj[index], self.list_NIH_score[index]



def load_ADNI(partial_rate, batch_size):
    # 1-1 data load
    targ_folder_raw = '/home/juhyeon/Projects/PLL_ADNI/data/raw'  # 이 폴더가 있는 곳으로 경로 설정 해주세요
    node_feat_csv = pd.read_csv(os.path.join(targ_folder_raw, 'node-feat.csv.gz'), header=None)
    graph_label_csv = pd.read_csv(os.path.join(targ_folder_raw, 'graph-label.csv.gz'), header=None)
    conmat = scipy.io.loadmat('/home/juhyeon/Projects/PLL_ADNI/data/ADNI/adni_connectome_aparc_count.mat')  # 이 파일이 있는 곳으로 경로 설정 해주세요.
    list_adj = conmat['connectome_aparc0x2Baseg_count'].T  # (179, 84, 84)

    # 1-2 label scaling - pytorch label starts from 0.
    class2idx = {
        3: 2,
        2: 1,
        1: 0
    }

    idx2class = {v: k for k, v in class2idx.items()}

    graph_label_csv.replace(class2idx, inplace=True)  # 이제 0, 1, 2로 encoding 되어있음.

    # 1-3 feature scaling
    mm = MinMaxScaler()

    ### (1) node feat scaling + to numpy
    list_feature = mm.fit_transform(node_feat_csv).reshape(179, 84, 2).astype(np.float32)

    ### (2) graph label to numpy
    list_NIH_score = graph_label_csv.to_numpy()

    ### (3) edge feature threshold + scaling (이미 numpy였음)

    ##### (3-1) threshold as 20
    for i in range(list_adj.shape[0]):
        for j in range(list_adj.shape[1]):
            for k in range(list_adj.shape[2]):
                if list_adj[i][j][k] <= 20:
                    list_adj[i][j][k] = 0

    ##### (3-2) scaling with minmax scaler
    list_adj = mm.fit_transform(list_adj.reshape(list_adj.shape[0], -1)).reshape(list_adj.shape[0], 84, 84).astype(np.float32)

    test_size = 0.1
    val_size = 0.1

    num_total = len(list_feature) # num_total:179

    num_train = int(num_total * (1 - test_size)) # num_train: 179-17=162
    num_test = int(num_total * test_size) # num_test: 17


    feature_train = list_feature[:num_train]
    adj_train = list_adj[:num_train]
    NIH_score_train = list_NIH_score[:num_train]

    feature_test = list_feature[num_total - num_test:]
    adj_test = list_adj[num_total - num_test:]
    NIH_score_test = list_NIH_score[num_total - num_test:]

    # set test dataloader
    test_set = GCNDataset(feature_test, adj_test, NIH_score_test)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size*4, shuffle=False, num_workers=4,
                             sampler=torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False))

    # generate partial labels for train
    labels = torch.Tensor(NIH_score_train).long()
    partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)

    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())

    # augmentation
    partial_matrix_dataset = GCNAugmentedDataset(feature_train, adj_train, partialY.float(), labels.float())

    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    partial_matrix_train_loader = DataLoader(dataset=partial_matrix_dataset,
                                                              batch_size=batch_size,
                                                              shuffle=(train_sampler is None),
                                                              num_workers=4,
                                                              pin_memory=True,
                                                              sampler=train_sampler,
                                                              drop_last=True)
    return partial_matrix_train_loader, partialY, train_sampler, test_loader

class GCNAugmentedDataset(Dataset):
    def __init__(
        self,
        list_feature,
        list_adj,
        given_label_matrix,
        list_NIH_score,
        feat_mask_apply_prob: float = 0.5,  # 매 instance 마다 feature masking augmentation 적용할 확률
        feat_mask_prob: float = 0.1,  # feature masking 적용할 때 masking 할 노드의 percentage
        edge_perturb_apply_prob: float = 0.5,  # 매 instance 마다 edge perturbation augmentation 적용할 확률
        edge_perturb_prob: float = 0.1,  # edge perturbation 적용할 때 perturb 할 edge 의 percentage
        seed: int = 0,  # random seed for reproducibility
    ):
        self.list_feature = list_feature.astype(np.float32)
        self.list_adj = list_adj.astype(np.float32)

        self.given_label_matrix =given_label_matrix

        self.list_NIH_score = list_NIH_score
        assert all(0.0 <= p <= 1.0 for p in [feat_mask_apply_prob, feat_mask_prob, edge_perturb_apply_prob, edge_perturb_prob])
        self.feat_mask_apply_prob = feat_mask_apply_prob
        self.feat_mask_prob = feat_mask_prob
        self.edge_perturb_apply_prob = edge_perturb_apply_prob
        self.edge_perturb_prob = edge_perturb_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.list_feature)

    def __getitem__(self, index):
        orig_feat, orig_adj, orig_score = self.list_feature[index], self.list_adj[index], self.list_NIH_score[index]

        each_label = self.given_label_matrix[index]

        aug_feat = np.copy(orig_feat)
        aug_adj = np.copy(orig_adj)
        num_nodes = orig_feat.shape[0]
        if self.rng.random() < self.feat_mask_apply_prob:
            # apply feature masking
            num_mask = int(num_nodes * self.feat_mask_prob)
            mask_indices = self.rng.choice(num_nodes, num_mask)
            aug_feat[mask_indices] = 0.0
        if self.rng.random() < self.edge_perturb_apply_prob:
            # apply feature masking
            num_perturb = int(num_nodes * num_nodes * self.edge_perturb_prob)
            num_perturb = num_perturb // 2  # perturb half and apply symmetry
            x_indices = self.rng.choice(num_nodes, num_perturb, replace=True)  # allow duplicates
            y_indices = self.rng.choice(num_nodes, num_perturb, replace=True)  # allow duplicates
            perturbed_values = self.rng.random(num_perturb)
            aug_adj[x_indices, y_indices] = perturbed_values
            aug_adj[y_indices, x_indices] = perturbed_values
            np.fill_diagonal(aug_adj, 0)  # restore perturbed diagonals
        return orig_feat, orig_adj, aug_feat, aug_adj, each_label, orig_score, index

# orig_feat , orig_adj as weak transform


