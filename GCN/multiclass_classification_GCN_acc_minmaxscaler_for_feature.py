import argparse
import sys
sys.setrecursionlimit(300000)

import time
import copy
import os
import itertools
import random

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

parser = argparse.ArgumentParser()
args = parser.parse_args("")
args.seed = 123
args.val_size = 0.1
args.test_size = 0.1
args.shuffle = True

################################################## STEP 1. data preparation ################################################
# 1-1 data load
targ_folder_raw='../ADNI-struct-count' #이 폴더가 있는 곳으로 경로 설정 해주세요
node_feat_csv = pd.read_csv(os.path.join(targ_folder_raw, 'node-feat.csv.gz'), header=None)
graph_label_csv = pd.read_csv(os.path.join(targ_folder_raw, 'graph-label.csv.gz'), header=None)
conmat = scipy.io.loadmat('../data/adni_connectome_aparc_count.mat') #이 파일이 있는 곳으로 경로 설정 해주세요.
list_adj = conmat['connectome_aparc0x2Baseg_count'].T # (179, 84, 84)


# 1-2 label scaling - pytorch label starts from 0.
class2idx = {
    3:2,
    2:1,
    1:0
}

idx2class = {v: k for k, v in class2idx.items()}

graph_label_csv.replace(class2idx, inplace=True) #이제 0, 1, 2로 encoding 되어있음.


# 1-3 feature scaling
mm = MinMaxScaler()

### (1) node feat scaling + to numpy
list_feature = mm.fit_transform(node_feat_csv).reshape(179,84,2)

### (2) graph label to numpy
list_NIH_score = graph_label_csv.to_numpy()

### (3) edge feature threshold + scaling (이미 numpy였음)

##### (3-1) threshold as 20
for i in range(list_adj.shape[0]):
    for j in range(list_adj.shape[1]):
        for k in range(list_adj.shape[2]):
            if list_adj[i][j][k] <=20:
                list_adj[i][j][k] = 0

##### (3-2) scaling with minmax scaler
list_adj = mm.fit_transform(list_adj.reshape(list_adj.shape[0],-1)).reshape(list_adj.shape[0], 84, 84)                


################################################## STEP 2. build model ################################################

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


class GCNAugmentedDataset(Dataset):
    def __init__(
        self,
        list_feature,
        list_adj,
        list_NIH_score,
        feat_mask_apply_prob: float = 0.5,  # 매 instance 마다 feature masking augmentation 적용할 확률
        feat_mask_prob: float = 0.1,  # feature masking 적용할 때 masking 할 노드의 percentage
        edge_perturb_apply_prob: float = 0.5,  # 매 instance 마다 edge perturbation augmentation 적용할 확률
        edge_perturb_prob: float = 0.1,  # edge perturbation 적용할 때 perturb 할 edge 의 percentage
        node_drop_apply_prob: float = 0.5,  # 매 instance 마다 node drop augmentation 적용할 확률
        node_drop_prob: float = 0.1,  # node drop 적용할 때 masking 할 노드의 percentage
        subgraph_apply_prob: float = 0.5,  # 매 instance 마다 subgraph augmentation 적용할 확률
        subgraph_prob: float = 0.1,  # subgraph 에서 제외될 노드의 percentage
        seed: int = 0,  # random seed for reproducibility
    ):
        self.list_feature = list_feature.astype(np.float32)
        self.list_adj = list_adj.astype(np.float32)
        self.list_NIH_score = list_NIH_score
        assert all(0.0 <= p <= 1.0
                   for p in [feat_mask_apply_prob, feat_mask_prob,
                             edge_perturb_apply_prob, edge_perturb_prob,
                             node_drop_apply_prob, node_drop_prob,
                             subgraph_apply_prob, subgraph_prob])
        self.feat_mask_apply_prob = feat_mask_apply_prob
        self.feat_mask_prob = feat_mask_prob
        self.edge_perturb_apply_prob = edge_perturb_apply_prob
        self.edge_perturb_prob = edge_perturb_prob
        self.node_drop_apply_prob = node_drop_apply_prob
        self.node_drop_prob = node_drop_prob
        self.subgraph_apply_prob = subgraph_apply_prob
        self.subgraph_prob = subgraph_prob
        self.rng = np.random.default_rng(seed)
        num_nodes = list_feature[0].shape[0]
        all_edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                all_edges.append((i, j))
        self.all_edges = np.array(all_edges)

    def __len__(self):
        return len(self.list_feature)

    def __getitem__(self, index):
        orig_feat, orig_adj, orig_score = self.list_feature[index], self.list_adj[index], self.list_NIH_score[index]

        aug_feat = np.copy(orig_feat)
        aug_adj = np.copy(orig_adj)
        num_nodes = orig_feat.shape[0]
        feature_dim = orig_feat.shape[1]

        if self.rng.random() < self.subgraph_apply_prob:
            num_subgraph_nodes = int(num_nodes * (1 - self.subgraph_prob))
            subgraph_nodes = []
            node_sample_prob = np.ones(num_nodes) / num_nodes

            while len(subgraph_nodes) < num_subgraph_nodes:
                center_node = self.rng.choice(num_nodes, 1, p=node_sample_prob)[0]
                connected_nodes = self.get_all_connected_nodes(index, center_node)
                subgraph_nodes.extend(connected_nodes)
                node_sample_prob[connected_nodes] = 0.0
                prob_sum = node_sample_prob.sum()
                if prob_sum > 0:
                    node_sample_prob = node_sample_prob / node_sample_prob.sum()
            subgraph_nodes = set(subgraph_nodes[:num_subgraph_nodes])
            deleted_nodes = np.array([n for n in range(num_nodes) if n not in subgraph_nodes])
            aug_feat[deleted_nodes] = 0.0
        if self.rng.random() < self.node_drop_apply_prob:
            num_drop_nodes = int(num_nodes * self.node_drop_prob)
            drop_nodes = self.rng.choice(num_nodes, num_drop_nodes, replace=False)
            if len(drop_nodes) > 0:
                aug_feat[drop_nodes] = 0.0
        if self.rng.random() < self.feat_mask_apply_prob:
            # apply feature masking
            num_mask = int(num_nodes * self.feat_mask_prob)
            mask_indices = self.rng.choice(num_nodes, num_mask, replace=False)
            if len(mask_indices) > 0:
                aug_feat[mask_indices] = self.rng.random((num_mask, feature_dim))  # 이젠 0 대신 noise로 mask
        if self.rng.random() < self.edge_perturb_apply_prob:
            # apply feature masking
            num_perturb = int(num_nodes * (num_nodes-1) * self.edge_perturb_prob * 0.5)
            xy_indices = self.all_edges[self.rng.choice(self.all_edges.shape[0], num_perturb, replace=False)]
            x_indices = xy_indices[:, 0]
            y_indices = xy_indices[:, 1]
            perturbed_values = self.rng.random(num_perturb)
            if len(x_indices) > 0:
                aug_adj[x_indices, y_indices] = perturbed_values
                aug_adj[y_indices, x_indices] = perturbed_values
        return orig_feat, orig_adj, aug_feat, aug_adj, orig_score

    def neighbors(self, index, node):
        return np.where(self.list_adj[index][node] > 0)[0]

    def get_all_connected_nodes(self, index, node):
        visited = set()
        self._get_all_connected_nodes(index, node, visited)
        return sorted(visited)  # sort for reproducibility

    def _get_all_connected_nodes(self, index, node, visited):
        visited.add(node)
        for neighbor in self.neighbors(index, node):
            if neighbor in visited:
                continue
            self._get_all_connected_nodes(index, neighbor, visited)


def partition(list_feature, list_adj, list_NIH_score, args):
    num_total = len(list_feature)
    num_train = int(num_total * (1 - args.test_size - args.val_size))
    num_val = int(num_total * args.val_size)
    num_test = int(num_total * args.test_size)
    
    feature_train = list_feature[:num_train]
    adj_train = list_adj[:num_train]
    NIH_score_train = list_NIH_score[:num_train]
    feature_val = list_feature[num_train:num_train + num_val]
    adj_val = list_adj[num_train:num_train + num_val]
    NIH_score_val = list_NIH_score[num_train:num_train + num_val]
    feature_test = list_feature[num_total - num_test:]
    adj_test = list_adj[num_total - num_test:]
    NIH_score_test = list_NIH_score[num_total - num_test:]
    
    train_set = GCNDataset(feature_train, adj_train, NIH_score_train)
    val_set = GCNDataset(feature_val, adj_val, NIH_score_val)
    test_set = GCNDataset(feature_test, adj_test, NIH_score_test)
    
    partition = {
        'train' : train_set,
        'val' : val_set,
        'test': test_set
    }
    
    return partition

# 2-2 module
class SkipConnection(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(SkipConnection, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):   # dimension이 다르면 dimension을 맞춰주는 작업
            in_x = self.linear(in_x)
        out = in_x + out_x
        return out
    
class GatedSkipConnection(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(GatedSkipConnection, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_coef_in = nn.Linear(out_dim, out_dim)
        self.linear_coef_out = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        z = self.gate_coefficient(in_x, out_x)
        out = torch.mul(z, out_x) + torch.mul(1.0 - z, in_x)
        return out
    
    def gate_coefficient(self, in_x, out_x):
        x1 = self.linear_coef_in(in_x)
        x2 = self.linear_coef_out(out_x)
        return self.sigmoid(x1+x2)
    
class Attention(nn.Module):
    
    def __init__(self, in_dim, output_dim, num_head):
        super(Attention, self).__init__()
        
        self.num_head = num_head
        self.atn_dim = output_dim // num_head
        
        self.linears = nn.ModuleList()
        self.corelations = nn.ParameterList()
        for i in range(self.num_head):
            self.linears.append(nn.Linear(in_dim, self.atn_dim))
            corelation = torch.FloatTensor(self.atn_dim, self.atn_dim)
            nn.init.xavier_uniform_(corelation)
            self.corelations.append(nn.Parameter(corelation))
            
        self.tanh = nn.Tanh()
        
    def forward(self, x, adj):
        heads = list()
        for i in range(self.num_head):
            x_transformed = self.linears[i](x)
            alpha = self.attention_matrix(x_transformed, self.corelations[i], adj)
            x_head = torch.matmul(alpha, x_transformed)
            heads.append(x_head)
        output = torch.cat(heads, dim=2)
        return output
    
    def attention_matrix(self, x_transformed, corelation, adj):
        x = torch.einsum('akj, ij->aki', (x_transformed, corelation))
        alpha = torch.matmul(x, torch.transpose(x_transformed, 1, 2))
        alpha = torch.mul(alpha, adj)
        alpha = self.tanh(alpha)
        return alpha
    
class GCNLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, n_node, act=None, bn=False, atn=False, num_head=1, dropout=0):
        super(GCNLayer, self).__init__()
        
        self.use_bn = bn
        self.use_atn = atn
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.bn = nn.BatchNorm1d(n_node)
        self.attention = Attention(out_dim, out_dim, num_head)
        self.activation = act
        self.dropout_rate = dropout
        self.dropout = nn.Dropout2d(self.dropout_rate)
        
    def forward(self, x, adj):
        out = self.linear(x)
        if self.use_atn:
            out = self.attention(out, adj)
        else:
            out = torch.matmul(adj, out)
        if self.use_bn:
            out = self.bn(out)
        if self.activation != None:
            out = self.activation(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        return out, adj
    
class GCNBlock(nn.Module):
    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, n_node, bn=True, atn=True, num_head=1, sc='gsc', dropout=0):
        super(GCNBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                       out_dim if i==n_layer-1 else hidden_dim,
                                       n_node,
                                       nn.ReLU() if i != n_layer-1 else None,
                                       bn,
                                       atn,
                                       num_head,
                                       dropout))
        
        self.relu = nn.ReLU()
        if sc=='gsc':
            self.sc = GatedSkipConnection(in_dim, out_dim)
        elif sc =='sc':
            self.sc = SkipConnection(in_dim, out_dim)
        elif sc=='no':
            self.sc = None
        else:
            assert False, "Wrong sc type."
    
    def forward(self, x, adj):
        residual = x
        for i, layer in enumerate(self.layers):
            out, adj = layer((x if i==0 else out), adj)
        if self.sc != None:
            out = self.sc(residual, out)
        out = self.relu(out)
        return out, adj
    
class ReadOut(nn.Module):
    
    def __init__(self, in_dim, out_dim, act=None):
        super(ReadOut, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear = nn.Linear(self.in_dim,
                               self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)   # activation function 따라 다른 방식의 initialization을 쓸 수 있음.
        self.activation = act
        
    def forward(self, x):
        out = self.linear(x)
        out = torch.sum(out, 1)
        if self.activation != None:
            out = self.activation(out)
        return out
    
class Predictor(nn.Module):
    
    def __init__(self, in_dim, out_dim, act=None):
        super(Predictor, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear = nn.Linear(self.in_dim,
                               self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act
        
    def forward(self, x):
        out = self.linear(x)
        if self.activation != None:
            out = self.activation(out)
        return out
    

# 2-3 Model    
class GCNNet(nn.Module):
    def __init__(self, args):
        super(GCNNet, self).__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(args.n_block):
            self.blocks.append(GCNBlock(args.n_layer,
                                      args.in_dim if i==0 else args.hidden_dim,
                                       args.hidden_dim,
                                       args.hidden_dim,
                                       args.n_node,
                                        args.bn,
                                        args.atn,
                                        args.num_head,
                                        args.sc,
                                        args.dropout
                                       ))
            self.readout = ReadOut(args.hidden_dim,
                                  args.pred_dim1,
                                   act=nn.ReLU())
            self.pred1 = Predictor(args.pred_dim1,
                                  args.pred_dim2,
                                  act=nn.ReLU())
            self.pred2 = Predictor(args.pred_dim2,
                                  args.pred_dim3,
                                  act=nn.Tanh())
            self.pred3 = Predictor(args.pred_dim3,
                                  args.out_dim)
            
    def forward(self, x, adj):
        for i, block in enumerate(self.blocks):
            out, adj = block((x if i==0 else out), adj)
        out = self.readout(out)
        out = self.pred1(out)
        out = self.pred2(out)
        out = self.pred3(out)
        
        return out #3개짜리 내보냄

    
################################################## STEP 3. utils for train, valid, test ################################################
    
def multi_acc(y_pred, y_test):
    y_pred = torch.Tensor(y_pred)
    y_test = torch.Tensor(y_test) 
    #print('y_test:', y_test) #0, 1, 2로 되어있음!
    y_pred_softmax = torch.log_softmax(y_pred, dim=0)
    #print('y_pred_softmax:', y_pred_softmax)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=0)
    #print('y_pred_tags:', y_pred_tags) # 여기서 0, 1, 2가 나와야 함..ㅎㅎ..
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def train(model, device, optimizer, criterion, data_train, bar, args):
    epoch_train_loss = 0
    for i, batch in enumerate(data_train):
        list_feature = batch[0].clone().to(device).float().detach().requires_grad_(True)
        list_adj = batch[1].clone().to(device).float().detach().requires_grad_(True)
        #원형은... torch.tensor(batch[1]).to(device).float()
        list_NIH_score = batch[2].clone().to(device).float().detach().requires_grad_(True)
        list_NIH_score = list_NIH_score.view(-1, 1)
        list_NIH_score = list_NIH_score.type(torch.long)
        #print('shape of list_NIH_score is:', list_NIH_score.shape)
        #print('list_NIH_score is:', list_NIH_score)
        
        model.train()
        optimizer.zero_grad()
        list_pred_NIH_score = model(list_feature, list_adj)
        #print('shape of list_pred_NIH_score is:', list_pred_NIH_score.shape)

        list_pred_NIH_score.require_grad = False
        train_loss = criterion(list_pred_NIH_score, list_NIH_score.squeeze(dim=-1))
        #print('train loss is:', train_loss)
        epoch_train_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
        
        bar.update(len(list_feature))
        
    epoch_train_loss /= len(data_train)
    
    return model, epoch_train_loss

def validate(model, device, criterion, data_val, bar, args):
    epoch_val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_val):
            list_feature = batch[0].clone().to(device).float().detach().requires_grad_(True)
            list_adj = batch[1].clone().to(device).float().detach().requires_grad_(True)
            #원형은... torch.tensor(batch[1]).to(device).float()
            list_NIH_score = batch[2].clone().to(device).float().detach().requires_grad_(True)
            list_NIH_score = list_NIH_score.view(-1,1)
            list_NIH_score = list_NIH_score.type(torch.long)

            model.eval()
            list_pred_NIH_score = model(list_feature, list_adj)
            list_pred_NIH_score.require_grad = False
            val_loss = criterion(list_pred_NIH_score, list_NIH_score.squeeze(dim=-1))
            epoch_val_loss += val_loss.item()
            
            bar.update(len(list_feature))

    epoch_val_loss /= len(data_val)
    
    return model, epoch_val_loss

def test(model, device, data_test, args):
    model.eval()
    with torch.no_grad():
        NIH_score_total = list()
        pred_NIH_score_total = list()
        for i, batch in enumerate(data_test):
            list_feature = batch[0].clone().to(device).float().detach().requires_grad_(True)
            list_adj = batch[1].clone().to(device).float().detach().requires_grad_(True)
            #원형은... torch.tensor(batch[1]).to(device).float()
            list_NIH_score = batch[2].clone().to(device).float().detach().requires_grad_(True)
            NIH_score_total += list_NIH_score.tolist()
            list_NIH_score = list_NIH_score.view(-1,1)
            list_NIH_score = list_NIH_score.type(torch.long)

            list_pred_NIH_score = model(list_feature, list_adj)
            pred_NIH_score_total += list_pred_NIH_score.view(-1).tolist()

        #print('true label: ', NIH_score_total)
        #print('pred label: ', pred_NIH_score_total)
        acc = multi_acc(pred_NIH_score_total, NIH_score_total) ## 여기를 어떻게 바꿀까?
        #std = np.std(np.array(NIH_score_total)-np.array(pred_NIH_score_total))

    return acc, NIH_score_total, pred_NIH_score_total

def experiment(dict_partition, device, bar, args):
    time_start = time.time()
    
    model = GCNNet(args)
    model.to(device)

        
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.l2_coef)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.l2_coef)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.l2_coef)
    else:
        assert False, 'Undefined Optimizer Type'
        
    criterion = nn.CrossEntropyLoss().to(device)
    #criterion = nn.L1Loss(reduction = 'sum')
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)

    list_train_loss = list()
    list_val_loss = list()

    data_train = DataLoader(dict_partition['train'], 
                            batch_size=args.batch_size,
                            shuffle=args.shuffle)

    data_val = DataLoader(dict_partition['val'],
                          batch_size=args.batch_size,
                          shuffle=args.shuffle)

    for epoch in range(args.epoch):

        #scheduler.step() - 순서 변경! optimizer 다음에 오도록.
        model, train_loss = train(model, device, optimizer, criterion, data_train, bar, args)
        scheduler.step()
        list_train_loss.append(train_loss)
        model, val_loss = validate(model, device, criterion, data_val, bar, args)
        list_val_loss.append(val_loss)

    data_test = DataLoader(dict_partition['test'],
                           batch_size=args.batch_size,
                           shuffle=args.shuffle)

    acc, NIH_score_total, pred_NIH_score_total = test(model, device, data_test, args) #원래 acc 가 앞에 있었음.
        
    time_end = time.time()
    time_required = time_end - time_start
    
    args.list_train_loss = list_train_loss
    args.list_val_loss = list_val_loss
    args.NIH_score_total = NIH_score_total
    args.pred_NIH_score_total = pred_NIH_score_total
    args.acc = acc
    args.time_required = time_required
    
    return args

dict_partition = partition(list_feature, list_adj, list_NIH_score, args)

################################################## STEP 4. hyperparameter tuning ################################################

# 4-1. hyperparameter list
args.batch_size = 128
args.lr = 0.001
args.l2_coef = 0
args.optim = 'Adam'
args.epoch = 30
args.n_block = 2
args.n_layer = 2
args.n_node = 84
args.in_dim = 2 #number of node feature
args.hidden_dim = 32
args.pred_dim1 = 32
args.pred_dim2 = 32
args.pred_dim3 = 32
args.out_dim = 3 #이거 맞아..
args.bn = True
args.sc = 'no'
args.atn = False
args.step_size = 10
args.gamma = 0.1
args.dropout = 0
args.num_head = 8

# 4-2. set device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') #여기 바꿔가면서 실험해야 함

# 4-3. hyperparameter space
list_lr = [0.01, 0.001] #, 0.0001, 0.00001]
list_n_block = [1, 2] #, 3]
list_n_layer = [1, 2] #, 3]
list_bn = [False, True]
list_sc = ['no', 'sc', 'gsc']
list_atn = [False, True]
list_hidden_dim = [32, 64] #, 256, 512]
list_num_head = [2, 4] #, 8] #, 16]
list_pred_dim1 = [32, 64] #, 256]
list_pred_dim2 = [32, 64] #, 256]
list_pred_dim3 = [32, 64] #, 256]

var1 = "lr"
var2 = "n_block"
var3 = "n_layer"
var4 = "bn"
var5 = "sc"
var6 = "atn"
var7 = "hidden_dim"
var8 = "num_head"
var9 = "pred_dim1"
var10 = "pred_dim2"
var11 = "pred_dim3"

all_ = [list_lr, list_n_block, list_n_layer, list_bn, list_sc, list_atn, list_hidden_dim, list_num_head,
        list_pred_dim1, list_pred_dim2, list_pred_dim3] 
h_space = [s for s in itertools.product(*all_)]

# 4-4. run!
dict_result = dict()
n_iter = 3072*args.epoch*(len(dict_partition['train'])+len(dict_partition['val']))
bar = tqdm(total=n_iter, file=sys.stdout, position=0)

for hy in h_space:
    args.lr = hy[0]
    args.n_block = hy[1]
    args.n_layer = hy[2]
    args.bn = hy[3]
    args.sc = hy[4]
    args.atn = hy[5]
    args.hidden_dim = hy[6]
    args.num_head = hy[7]
    args.pred_dim1 = hy[8]
    args.pred_dim2 = hy[9]
    args.pred_dim3 = hy[10]
    exp_name = var1+':'+str(hy[0])+'/'+var2+':'+str(hy[1])+'/'+var3+':'+str(hy[2])+'/'+var4+':'+str(hy[3])+'/'+var5+':'+str(hy[4])+'/'+var6+':'+str(hy[5])+'/'+var7+':'+str(hy[6])+'/'+var8+':'+str(hy[7])+'/'+var9+':'+str(hy[8])+'/'+var10+':'+str(hy[9])+'/'+var11+':'+str(hy[10])
    args.exp_name = exp_name
    result = vars(experiment(dict_partition, device, bar, args))
    print(args.exp_name + " took " + str(int(args.time_required)) + "seconds.")
    dict_result[args.exp_name] = copy.deepcopy(result)
        
    torch.cuda.empty_cache()
bar.close()

# 4-5 save exeperiment result
df_result = pd.DataFrame(dict_result).transpose()
df_result.to_csv('./GCN_result/GCN_hyp_tuning_ADNI_minmax_scaled_label+feature.csv')