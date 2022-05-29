import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SkipConnection(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(SkipConnection, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):  # dimension이 다르면 dimension을 맞춰주는 작업
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
        return self.sigmoid(x1 + x2)


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
    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, n_node, bn=True, atn=True, num_head=1, sc='gsc',
                 dropout=0):
        super(GCNBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i == 0 else hidden_dim,
                                        out_dim if i == n_layer - 1 else hidden_dim,
                                        n_node,
                                        nn.ReLU() if i != n_layer - 1 else None,
                                        bn,
                                        atn,
                                        num_head,
                                        dropout))

        self.relu = nn.ReLU()
        if sc == 'gsc':
            self.sc = GatedSkipConnection(in_dim, out_dim)
        elif sc == 'sc':
            self.sc = SkipConnection(in_dim, out_dim)
        elif sc == 'no':
            self.sc = None
        else:
            assert False, "Wrong sc type."

    def forward(self, x, adj):
        residual = x
        for i, layer in enumerate(self.layers):
            out, adj = layer((x if i == 0 else out), adj)
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
        nn.init.xavier_uniform_(self.linear.weight)  # activation function 따라 다른 방식의 initialization을 쓸 수 있음.
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
    def __init__(self, n_layer=2, n_block=2, in_dim=2, hidden_dim=32, n_node=84, bn=True, sc='no', atn=False, num_head=8, dropout=0, pred_dim1=32, pred_dim2=32, pred_dim3=32, out_dim=3) :
        super(GCNNet, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(GCNBlock(n_layer,
                                        in_dim if i == 0 else hidden_dim,
                                        hidden_dim,
                                        hidden_dim,
                                        n_node,
                                        bn,
                                        atn,
                                        num_head,
                                        sc,
                                        dropout
                                        ))
        self.readout = ReadOut(hidden_dim,
                                   pred_dim1,
                                   act=nn.ReLU())
        self.pred1 = Predictor(pred_dim1,
                                   pred_dim2,
                                   act=nn.ReLU())
        self.pred2 = Predictor(pred_dim2,
                                   pred_dim3,
                                   act=nn.Tanh())
        self.pred3 = Predictor(pred_dim3,
                                   out_dim)
        self.final = nn.Softmax(dim=1)

        # contrastive head
        self.head = nn.Sequential(
            nn.Linear(pred_dim3, pred_dim3),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim3, pred_dim3))

    def forward(self, x, adj):
        for i, block in enumerate(self.blocks):
            out, adj = block((x if i == 0 else out), adj)
        feat = self.readout(out)
        a = self.pred1(feat)
        a = self.pred2(a)
        logits = self.pred3(a)
        # logits = self.final(a)
        feat_c = self.head(feat)

        return logits, F.normalize(feat_c, dim=1)   #Input: x, adj , output: logits(classifier), feature embeddings