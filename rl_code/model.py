# 构建一个2层的GNN模型
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, seed, conv_type='sage'):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        if conv_type == 'sage':
            self.conv1 = dglnn.SAGEConv(
                in_feats=in_feats, out_feats=hid_feats, aggregator_type='lstm')
            self.conv2 = dglnn.SAGEConv(
                in_feats=hid_feats, out_feats=out_feats, aggregator_type='lstm')
        elif conv_type == 'gcn':
            self.conv1 = dglnn.GraphConv(in_feats, hid_feats)
            self.conv2 = dglnn.GraphConv(hid_feats, out_feats)

        self.linear = nn.Linear(hid_feats, out_feats)

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        # h = F.relu(h)
        # h = self.linear(h)
        # h = F.softmax(h, dim=1)
        return h
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, use_bn, use_reset):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): size of hidden_layers
            use_bn (bool): use batch norm or not.
            use_reset (bool): weights initialization used in original paper
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        if use_bn:
            self.bn0 = nn.BatchNorm1d(state_size)
            self.bn1 = nn.BatchNorm1d(hidden_layers[0])

        use_bias = not use_bn
        self.fcs1 = nn.Linear(state_size, hidden_layers[0], bias=use_bias)
        self.fcs2 = nn.Linear(hidden_layers[0]+action_size, hidden_layers[1])
        self.fcs3 = nn.Linear(hidden_layers[1], 1)

        self.use_bn = use_bn
        
        if use_reset:
            self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fcs2.weight.data.uniform_(*hidden_init(self.fcs2))
        self.fcs3.weight.data.uniform_(-3e-3, 3e-3)
        self.fcs3.bias.data.uniform_(-3e-3, 3e-3)        

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.use_bn:
            state = self.bn0(state)
            xs = F.relu(self.bn1(self.fcs1(state)))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fcs2(x))
        else:
            xs = F.relu(self.fcs1(state))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fcs2(x))

        return self.fcs3(x)