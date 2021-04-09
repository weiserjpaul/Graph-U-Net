import os
import dgl
import h5py
import time
import numpy as np
import torch
import networkx as nx
from dgl.nn import GMMConv, GraphConv, ChebConv
from dgl.nn.pytorch import Sequential
import dgl.function as fn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.GDL_CC_support import create_log_file_name, apply_filter, add_additive_noise, pre_processing, order_data
from src.GDL_CC_support import mse_complex, mse_complex_summed, order_data_single_input
from src.support_methods import create_brute_force_FT_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class GraphUnet(torch.nn.Module):
    def __init__(self, ks, in_chan, out_chan = 40, hidden_feats = 20, depth = 0, drop_p = 0):
        super(GraphUnet, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.hidden_feats = hidden_feats
        self.depth = depth
        self.ks = ks
        self.drop_p = drop_p
        self.conv_down = torch.nn.ModuleList()
        self.conv_up = torch.nn.ModuleList()
        self.pool = torch.nn.ModuleList()
        self.unpool = torch.nn.ModuleList()


        ## Layer ##
        #self.conv_down.append(GraphConv(self.in_chan, np.power(2,1)*self.hidden_feats, bias=True))
        self.conv_down.append(GMMConv(self.in_chan, np.power(2,1)*self.hidden_feats, 2, 10, 'mean',bias=True))

        for i in range(self.depth):
            self.pool.append(Pool(self.ks[i], np.power(2,i+1)*self.hidden_feats, self.drop_p))
            #self.conv_down.append(GraphConv(np.power(2,i+1)*self.hidden_feats, np.power(2,i+2)*self.hidden_feats, bias=False))
            self.conv_down.append(GMMConv(np.power(2,i+1)*self.hidden_feats, np.power(2,i+2)*self.hidden_feats, 2, 10, 'mean', bias=False))

        for i in range(self.depth-1):
            self.unpool.append(Unpool())
            #self.conv_up.append(GraphConv((np.power(2,self.depth-i)+np.power(2,self.depth-i+1))*self.hidden_feats, np.power(2,self.depth-i)*self.hidden_feats, bias=False))
            self.conv_up.append(GMMConv((np.power(2,self.depth-i)+np.power(2,self.depth-i+1))*self.hidden_feats, np.power(2,self.depth-i)*self.hidden_feats, 2, 10, 'mean', bias=False))

        if self.depth == 0:
            #self.conv_up.append(GraphConv(np.power(2,1)*self.hidden_feats, self.out_chan, bias=True))
            self.conv_up.append(GMMConv(np.power(2,1)*self.hidden_feats, self.out_chan, 2, 10, 'mean', bias=True))
        else:
            self.unpool.append(Unpool())
            #self.conv_up.append(GraphConv((np.power(2,1)+np.power(2,1+1))*self.hidden_feats, self.out_chan, bias=True))
            self.conv_up.append(GMMConv((np.power(2,1)+np.power(2,1+1))*self.hidden_feats, self.out_chan, 2, 10, 'mean', bias=True))



    def forward(self, g, g_undersample, n_feat, pkor, pkor_undersample, b_undersample):
        skip_feat = []
        skip_graph = []
        skip_pkor = []
        skip_idx = []

        if b_undersample == True:
            #out = self.conv_down[0](g_undersample, n_feat)
            out = self.conv_down[0](g_undersample, n_feat, pkor_undersample)
        else:
            #out = self.conv_down[0](g, n_feat)
            out = self.conv_down[0](g, n_feat, pkor)
        out = torch.relu(out)

        for i in range(self.depth):
            skip_feat.append(out)
            skip_pkor.append(pkor)
            skip_graph.append(g)
            g, out, pkor, idx = self.pool[i](g, out, pkor)
            #out = self.conv_down[i + 1](g, out)
            out = self.conv_down[i + 1](g, out, pkor)
            out = torch.relu(out)
            skip_idx.append(idx)

        for i in range(self.depth-1):
            g, out = self.unpool[i](skip_graph[-i-1], out, skip_idx[-i-1])
            #out = self.conv_up[i](g, torch.cat((out, skip_feat[-i-1]), dim=1))
            out = self.conv_up[i](g, torch.cat((out, skip_feat[-i-1]), dim=1), skip_pkor[-i-1])
            out = torch.relu(out)

        if self.depth == 0:
            #out = self.conv_up[-1](skip_g[0], out)
            out = self.conv_up[-1](skip_g[0], out, skip_pkor[0])
            out = torch.tanh(out)
        else:
            g, out = self.unpool[-1](skip_graph[0], out, skip_idx[0])
            #out = self.conv_up[-1](g, torch.cat((out, skip_feat[0]), dim=1))
            out = self.conv_up[-1](g, torch.cat((out, skip_feat[0]), dim=1), skip_pkor[0])
            out = torch.tanh(out)

        return out




class Pool(torch.nn.Module):
    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = torch.nn.Sigmoid()
        self.proj = torch.nn.Linear(in_dim, 1)
        self.drop = torch.nn.Dropout(p=p) if p > 0 else torch.nn.Identity()


    def forward(self, g, h, pkor):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, pkor, self.k)


class Unpool(torch.nn.Module):
    def __init__(self):
        super(Unpool, self).__init__()

    def forward(self, g, h, idx):
        new_h = h.new_zeros([len(list(g.nodes())), h.shape[1]])
        new_h[idx] = h
        return g, new_h



def top_k_graph(scores, g, h, pkor, k):
    num_nodes = len(list(g.nodes())) #g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)), largest=True, sorted=False)
    idx = idx.to(device)
    new_h = h[idx,:]
    values = torch.unsqueeze(values, -1)

    new_h = torch.mul(new_h, values)
    #g = dgl.transform.khop_graph(g, 2)
    g = g.to(device)
    g = g.subgraph(idx)

    new_pkor = pkor[g.edata[dgl.EID]]

    return g, new_h, new_pkor, idx
