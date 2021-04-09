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

#from plotly.subplots import make_subplots
#import plotly.graph_objects as go







class MaxPool(torch.nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def forward(self, g, h):
        g.ndata['h'] = h
        g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.max('m', 'h_max'))
        h_max = g.ndata['h_max']
        return h_max[self.pool]

class UpSample(torch.nn.Module):
    def __init__(self, upsample):
        super().__init__()
        self.upsample = upsample

    def forward(self, g, h):
        h_up = torch.zeros(len(g.nodes()),h.shape[1]).to(device)
        h_up[self.upsample] = h
        g.ndata['h'] = h_up
        g.update_all(message_func=fn.copy_u('h', 'm'),  reduce_func=fn.max('m', 'h_out'))
        h_out = g.ndata['h_out']
        return h_out



class UNet_old(torch.nn.Module):
    def __init__(self, in_chan, out_chan = 40, hidden_feats = 20, depth = 0):
        super(UNet1, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.hidden_feats = hidden_feats
        self.depth = depth
        self.layers = torch.nn.ModuleList()

        EDGES, INDICES, PKOORDS = self.initialize()

        self.pkor0 = PKOORDS[0].to(device)
        self.pkor1 = PKOORDS[1].to(device)
        self.pkor2 = PKOORDS[2].to(device)
        self.pkor = [self.pkor0, self.pkor1, self.pkor2]

        ## Graphs ##
        self.g_max1 = dgl.graph((EDGES[0][0], EDGES[0][1])).to(device)
        self.g_max2 = dgl.graph((EDGES[1][0], EDGES[1][1])).to(device)
        self.g1 = dgl.graph((EDGES[2][0], EDGES[2][1])).to(device)
        self.g2 = dgl.graph((EDGES[3][0], EDGES[3][1])).to(device)
        self.g3 = dgl.graph((EDGES[4][0], EDGES[4][1])).to(device)
        self.g_max = [self.g_max1, self.g_max2]
        self.g = [self.g1, self.g2, self.g3]

        ## Layer ##
        self.layers.append(GMMConv(self.in_chan, np.power(2,1)*self.hidden_feats, 2, 10, 'mean', bias=True))

        for i in range(self.depth):
            self.layers.append(MaxPool(INDICES[i]))
            self.layers.append(GMMConv(np.power(2,i+1)*self.hidden_feats, np.power(2,i+2)*self.hidden_feats, 2, 10, 'mean', bias=False))

        for i in range(self.depth-1):
            self.layers.append(UpSample(INDICES[-i-1]))
            self.layers.append(GMMConv((np.power(2,self.depth-i)+np.power(2,self.depth-i+1))*self.hidden_feats, np.power(2,self.depth-i)*self.hidden_feats, 2, 10, 'mean', bias=False))

        if self.depth == 0:
            self.layers.append(GMMConv(np.power(2,1)*self.hidden_feats, self.out_chan, 2, 10, 'mean', bias=True))
        else:
            self.layers.append(UpSample(INDICES[0]))
            self.layers.append(GMMConv((np.power(2,1)+np.power(2,1+1))*self.hidden_feats, self.out_chan, 2, 10, 'mean', bias=True))



    def forward(self, g, g_undersample, n_feat, p_coord, p_coord_undersample, b_undersample):
        skip = []

        if b_undersample == True:
            out = self.layers[0](g_undersample, n_feat, p_coord_undersample)
        else:
            out = self.layers[0](self.g[0], n_feat, self.pkor[0])
        out = torch.relu(out)

        for i in range(self.depth):
            skip.append(out)
            out = self.layers[2*i + 1](self.g_max[i], out)
            out = self.layers[2*i + 2](self.g[i+1], out, self.pkor[i+1])
            out = torch.relu(out)

        for i in range(self.depth-1):
            out = self.layers[2*i + 2*self.depth + 1](self.g_max[-i-1], out)
            out = self.layers[2*i + 2*self.depth + 2](self.g[-i-2], torch.cat((out, skip[-i-1]),dim=1), self.pkor[-i-2])
            out = torch.relu(out)

        if self.depth == 0:
            out = self.layers[-1](self.g[0], out, self.pkor[0])
            out = torch.tanh(out)
        else:
            out = self.layers[-2](self.g_max[0], out)
            out = self.layers[-1](self.g[0], torch.cat((out, skip[0]),dim=1), self.pkor[0])
            out = torch.tanh(out)

        return out




    def initialize(self):
        fh002 = h5py.File("../data/Vol_002/Vol_002_Case10_16.h5", "r")
        #fh002 = h5py.File("Z:/data/Vol_002/Vol_002_Case10_16.h5", "r")
        kx_s = fh002["kx"]
        ky_s = fh002["ky"]
        kx_s = np.transpose(kx_s,[1,0])
        ky_s = np.transpose(ky_s,[1,0])
        n_circ = kx_s.shape[1]

        kx_s = np.pi * (kx_s / (np.max(kx_s) / (n_circ-0.5/n_circ) ))
        ky_s = np.pi * (ky_s / (np.max(ky_s) / (n_circ-0.5/n_circ) ))
        W, k_coord = create_brute_force_FT_matrix([32,32], kx_s, ky_s)


        ### Graph 1 ###
        path_to_data = "../data/"
        name_dataset = 'DATA_CRT_32_sim_ver6.h5'
        f_handle = h5py.File(path_to_data + name_dataset,'r')

        edge_attr_raw = torch.tensor(np.array(f_handle['edge_attr']), dtype=torch.float32)
        edge_index_raw = torch.tensor(np.array(f_handle['edge_index']) - 1, dtype=torch.long)

        g1 = dgl.graph((edge_index_raw[0,:], edge_index_raw[1,:])).to(device)
        g1.edata['w'] = edge_attr_raw.to(device)

        pkor = torch.tensor(np.array(f_handle['pkor_vec']), dtype=torch.float32)


        ### Graph MAX 1 ###
        edges_sg1 = [[],[]]
        pool1 = []
        m1 = 4

        def index(ring, node):
            return ring * 388 + node


        for ii in range(16):
            for i in range(388):
                if i%m1 == m1-1 and ii%2 == 1:
                    pool1.append(1)
                    for jj in range(2):
                        for j in range(m1):
                            edges_sg1[0].append(index(ii-jj, i-j))
                            edges_sg1[1].append(index(ii,i))
                            edges_sg1[0].append(index(ii, i))
                            edges_sg1[1].append(index(ii-jj,i-j))
                else:
                    pool1.append(0)

        pool1 = np.array(pool1)
        pool11 = np.array([i for i, x in enumerate(pool1) if x == 1])

        g_max1 = dgl.graph((edges_sg1[0], edges_sg1[1]))


        ### Graph 2 ###
        k0 = np.delete(k_coord[:,0]*pool1, np.where(k_coord[:,0]*pool1 == 0))
        k1 = np.delete(k_coord[:,1]*pool1, np.where(k_coord[:,1]*pool1 == 0))

        dist = np.zeros((len(k0), len(k0)))
        for i in range(len(k0)):
            dist[i] = np.power(np.power(k0-k0[i],2) + np.power(k1-k1[i],2), 1/2)

        radii = np.unique(np.round(np.power(np.power(k0,2) + np.power(k1,2), 1/2), 5))
        rad = radii[1] * 0.8
        dist[dist>rad] = 0

        edges_g2 = [[],[]]
        pkor2 = [[],[]]
        for i in range(dist.shape[0]):
            for ii in range(dist.shape[1]):
                 if dist[i,ii] != 0:
                        edges_g2[0].append(i)
                        edges_g2[1].append(ii)
                        edges_g2[0].append(ii)
                        edges_g2[1].append(i)

                        pkor2[0].append(k0[ii]-k0[i])
                        pkor2[1].append(k1[ii]-k1[i])
                        pkor2[0].append(k0[i]-k0[ii])
                        pkor2[1].append(k1[i]-k1[ii])

        for i in range(dist.shape[0]):
            edges_g2[0].append(i)
            edges_g2[1].append(i)
            pkor2[0].append(0)
            pkor2[1].append(0)

        pkor2 = torch.tensor(np.transpose(np.array(pkor2)), dtype=torch.float32)

        ### Graph MAX 2 ###
        edges_sg2 = [[],[]]
        pool2 = []
        m1 = 4
        m2 = 4

        for ii in range(16):
            for i in range(388):
                if i%(m1*m2) == m1*m2-1 and ii%(2*2) == 2*2-1:
                    pool2.append(1)
                    for jj in range(2):
                        for j in range(m2):
                            edges_sg2[0].append(index(ii-(jj*2), i-(j*m1)))
                            edges_sg2[1].append(index(ii, i))
                            edges_sg2[0].append(index(ii, i))
                            edges_sg2[1].append(index(ii-(jj*2), i-(j*m1)))
                else:
                    pool2.append(0)

                if i == 387 and ii%(2*2) == 2*2-1:
                    for jj in range(2):
                        edges_sg2[0].append(index(ii-jj*2, i))
                        edges_sg2[1].append(index(ii-jj*2, i))

        pool2 = np.array(pool2)
        pool22 = [i for i, x in enumerate(pool2) if x == 1]

        pool222 = []
        c = -1
        for i in range(len(pool1)):
            if pool1[i] == 1:
                c += 1
                if pool2[i] == 1:
                    pool222.append(c)

        for i in range(len(edges_sg2[0])):
            val0 = edges_sg2[0][i]
            val1 = edges_sg2[1][i]
            edges_sg2[0][i] = np.where(pool11 == val0)[0][0]
            edges_sg2[1][i] = np.where(pool11 == val1)[0][0]

        g_max2 = dgl.graph((edges_sg2[0], edges_sg2[1]))

        ### Graph 3 ###
        k0 = np.delete(k_coord[:,0]*pool2, np.where(k_coord[:,0]*pool2 == 0))
        k1 = np.delete(k_coord[:,1]*pool2, np.where(k_coord[:,1]*pool2 == 0))


        dist = np.zeros((len(k0), len(k0)))
        for i in range(len(k0)):
            dist[i] = np.power(np.power(k0-k0[i],2) + np.power(k1-k1[i],2), 1/2)

        radii = np.unique(np.round(np.power(np.power(k0,2) + np.power(k1,2), 1/2), 5))
        rad = radii[1] * 0.8
        dist[dist>rad] = 0

        edges_g3 = [[],[]]
        pkor3 = [[],[]]
        for i in range(dist.shape[0]):
            for ii in range(dist.shape[1]):
                 if dist[i,ii] != 0:
                        edges_g3[0].append(i)
                        edges_g3[1].append(ii)
                        edges_g3[0].append(ii)
                        edges_g3[1].append(i)

                        pkor3[0].append(k0[ii]-k0[i])
                        pkor3[1].append(k1[ii]-k1[i])
                        pkor3[0].append(k0[i]-k0[ii])
                        pkor3[1].append(k1[i]-k1[ii])

        for i in range(dist.shape[0]):
            edges_g3[0].append(i)
            edges_g3[1].append(i)
            pkor3[0].append(0)
            pkor3[1].append(0)

        pkor3 = torch.tensor(np.transpose(np.array(pkor3)), dtype=torch.float32)


        EDGES = [edges_sg1, edges_sg2,
                [edge_index_raw[0,:], edge_index_raw[1,:]],
                edges_g2 ,edges_g3]
        INDICES = [pool11, pool222]
        PKOORDS =  [pkor, pkor2, pkor3]

        return EDGES, INDICES, PKOORDS
