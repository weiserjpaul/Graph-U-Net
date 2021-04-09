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



class UNet(torch.nn.Module):
    def __init__(self, edge, window_size_nodes, window_size_rings, window_shift_nodes,
                    window_shift_rings, pkor,
                    in_chan, out_chan = 40, hidden_feats = 20, depth = 0):

        super(UNet, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.hidden_feats = hidden_feats
        self.depth = depth
        self.layers = torch.nn.ModuleList()

        EDGES, INDICES, PKOORDS = self.initialize(edge = edge,
                                                    window_size_nodes = window_size_nodes,
                                                    window_size_rings = window_size_rings,
                                                    window_shift_nodes = window_shift_nodes,
                                                    window_shift_rings = window_shift_rings,
                                                    pkor = pkor)


        self.g = [dgl.graph((EDGES[0][0], EDGES[0][1])).to(device)]
        self.g_max = []
        self.pkor = [PKOORDS[0].to(device)]
        for i in range(depth):
            self.g_max.append(dgl.graph((EDGES[2*i+1][0], EDGES[2*i+1][1])).to(device))
            self.g.append(dgl.graph((EDGES[2*i+2][0], EDGES[2*i+2][1])).to(device))
            self.pkor.append(PKOORDS[i+1].to(device))



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




    def initialize(self, edge, window_size_nodes, window_size_rings, window_shift_nodes, window_shift_rings, pkor):

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


        #depth = depth
        window_size_nodes = window_size_nodes
        window_size_rings = window_size_rings
        window_shift_nodes = window_shift_nodes
        window_shift_rings = window_shift_rings

        edges_0 = edge[0,:]
        edges_1 = edge[1,:]
        pkor = pkor
        k_coord = k_coord

        n_edges = len(edges_0)
        n_rings = 16
        n_nodes_per_ring = 388


        EDGES = [[edges_0, edges_0]]
        INDICES = []
        PKOORDS = [pkor]


        #print(len(edges_0))
        #print(pkor.shape)
        #print(len(list(g1.nodes())))

        def index(ring, node, n_nodes_per_ring):
            if node < 0:
                node = n_nodes_per_ring + node
            return ring * n_nodes_per_ring + node



        for d in range(self.depth):

            ### Pooling Graph ###
            edges_pgraph_0 = []
            edges_pgraph_1 = []
            index_pgraph = []
            n_rings_new = 0


            for ii in range(n_rings):
                b2 = (ii >= window_size_rings-1) and ((ii-window_size_rings+1)%window_shift_rings == 0)
                #b2 = (ii%window_shift_rings == window_shift_rings-1)
                b4 = (ii == n_rings-1)
                if b2 or b4:
                    n_rings_new += 1
                for i in range(n_nodes_per_ring):
                    b1 = (i%window_shift_nodes == window_shift_nodes-1)
                    b3 = (i == n_nodes_per_ring-1)

                    if (b1 and b2) or (b3 and b2) or (b1 and b4) or (b3 and b4):
                        index_pgraph.append(1)
                        for jj in range(window_size_rings):
                            for j in range(window_size_nodes):
                                edges_pgraph_0.append(index(ii-jj, i-j, n_nodes_per_ring))
                                edges_pgraph_1.append(index(ii, i, n_nodes_per_ring))
                                edges_pgraph_0.append(index(ii, i, n_nodes_per_ring))
                                edges_pgraph_1.append(index(ii-jj, i-j, n_nodes_per_ring))
                    else:
                        index_pgraph.append(0)

            index_pgraph = np.array(index_pgraph)
            enum_pgraph = np.array([i for i, x in enumerate(index_pgraph) if x == 1])


            ### Convolution Graph ###
            edges_cgraph_0 = []
            edges_cgraph_1 = []
            pkor_conv = [[],[]]

            k0 = np.delete(k_coord[:,0]*index_pgraph, np.where(k_coord[:,0]*index_pgraph == 0))
            k1 = np.delete(k_coord[:,1]*index_pgraph, np.where(k_coord[:,1]*index_pgraph == 0))

            dist = np.zeros((len(k0), len(k0)))
            for i in range(len(k0)):
                dist[i] = np.power(np.power(k0-k0[i],2) + np.power(k1-k1[i],2), 1/2)

            radii = np.unique(np.round(np.power(np.power(k0,2) + np.power(k1,2), 1/2), 5))
            rad = radii[1] * 0.8
            dist[dist>rad] = 0


            for i in range(dist.shape[0]):
                for ii in range(dist.shape[1]):
                     if dist[i,ii] != 0:
                            edges_cgraph_0.append(i)
                            edges_cgraph_1.append(ii)
                            edges_cgraph_0.append(ii)
                            edges_cgraph_1.append(i)

                            pkor_conv[0].append(k0[ii]-k0[i])
                            pkor_conv[1].append(k1[ii]-k1[i])
                            pkor_conv[0].append(k0[i]-k0[ii])
                            pkor_conv[1].append(k1[i]-k1[ii])

            for i in range(dist.shape[0]):
                edges_cgraph_0.append(i)
                edges_cgraph_1.append(i)
                pkor_conv[0].append(0)
                pkor_conv[1].append(0)

            pkor_conv = torch.tensor(np.transpose(np.array(pkor_conv)), dtype=torch.float32)


            ### Re-Initialize ###

            #n_rings = np.int(n_rings / window_size_rings)
            n_rings = n_rings_new
            n_nodes_per_ring = np.int(len(enum_pgraph) / n_rings)
            k_coord = np.transpose(np.array([k0,k1]))

            EDGES.append([edges_pgraph_0, edges_pgraph_1])
            EDGES.append([edges_cgraph_0, edges_cgraph_1])
            INDICES.append(enum_pgraph)
            PKOORDS.append(pkor_conv)

            #print("k_coord shape: ", k_coord.shape)
            #print("n_rings: ", n_rings)
            #print("n_nodes_per_ring", n_nodes_per_ring)
            #print("pkor_conv: ", pkor_conv.shape)



        return EDGES, INDICES, PKOORDS
