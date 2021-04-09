import os
import dgl
import h5py
import time
import numpy as np
import torch
import networkx as nx
from dgl.nn import GMMConv
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.GDL_CC_support import create_log_file_name, apply_filter, add_additive_noise, pre_processing, order_data
from src.GDL_CC_support import mse_complex, mse_complex_summed, order_data_single_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#from plotly.subplots import make_subplots
#import plotly.graph_objects as go



def train(data_train_x, data_train_y, model, optimizer, g, g_undersample,
            pkor, pkor_undersample, b_undersample,
            mini_batch=10, noise_lvl=0):
    model.train()
    total_loss = 0

    mini_batch_idx = 0
    mini_batch_loss = 0

    for i in np.arange(data_train_x.shape[0]):
        mini_batch_idx += 1
        inn = data_train_x[i, ].to(device)

        optimizer.zero_grad()
        pred = model(g, g_undersample, inn, pkor, pkor_undersample, b_undersample)
        loss = F.mse_loss(pred, data_train_y[i, ].to(device))

        mini_batch_loss += loss

        if mini_batch_idx == mini_batch:
            mini_batch_loss = mini_batch_loss / mini_batch
            mini_batch_loss.backward()
            optimizer.step()
            mini_batch_idx = 0
            mini_batch_loss = 0

        total_loss += loss

    return total_loss / data_train_x.shape[0]




def train_with_noise(data_train_x, data_train_y, model, optimizer, g, g_undersample,
                     pkor, pkor_undersample, b_undersample,
                     epoch, g_filt, undersample,
                     mini_batch=10, noise_lvl=0):
    model.train()
    total_loss = 0
    mini_batch_idx = 0
    mini_batch_loss = 0
    maximum_signal = data_train_x.max() * 0.0001
    grow_period = 5

    for i in np.arange(data_train_x.shape[0]):
        mini_batch_idx += 1
        inn = data_train_x[i, ]
        lin_scaling = (30 * np.min([epoch, grow_period])/ grow_period) + 1
        nois = maximum_signal * torch.rand(1) * lin_scaling * torch.randn_like(inn)
        #nois = maximum_signal * torch.rand(1) * epoch * torch.randn_like(inn)
        inn = (inn.to(device) + nois.to(device)) * g_filt
        inn = (inn * undersample[:,np.newaxis])

        optimizer.zero_grad()
        pred = model(g, g_undersample, inn, pkor, pkor_undersample, b_undersample)
        loss = F.mse_loss(pred, data_train_y[i, ].to(device))

        mini_batch_loss += loss

        if mini_batch_idx == mini_batch:
            mini_batch_loss = mini_batch_loss / mini_batch
            mini_batch_loss.backward()
            optimizer.step()
            mini_batch_idx = 0
            mini_batch_loss = 0

        total_loss += loss
    return total_loss / data_train_x.shape[0]
