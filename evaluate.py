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



def validate(data_val_x, data_val_y, model, g, g_undersample,
             pkor, pkor_undersample, b_undersample):
        model.eval()
        total_loss = 0
        for i in np.arange(data_val_x.shape[0]):
            inn = data_val_x[i, ].to(device)
            #pred = model(g, inn, pkor)
            #loss = F.mse_loss(pred, data_val_y[i, ].to(device))

            loss = validate1(model = model,
                                g = g,
                                g_undersample = g_undersample,
                                inn = inn,
                                pkor = pkor,
                                pkor_undersample = pkor_undersample,
                                b_undersample = b_undersample,
                                data_val_y = data_val_y[i, ])

            total_loss += loss
        return total_loss/data_val_x.shape[0]


def validate_with_noise(data_val_x, data_val_y, model, g, g_undersample,
                        pkor, pkor_undersample, b_undersample,
                        epoch, g_filt, undersample):

        model.eval()
        total_loss = 0
        maximum_signal = data_val_x.max() * 0.0001

        for i in np.arange(data_val_x.shape[0]):
            inn = data_val_x[i, ]
            #pred = model(g, inn, pkor)

            lin_scaling = (30 * np.min([epoch, 5])/ 5) + 1
            nois = maximum_signal * torch.rand(1) * lin_scaling * torch.randn_like(inn)
            inn = (inn.to(device) + nois.to(device)) * g_filt
            inn = (inn * undersample[:,np.newaxis])

            loss = validate1(model = model,
                                g = g,
                                g_undersample = g_undersample,
                                inn = inn,
                                pkor = pkor,
                                pkor_undersample = pkor_undersample,
                                b_undersample = b_undersample,
                                data_val_y = data_val_y[i, ])

            total_loss += loss
        return total_loss/data_val_x.shape[0]


def validate1(model, g, g_undersample, inn, pkor, pkor_undersample,
                b_undersample, data_val_y):

    pred = model(g, g_undersample, inn, pkor, pkor_undersample, b_undersample)
    loss = F.mse_loss(pred, data_val_y.to(device))
    return loss.data.cpu()




def test(dat_to_test, model, g, g_undersample, pkor, pkor_undersample,
                b_undersample):

    model.eval()
    pred_all = []

    for i in np.arange(dat_to_test.shape[0]):
        inn = dat_to_test[i, ].to(device)
        pred = model(g, g_undersample, inn, pkor, pkor_undersample,b_undersample)
        pred_all.append(pred.cpu().detach().numpy())

    return np.array([pred_all])[0]


def test_with_noise(dat_to_test, model, g, g_undersample, pkor, pkor_undersample,
                    b_undersample, g_filt, undersample):

    model.eval()
    pred_all = []
    maximum_signal = dat_to_test.max() * 0.0001

    for i in np.arange(dat_to_test.shape[0]):
        inn = dat_to_test[i, ]
        lin_scaling = 31
        nois = maximum_signal * torch.rand(1) * lin_scaling * torch.randn_like(inn)
        inn = (inn.to(device) + nois.to(device)) * g_filt
        inn = (inn * undersample[:,np.newaxis])

        pred = model(g, g_undersample, inn, pkor, pkor_undersample, b_undersample)
        pred_all.append(pred.cpu().detach().numpy())

    return np.array([pred_all])[0]
