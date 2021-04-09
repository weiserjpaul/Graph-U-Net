import os
import numpy as np
import torch
import h5py

from torch_geometric.data import Data, DataLoader

def create_log_file_name(base_name_='Log_file'):
    c = 1
    base_name = '{:s}_{:d}.log'.format(base_name_, c)
    while os.path.isfile(base_name):
        c += 1
        base_name = '{:s}_{:d}.log'.format(base_name_, c)
    return base_name


def apply_filter(x, f):
    x_out = torch.zeros_like(x)
    for i in np.arange(0, x.shape[0]):
        # print('x shape is: ', x.shape)
        # print('f shape is: ', f.shape)
        x_out[i, ] = x[i, ] * f
    return x_out


def add_additive_noise(dat, nois_lvl, cuda_mod=False):
    if cuda_mod:
        dat = dat + nois_lvl * torch.tensor(np.random.standard_normal(dat.shape), dtype=torch.float32).cuda()
    else:
        dat = dat + nois_lvl * torch.tensor(np.random.standard_normal(dat.shape), dtype=torch.float32)
    return dat


def pre_processing(x, af, hf, x_vl=None, n_lvl=0, max_x=None, n_coi=20):
    if n_lvl != 0:
        x = add_additive_noise(x, n_lvl)
        if torch.is_tensor(x_vl):
            x_vl = add_additive_noise(x_vl, n_lvl)

    x = apply_filter(apply_filter(x, hf[:, :2*n_coi]), af[:, :2*n_coi])
    if torch.is_tensor(x_vl):
        x_vl = apply_filter(apply_filter(x_vl, hf[:, :2*n_coi]), af[:, :2*n_coi])

    if not torch.is_tensor(max_x):
        max_x = torch.max(torch.sqrt(torch.pow(x[:, :, 0:n_coi], 2) + torch.pow(x[:, :, n_coi:], 2)))
        if torch.is_tensor(x_vl):
            max_x_vl = torch.max(torch.sqrt(torch.pow(x_vl[:, :, 0:n_coi], 2) + torch.pow(x_vl[:, :, n_coi:], 2)))
            if max_x_vl > max_x:
                max_x = max_x_vl
            x_vl = x_vl / max_x

    x = x / max_x

    return x, x_vl, max_x


def order_data(x, y):
    data = []
    for i in np.arange(0, x.shape[0]):
        data.append(Data(x=x[i, ], y=y[i, ]))  # x has size [num_nodes, num_nodes_features]
    return DataLoader(data, shuffle=False)

def order_data_single_input(x):
    data = []
    for i in np.arange(0, x.shape[0]):
        data.append(Data(x=x[i, ]))
    return DataLoader(data, shuffle=False)
                    

def mse_complex(a, b, n_coi=20):
    c = a-b
    c = torch.mean(torch.pow(c[:, 0:n_coi], 2)+torch.pow(c[:, n_coi:], 2))
    return c


def mse_complex_summed(a, b, n_coi=20):
    a_r = torch.sum(a[:, :n_coi], 1, keepdim=True)
    a_i = torch.sum(a[:, n_coi:], 1, keepdim=True)

    b_r = torch.sum(b[:, :n_coi], 1, keepdim=True)
    b_i = torch.sum(b[:, n_coi:], 1, keepdim=True)

    return torch.mean(torch.pow(a_r - b_r, 2) + torch.pow(a_i - b_i, 2))
