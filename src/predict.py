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
from src.evaluate import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#from plotly.subplots import make_subplots
#import plotly.graph_objects as go




def interfer_with_test_data(pth2data, data_nam, labell, pth2save, af_mat, hf_mat,
                                model, g, pkor):
    print("File: ", data_nam)
    print("Label: ", labell)
    fh_dat = h5py.File(pth2data+data_nam,'r')
    chunk_size = 1000
    n_chunks = np.int(np.ceil(fh_dat[labell].shape[0]/chunk_size))
    t=0

    for i in range(n_chunks):
        sta_time = time.time()

        x_test = torch.tensor(fh_dat[labell][i*chunk_size:(i+1)*chunk_size,:6208,:40],dtype=torch.float32)

        x_test, dd, ddd = pre_processing(x_test, af_mat, hf_mat, n_coi=20)
        x_test_pred = test(dat_to_test = x_test,
                            model = model,
                            g = g,
                            pkor = pkor)

        #print(x_test_pred.shape)

        if i == 0:
            fh_pred = h5py.File(pth2save,'a')
            fh_pred.create_dataset(labell+'_pr', data=x_test_pred, chunks=True, maxshape=(None,6208,40))
        else:
            new_len = (fh_pred[labell+'_pr'].shape[0] + x_test_pred.shape[0])
            fh_pred[labell+'_pr'].resize(new_len, axis = 0)
            fh_pred[labell+'_pr'][-x_test_pred.shape[0]:] = x_test_pred

        t_chunk = time.time()-sta_time
        t += t_chunk
        print("Chunk (" + str(i+1) + "/" + str(n_chunks) + ') in ', t_chunk)
    fh_dat.close()
    fh_pred.close()
    print("total time: ", t, "\n")


def interfer_with_test_data_old(pth2data, data_nam, labell, pth2save):

    sta_time = time.time()
    fh = h5py.File(pth2data+data_nam,'r')
    x_test = torch.tensor(fh[labell][:,:6208,:40],dtype=torch.float32)
    fh.close()
    print('Loaded in ', time.time()-sta_time)

    sta_time = time.time()
    x_test, dd, ddd = pre_processing(x_test, af_mat, hf_mat, n_coi=20)
    x_test_pred = test(x_test)
    print('Predicted in ', time.time()-sta_time)

    sta_time = time.time()
    fh = h5py.File(pth2save,'a')
    fh.create_dataset(labell+'_pr', data=x_test_pred)
    fh.close()
    print('Saved in ', time.time()-sta_time)
