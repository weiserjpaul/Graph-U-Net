import os
import sys
import dgl
import h5py
import time
import torch
import shutil
import argparse
import numpy as np
import networkx as nx
from dgl.nn import GMMConv
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.GDL_CC_support import create_log_file_name, apply_filter, add_additive_noise, pre_processing, order_data
from src.GDL_CC_support import mse_complex, mse_complex_summed, order_data_single_input
from src.network1 import UNet_old
from src.network2 import UNet
from src.network3 import GraphUnet
from src.train import train, train_with_noise
from src.evaluate import test, test_with_noise, validate, validate_with_noise
from src.predict import interfer_with_test_data
from src.support_methods import get_indices
from src.undersample import get_undersample


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#from plotly.subplots import make_subplots
#import plotly.graph_objects as go


#print(sys.path)
#print(sys.version_info)


##### parameters #####

dict = {}

dict["model_ids"] = ["UNet", "GraphUnet"]
dict["id"] = dict["model_ids"][0]

#params = [10, 20, 100, -1, -1]   # parameters for grid_search
#params = [50, 20, 100, -1, -1]   # parameters for training
#params = [10, 7, 20, 5, 2]   # paramters for testing
params = [50, 20, 250, -1, -1]   # paramters for dgx
dict["dgx"] = True

dict["n_epochs"] = params[0] #5 #50
dict["n_epochs_with_noise"] = params[1] #2 #20
dict["batch_size"] = params[2] #20 #100
dict["n_batches"] = params[3] #5 #-1   # set to -1 to iterate through the whole dataset
dict["n_batches_val"] = params[4] #2 #-1   # set to -1 to iterate through the whole dataset

dict["undersample"] = True
dict["n_fully_sampled_rings"] = 6

dict["self_edge"] = False

#dict["model_name"] = "model1"
dict["model_name"] = "test"

dict["hidden_feats"] = 20
dict["depth"] = 2 #3 #2
if dict["id"] == "GraphUnet":
    dict["ks"] = [0.33] * dict["depth"]
if dict["id"] == "UNet":
    dict["window_size_nodes"] = 4 #3 #6
    dict["window_size_rings"] = 2 #2 #3
    dict["window_shift_nodes"] = 4 #2 #4
    dict["window_shift_rings"] = 2 #2 #2



parser = argparse.ArgumentParser()
parser.add_argument("--name", "--model_name", help="Define the name of the model")
parser.add_argument("--hf", "--hidden_features", help="Define the number of features in the hidden layers", type=int)
parser.add_argument("--depth", help="Define the depth of the U-Net", type=int)
parser.add_argument("--selfedge", help="Define if the nodes are connected to themselves")

args = parser.parse_args()
if args.name:
    dict["model_name"] = args.name
if args.hf:
    dict["hidden_feats"] = args.hf
if args.depth:
    if args.depth <=2:
        dict["depth"] = args.depth
    else:
        raise argparse.ArgumentTypeError('Only a maximum depth of 2 is supported')
if args.selfedge:
    if args.selfedge.lower() in ('yes', 'true', 't', 'y', '1'):
        dict["self_edge"] = True
    elif args.selfedge.lower() in ('no', 'false', 'f', 'n', '0'):
        dict["self_edge"] = False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected for argument undersample')

dict["path_to_model"] = "models/" + dict["model_name"] + "/"
dict["path_to_predictions"] = dict["path_to_model"] + "predictions/"
dict["path_to_data"] = "../data/"
#dict["path_to_data"] = "Z:/data/"
dict["pth2save"] = dict["path_to_model"] + "predictions/Pred_Vol_010_Pos_06-10.h5"



b_train = True
b_val = True
b_pred = True
b_clean_test = True


##### Initialization #####
if b_clean_test == True:
    if dict["model_name"] != "test":
        if os.path.isdir(dict["path_to_model"]) == True:
            print("Model already exists. Choose different model name.")
            sys.exit()
        else:
            os.mkdir(dict["path_to_model"])
            os.mkdir(dict["path_to_predictions"])
    else:
        if os.path.isdir(dict["path_to_model"]) == True:
            shutil.rmtree(dict["path_to_model"])
        os.mkdir(dict["path_to_model"])
        os.mkdir(dict["path_to_predictions"])

file = open(dict["path_to_model"] + "logfile.txt","a")
file.write("Notes:" + "\n")
file.write("Model Name: " + dict["model_name"] + "\n")
file.write("Model ID: " + dict["id"] + "\n")
file.write("\n")

file.write("### Architecture-parameters: ###" + "\n")
file.write("Number of features in hidden layers: " + str(dict["hidden_feats"]) + "\n")
file.write("Depth: " + str(dict["depth"]) + "\n")
file.write("Self-Edge: " + str(dict["self_edge"]) + "\n")
if dict["id"] == "UNet":
    file.write("window_size_nodes: " + str(dict["window_size_nodes"]) + "\n")
    file.write("window_size_rings: " + str(dict["window_size_rings"]) + "\n")
    file.write("window_shift_nodes: " + str(dict["window_shift_nodes"]) + "\n")
    file.write("window_shift_rings: " + str(dict["window_shift_rings"]) + "\n")
if dict["id"] == "GraphUnet":
    file.write("Pooling Dropout: " + str(dict["ks"]) + "\n")
file.write("\n")

file.write("### Undersampling-parameters: ###" + "\n")
file.write("Undersample: " + str(dict["undersample"]) + "\n")
file.write("Number of fully sampled inner rings: " + str(dict["n_fully_sampled_rings"]) + "\n")
file.write("\n")

file.write("### Training-parameters: ###" + "\n")
file.write("Total epochs trained: " + str(dict["n_epochs"]) + "\n")
file.write("Epochs trained with noise: " + str(dict["n_epochs_with_noise"]) + "\n")
file.close()


##### Initialization of Data #####

sta_time = time.time()

name_dataset = 'DATA_CRT_32_sim_ver6.h5'
f_handle =h5py.File(dict["path_to_data"]+name_dataset,'r')
f_val = h5py.File(dict["path_to_data"]+name_dataset,'r')
#f_val = h5py.File('Valid_Test_Vol_001_BS.h5', "r")
name_dataset_old = "DATA_CRT_32_old.h5"
f_handle_old =h5py.File(dict["path_to_data"]+name_dataset_old,'r')

if dict["dgx"] == True:
    data_all_in =  torch.tensor(f_handle['input_data'][:,:6208,:40], dtype=torch.float32)
    data_all_out =  torch.tensor(f_handle['output_data'][:,:6208,:40], dtype=torch.float32)


af_mat = torch.tensor(np.repeat(np.array(f_handle['AF_vec']).transpose(),40, axis=1),dtype=torch.float32)
hf_mat = torch.tensor(np.repeat(np.array(f_handle['HF_vec']).transpose(),40, axis=1),dtype=torch.float32)
if dict["self_edge"] == True:
    edge_attr_raw = torch.tensor(np.array(f_handle['edge_attr']), dtype=torch.float32)
    edge_index_raw = torch.tensor(np.array(f_handle['edge_index']) - 1, dtype=torch.long)  # minus one is because of different indexinf between python and matlab
    pkor = torch.tensor(np.array(f_handle['pkor_vec']), dtype=torch.float32).to(device)
else:
    edge_attr_raw = torch.tensor(np.array(f_handle_old['edge_attr']), dtype=torch.float32)
    edge_index_raw = torch.tensor(np.array(f_handle_old['edge_index']) - 1, dtype=torch.long)  # minus one is because of different indexinf between python and matlab
    pkor = torch.tensor(np.array(f_handle_old['pkor_vec']), dtype=torch.float32).to(device)

g = dgl.graph((edge_index_raw[0,:], edge_index_raw[1,:])).to(device)
g.edata['w'] = edge_attr_raw.to(device)


undersample_1, undersample_2 = get_undersample(undersample = dict["undersample"],
                                                n_fully_sampled_rings = dict["n_fully_sampled_rings"],
                                                path_to_data = dict["path_to_data"])

edges_index_undersample = []
for i in range(len(edge_index_raw[0,:])):
    val = edge_index_raw[0,i]
    if undersample_1[val] == 1:
        edges_index_undersample.append(i)

pkor_undersample = pkor[edges_index_undersample].to(device)
edge_index_raw_undersample = edge_index_raw[:,edges_index_undersample]
g_undersample = dgl.graph((edge_index_raw_undersample[0,:], edge_index_raw_undersample[1,:])).to(device)



print('Initialization in ', time.time()-sta_time)


##### initiate model #####
"""
model = UNet_old(in_chan = 40,
                out_chan = 40,
                hidden_feats = dict["hidden_feats"],
                depth = dict["depth"]).to(device)
"""
if dict["id"] == "UNet":
    model = UNet(edge = edge_index_raw,
                    window_size_nodes = dict["window_size_nodes"],
                    window_size_rings = dict["window_size_rings"],
                    window_shift_nodes = dict["window_shift_nodes"],
                    window_shift_rings = dict["window_shift_rings"],
                    pkor = pkor,
                    in_chan = 40,
                    out_chan = 40,
                    hidden_feats = dict["hidden_feats"],
                    depth = dict["depth"]).to(device)
if dict["id"] == "GraphUnet":
    model = GraphUnet(ks = dict["ks"],
                        in_chan = 40,
                        out_chan = 40,
                        hidden_feats = dict["hidden_feats"],
                        depth = dict["depth"]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)



##### training #####

old_loss_batch = 0
old_loss_epoch = 0
activate_train_with_noise = False
log_batch_loss = np.array([])
log_epoch_loss = np.array([])

old_loss_batch_val = 0
old_loss_epoch_val = 0
log_batch_loss_val = np.array([])
log_epoch_loss_val = np.array([])
lowest_epoch_loss_val = None

model_save_name = "model"

start_indices, end_indices, dict["n_batches"], \
start_val_indices, end_val_indices, dict["n_batches_val"], \
start_test_indices, end_test_indices, dict["n_batches_test"] = get_indices(f_handle['input_data'].shape,
                                                                        batch_size = dict["batch_size"],
                                                                        validation_volunteer = 6,
                                                                        path_to_model = dict["path_to_model"],
                                                                        n_batches = dict["n_batches"],
                                                                        n_batches_val = dict["n_batches_val"])



for epoch in range(dict["n_epochs"]): # former (1,50)
    if b_train:
        sta_epoch = time.time()
        count_size = 0
        cur_loss_epoch = 0
        if epoch == (dict["n_epochs"] - dict["n_epochs_with_noise"]):
            activate_train_with_noise = True
            print("+++ Training with noise from here on! +++\n")

            file = open(dict["path_to_model"] + "logfile.txt","a")
            file.write("Training: \n")
            file.write("number of batches per epoch: " + str(dict["n_batches"]) + "\n")
            file.write("Batch Loss: ")
            for val in range(len(log_batch_loss)):
                file.write(str(log_batch_loss[val]))
                file.write(" ")
            file.write("\n")
            file.write("Epoch Loss: " + np.array2string(log_epoch_loss) + "\n")
            file.write("Validation Batch Loss: ")
            for val in range(len(log_batch_loss_val)):
                file.write(str(log_batch_loss_val[val]))
                file.write(" ")
            file.write("\n")
            file.write("Validation Epoch Loss: " + np.array2string(log_epoch_loss_val) + "\n")
            file.write("+++ Training with noise from here on! +++\n")
            file.close()
            log_batch_loss = np.array([])
            log_epoch_loss = np.array([])
            log_batch_loss_val = np.array([])
            log_epoch_loss_val = np.array([])

            model_save_name = "model_NOISE"
            lowest_epoch_loss_val = None


        for i in range(dict["n_batches"]):
            sta_batch = time.time()
            if dict["dgx"] == False:
                x_raw = torch.tensor(f_handle['input_data'][start_indices[i]:end_indices[i],:6208,:40], dtype=torch.float32)
                y_raw = torch.tensor(f_handle['output_data'][start_indices[i]:end_indices[i],:6208,:40], dtype=torch.float32)
            elif dict["dgx"] == True:
                x_raw = data_all_in[start_indices[i]:end_indices[i]]
                y_raw = data_all_out[start_indices[i]:end_indices[i]]
            x_pp, x_vl_pp, norm_x = pre_processing(x_raw, af_mat, hf_mat, n_coi=20)
            y_pp, y_vl_pp, norm_y = pre_processing(y_raw, af_mat, hf_mat, n_coi=20)

            if dict["undersample"] == True:
                x_pp = (x_pp * undersample_1[np.newaxis,:,np.newaxis]).float()



            if activate_train_with_noise == False:
                cur_loss_batch = train(data_train_x = x_pp,
                                        data_train_y = y_pp,
                                        model = model,
                                        optimizer = optimizer,
                                        g = g,
                                        g_undersample = g_undersample,
                                        pkor = pkor,
                                        pkor_undersample = pkor_undersample,
                                        b_undersample = dict["undersample"])
            else:
                g_filt = (hf_mat * af_mat) / norm_x
                cur_loss_batch = train_with_noise(data_train_x = x_raw,
                                                    data_train_y = y_pp,
                                                    model = model,
                                                    optimizer = optimizer,
                                                    g = g,
                                                    g_undersample = g_undersample,
                                                    pkor = pkor,
                                                    pkor_undersample = pkor_undersample,
                                                    b_undersample = dict["undersample"],
                                                    epoch = epoch - (dict["n_epochs"] - dict["n_epochs_with_noise"]),
                                                    g_filt = g_filt.to(device),
                                                    undersample = torch.tensor(undersample_1).to(device))


            cur_loss_batch = cur_loss_batch.cpu().detach().numpy()
            log_batch_loss = np.append(log_batch_loss, cur_loss_batch)
            delt_loss_batch = old_loss_batch - cur_loss_batch
            sto_batch = time.time() - sta_batch
            log_batch = ' ~ Epoch: {:03d}, Batch: ({:03d}/{:03d}) Loss: {:.8f}, Time: {:.4f}, Change in loss: {:.8f}'
            print(log_batch.format(epoch+1, i+1, dict["n_batches"], cur_loss_batch, sto_batch, delt_loss_batch))

            old_loss_batch = cur_loss_batch
            cur_loss_epoch += cur_loss_batch * x_raw.shape[0]
            count_size += x_raw.shape[0]


        cur_loss_epoch = cur_loss_epoch / count_size
        log_epoch_loss = np.append(log_epoch_loss, cur_loss_batch)
        delt_loss_epoch = old_loss_epoch - cur_loss_epoch
        sto_epoch = time.time() - sta_epoch
        log_epoch = 'Epoch: {:03d}, Loss: {:.8f}, Time: {:.4f}, Change in loss: {:.8f}\n'
        print(log_epoch.format(epoch+1, cur_loss_epoch, sto_epoch ,delt_loss_epoch))
        old_loss_epoch = cur_loss_epoch




### validation ###
    if b_val:
        count_size = 0
        cur_loss_epoch_val = 0
        sta_epoch_val = time.time()
        for i in range(dict["n_batches_val"]):
            sta_batch_val = time.time()
            if dict["dgx"] == False:
                x_raw = torch.tensor(f_val['input_data'][start_val_indices[i]:end_val_indices[i],:6208,:40], dtype=torch.float32)
                y_raw = torch.tensor(f_val['output_data'][start_val_indices[i]:end_val_indices[i],:6208,:40], dtype=torch.float32)
            elif dict["dgx"] == True:
                x_raw = data_all_in[start_val_indices[i]:end_val_indices[i]]
                y_raw = data_all_out[start_val_indices[i]:end_val_indices[i]]
            x_pp, x_vl_pp, norm_x = pre_processing(x_raw, af_mat, hf_mat, n_coi=20)
            y_pp, y_vl_pp, norm_y = pre_processing(y_raw, af_mat, hf_mat, n_coi=20)

            if dict["undersample"] == True:
                x_pp = (x_pp * undersample_1[np.newaxis,:,np.newaxis]).float()

            if activate_train_with_noise == False:
                cur_loss_batch_val = validate(data_val_x = x_pp,
                                                data_val_y = y_pp,
                                                model = model,
                                                g = g,
                                                g_undersample = g_undersample,
                                                pkor = pkor,
                                                pkor_undersample = pkor_undersample,
                                                b_undersample = dict["undersample"])
            else:
                g_filt = (hf_mat * af_mat) / norm_x
                cur_loss_batch_val = validate_with_noise(data_val_x = x_raw,
                                                            data_val_y = y_pp,
                                                            model = model,
                                                            g = g,
                                                            g_undersample = g_undersample,
                                                            pkor = pkor,
                                                            pkor_undersample = pkor_undersample,
                                                            b_undersample = dict["undersample"],
                                                            epoch = epoch - (dict["n_epochs"] - dict["n_epochs_with_noise"]),
                                                            g_filt = g_filt.to(device),
                                                            undersample = torch.tensor(undersample_1).to(device))



            cur_loss_batch_val = cur_loss_batch_val.cpu().detach().numpy()
            log_batch_loss_val = np.append(log_batch_loss_val, cur_loss_batch_val)
            delt_loss_batch_val = old_loss_batch_val - cur_loss_batch_val
            sto_batch_val = time.time() - sta_batch_val
            log_batch_val = ' ~ Validate Epoch: {:03d}, Batch: ({:03d}/{:03d}) Loss: {:.8f}, Time: {:.4f}, Change in loss: {:.8f}'
            print(log_batch_val.format(epoch+1, i+1, dict["n_batches_val"], cur_loss_batch_val, sto_batch_val, delt_loss_batch_val))

            old_loss_batch_val = cur_loss_batch_val
            cur_loss_epoch_val += cur_loss_batch_val * x_raw.shape[0]
            count_size += x_raw.shape[0]


        cur_loss_epoch_val = cur_loss_epoch_val / count_size
        log_epoch_loss_val = np.append(log_epoch_loss_val, cur_loss_batch_val)
        delt_loss_epoch_val = old_loss_epoch_val - cur_loss_epoch_val
        sto_epoch_val = time.time() - sta_epoch_val
        log_epoch_val = 'Validate Epoch: {:03d}, Loss: {:.8f}, Time: {:.4f}, Change in loss: {:.8f}\n'
        print(log_epoch_val.format(epoch+1, cur_loss_epoch_val, sto_epoch_val ,delt_loss_epoch_val))
        old_loss_epoch_val = cur_loss_epoch_val

        if activate_train_with_noise == False:
            if lowest_epoch_loss_val == None or cur_loss_epoch_val < lowest_epoch_loss_val:
                lowest_epoch_loss_val = cur_loss_epoch_val
                torch.save(model.state_dict(), dict["path_to_model"] + model_save_name + ".pt")
        else:
            if epoch - (dict["n_epochs"] - dict["n_epochs_with_noise"]) >= 4:
                if lowest_epoch_loss_val == None or cur_loss_epoch_val < lowest_epoch_loss_val:
                    lowest_epoch_loss_val = cur_loss_epoch_val
                    torch.save(model.state_dict(), dict["path_to_model"] + model_save_name + ".pt")






file = open(dict["path_to_model"] + "logfile.txt","a")
file.write("Batch Loss: ")
for val in range(len(log_batch_loss)):
    file.write(str(log_batch_loss[val]))
    file.write(" ")
file.write("\n")
file.write("Epoch Loss: " + np.array2string(log_epoch_loss) + "\n")
file.write("Validation Batch Loss: ")
for val in range(len(log_batch_loss_val)):
    file.write(str(log_batch_loss_val[val]))
    file.write(" ")
file.write("\n")
file.write("Validation Epoch Loss: " + np.array2string(log_epoch_loss_val) + "\n")
file.write("\n")
file.close()


##### load model #####

if b_pred:
    print("+++ Testing +++ \n")

    model_noise = model
    b_pred_noise = True

    model.load_state_dict(torch.load(dict["path_to_model"] + "model.pt"))
    if os.path.exists(dict["path_to_model"] + "model_NOISE.pt"):
        model_noise.load_state_dict(torch.load(dict["path_to_model"] + "model_NOISE.pt"))
    else:
        print("Could not find a Model thats was trained with noise!!! \n")
        b_pred_noise = False

##### predict #####


    label_p = "pred"
    label_p_noise = "pred_noise"
    label_t = "true"
    #n_batches_val = 2
    for i in range(dict["n_batches_test"]):
        percent = str(np.round((i+1)/dict["n_batches_test"]*100, decimals = 2))
        print("Progress: " + percent + "%  ", end='\r')
        if dict["dgx"] == False:
            x_raw = torch.tensor(f_val['input_data'][start_test_indices[i]:end_test_indices[i],:6208,:40], dtype=torch.float32)
            y_raw = torch.tensor(f_val['output_data'][start_test_indices[i]:end_test_indices[i],:6208,:40], dtype=torch.float32)
        elif dict["dgx"] == True:
            x_raw = data_all_in[start_test_indices[i]:end_test_indices[i]]
            y_raw = data_all_out[start_test_indices[i]:end_test_indices[i]]
        x_pp, x_vl_pp, norm_x = pre_processing(x_raw, af_mat, hf_mat, n_coi=20)
        y_pp, y_vl_pp, norm_y = pre_processing(y_raw, af_mat, hf_mat, n_coi=20)

        if dict["undersample"] == True:
            x_pp = (x_pp * undersample_1[np.newaxis,:,np.newaxis]).float()

        x_test_pred = test(dat_to_test = x_pp,
                        model = model,
                        g = g,
                        g_undersample = g_undersample,
                        pkor = pkor,
                        pkor_undersample = pkor_undersample,
                        b_undersample = dict["undersample"])

        if b_pred_noise:
            g_filt = (hf_mat * af_mat) / norm_x
            x_test_pred_noise = test_with_noise(dat_to_test = x_raw,
                                                model = model_noise,
                                                g = g,
                                                g_undersample = g_undersample,
                                                pkor = pkor,
                                                pkor_undersample = pkor_undersample,
                                                b_undersample = dict["undersample"],
                                                g_filt = g_filt.to(device),
                                                undersample = torch.tensor(undersample_1).to(device))


        if i == 0:
            fh_pred = h5py.File(dict["pth2save"],'a')
            fh_pred.create_dataset(label_p, data=x_test_pred, chunks=True, maxshape=(None,6208,40))
            if b_pred_noise:
                fh_pred.create_dataset(label_p_noise, data=x_test_pred_noise, chunks=True, maxshape=(None,6208,40))
            fh_pred.create_dataset(label_t, data=y_pp, chunks=True, maxshape=(None,6208,40))
        else:
            new_len = (fh_pred[label_p].shape[0] + x_test_pred.shape[0])
            fh_pred[label_p].resize(new_len, axis = 0)
            fh_pred[label_p][-x_test_pred.shape[0]:] = x_test_pred

            if b_pred_noise:
                new_len = (fh_pred[label_p_noise].shape[0] + x_test_pred_noise.shape[0])
                fh_pred[label_p_noise].resize(new_len, axis = 0)
                fh_pred[label_p_noise][-x_test_pred_noise.shape[0]:] = x_test_pred_noise

            new_len = (fh_pred[label_t].shape[0] + y_pp.shape[0])
            fh_pred[label_t].resize(new_len, axis = 0)
            fh_pred[label_t][-y_pp.shape[0]:] = y_pp

    fh_pred.close()
    print("\n")
