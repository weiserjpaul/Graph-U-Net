import h5py
import numpy as np




def get_undersample(undersample = False, n_fully_sampled_rings = 6, path_to_data = "/home/cir/pweiser/mrsi/data/"):
    fh002 = h5py.File(path_to_data + "Vol_002/Vol_002_Case10_16.h5", "r")
    kx_s = fh002["kx"]
    ky_s = fh002["ky"]

    kx_s = np.transpose(kx_s,[1,0])
    ky_s = np.transpose(ky_s,[1,0])

    n_circ = kx_s.shape[1]

    kx_s = np.pi * (kx_s / (np.max(kx_s) / (n_circ-0.5/n_circ) ))
    ky_s = np.pi * (ky_s / (np.max(ky_s) / (n_circ-0.5/n_circ) ))


    W, k_coord = create_brute_force_FT_matrix([32,32], kx_s, ky_s)

    radii = np.sqrt(np.power(k_coord[:,0],2) + np.power(k_coord[:,1],2))
    radii_unique = np.unique(np.round(radii,decimals=5))

    undersample_bin_1, undersample_bin_2 = circs_to_undersample(undersample = undersample, n_fully_sampled_rings = n_fully_sampled_rings)

    undersample_1 = np.array([1 if x in (radii_unique*undersample_bin_1) else 0 for x in np.round(radii,decimals=5)])
    undersample_2 = np.array([1 if x in (radii_unique*undersample_bin_2) else 0 for x in np.round(radii,decimals=5)])

    fh002.close()

    return undersample_1, undersample_2



def circs_to_undersample(undersample = False, n_fully_sampled_rings = 6):
    n_circs = 16
    undersample_bin_1 = np.array([])
    undersample_bin_2 = np.array([])
    if undersample == False:
        n_fully_sampled_rings = n_circs

    for i in range(n_circs):
        if i < n_fully_sampled_rings:
            undersample_bin_1 = np.append(undersample_bin_1, 1)
            undersample_bin_2 = np.append(undersample_bin_2, 0)
            last_elem = 1
        else:
            if last_elem == 1:
                undersample_bin_1 = np.append(undersample_bin_1, 0)
                undersample_bin_2 = np.append(undersample_bin_2, 1)
                last_elem = 0
            else:
                undersample_bin_1 = np.append(undersample_bin_1, 1)
                undersample_bin_2 = np.append(undersample_bin_2, 0)
                last_elem = 1
    return undersample_bin_1, undersample_bin_2



def create_brute_force_FT_matrix(In, GridX, GridY):

    N1 = GridX.shape[0]
    N2 = GridX.shape[1]

    NN1 = In[0]
    NN2 = In[1]

    XX, YY = np.meshgrid(np.arange(-NN1/2+1,NN1/2+1), np.arange(-NN2/2+1,NN2/2+1))

    ii = 1j

    W = np.zeros([np.size(GridX), NN1*NN2], dtype = 'complex_')
    kp = 0

    k_coord = np.zeros([np.size(GridX), 2])

    for j in range(N2):
        for k in range(N1):
            Map=np.exp(-ii*(GridX[k,j]*XX + GridY[k,j]*YY))

            W[kp,:] = Map.flatten("F")

            k_coord[kp,0] = GridX[k,j]
            k_coord[kp,1] = GridY[k,j]
            kp += 1

    return W, k_coord
