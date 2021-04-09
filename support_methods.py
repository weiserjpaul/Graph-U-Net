import numpy as np




def get_indices(shape, batch_size, validation_volunteer, path_to_model, n_batches = -1, n_batches_val = -1):
    #shape = (33600, 6208, 40)
    #batch_size = 50
    #validation_volunteer = 5 #0,...,6
    train_volunteers = [0,1,2,3,4,5,6]
    train_volunteers.remove(validation_volunteer)

    file = open(path_to_model + "logfile.txt","a")
    file.write("Trained on volunteers: " + str(train_volunteers) + "\n")
    file.write("Validated on volunteer: " + str(validation_volunteer) + "\n")
    file.write("\n")
    file.close()

    n_partitions_per_patient = 16
    n_time_points_per_patient = 15
    n_volunteers = 6
    positions = 10
    aug = 2
    slices_per_vol = n_partitions_per_patient*n_time_points_per_patient*positions


    n_indices = shape[0] - (slices_per_vol*aug)
    if n_batches == -1:
        n_batches = np.int(np.ceil(n_indices/batch_size))
    n_val_indices = (slices_per_vol*aug)
    """
    if n_batches_val == -1:
        n_batches_val = np.int(np.ceil(n_val_indices/batch_size))
    """

    start_indices = []
    end_indices = []

    patient_start_end_indices = [validation_volunteer*slices_per_vol, (validation_volunteer+1)*slices_per_vol]
    patient_start_end_indices += [validation_volunteer*slices_per_vol + slices_per_vol*n_volunteers]
    patient_start_end_indices += [(validation_volunteer+1)*slices_per_vol + slices_per_vol*n_volunteers]
    #print(patient_start_end_indices)

    for i in range(n_batches):
        s = i*batch_size
        e = (i+1)*batch_size

        if e <  patient_start_end_indices[0]:
            start_indices.append(s)
            end_indices.append(e)
        elif s < patient_start_end_indices[0]:
            start_indices.append(s)
            end_indices.append(patient_start_end_indices[0])
        elif e+slices_per_vol <  patient_start_end_indices[2]:
            start_indices.append(s+slices_per_vol)
            end_indices.append(e+slices_per_vol)
        elif s+slices_per_vol < patient_start_end_indices[2]:
            start_indices.append(s+slices_per_vol)
            end_indices.append(patient_start_end_indices[2])
        else:
            start_indices.append(s+2*slices_per_vol)
            end_indices.append(e+2*slices_per_vol)

    #print(start_indices)
    #print(end_indices)
    #print(np.array(start_indices) - np.array(end_indices))


    start_val_indices = []
    end_val_indices = []

    total_batches_val = np.int(np.ceil(n_val_indices/batch_size))

    for i in range(total_batches_val):
        s = i*batch_size
        e = (i+1)*batch_size

        if e + patient_start_end_indices[0] <  patient_start_end_indices[1]:
            start_val_indices.append(s + patient_start_end_indices[0])
            end_val_indices.append(e + patient_start_end_indices[0])
        elif s + patient_start_end_indices[0] < patient_start_end_indices[1]:
            start_val_indices.append(s + patient_start_end_indices[0])
            end_val_indices.append(patient_start_end_indices[1])

        elif e+slices_per_vol <  patient_start_end_indices[2]:
            start_val_indices.append(s+slices_per_vol*(n_volunteers+validation_volunteer-1))
            end_val_indices.append(e+slices_per_vol*(n_volunteers+validation_volunteer-1))
        elif s+slices_per_vol < patient_start_end_indices[2]:
            start_val_indices.append(s+slices_per_vol*(n_volunteers+validation_volunteer-1))
            end_val_indices.append(patient_start_end_indices[3])


    mid = np.int(np.ceil(len(start_val_indices)/2))
    n_batches_test = len(start_val_indices) - mid
    start_test_indices = start_val_indices[mid:]
    end_test_indices = end_val_indices[mid:]

    if n_batches_val == -1 or n_batches_val > mid:
        if n_batches_val > mid:
            print("Number of validation batches exceeds maximum")
            print("n_batches is set to " + str(mid))
        n_batches_val = mid
        start_val_indices = start_val_indices[:mid]
        end_val_indices = end_val_indices[:mid]
    else:
        start_val_indices = start_val_indices[:n_batches_val]
        end_val_indices = end_val_indices[:n_batches_val]


    #print(start_val_indices)
    #print(end_val_indices)
    #print(np.array(start_val_indices) - np.array(end_val_indices))

    return start_indices, end_indices, n_batches, \
            start_val_indices, end_val_indices, n_batches_val, \
            start_test_indices, end_test_indices, n_batches_test




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
