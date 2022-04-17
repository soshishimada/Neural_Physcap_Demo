import numpy as np
import copy
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import json
import os

def openpose_interpolate_realtime(p_2ds):
    for i in range(len(p_2ds)):
        # find frames that contain 0 (error).
        if not np.all(p_2ds[i]):
            # Provide the latest non-error joint position to the error joints .
            for j in range(len(p_2ds[i])):
                if p_2ds[i][j][0]==0:
                    p_2ds[i][j][0]=copy.copy(p_2ds[i-1][j][0])
                    p_2ds[i][j][1]=copy.copy(p_2ds[i-1][j][1])
    return p_2ds

def find_start_end_of_errors(j_line):
    zero_in_flag = 0
    all_pairs = []
    for j in range(len(j_line)):
        if j_line[j] == 0:
            if not zero_in_flag:
                start_index = j - 1
            zero_in_flag = 1

        if j_line[j] != 0 and zero_in_flag:
            zero_in_flag = 0
            end_index = j
            all_pairs.append((start_index, end_index))
    return all_pairs

def apply_interpolations(all_pairs,j_line,p_2ds):
    for pair in all_pairs:
        f = interp1d([pair[0], pair[1]], [j_line[pair[0]], j_line[pair[1]]])
        x = np.linspace(pair[0], pair[1], num=(pair[1] - pair[0] - 1) + 2, endpoint=True)
        for k, inter_val in enumerate(f(x)[1:-1]):
            j_line[pair[0] + k + 1] = inter_val
    return j_line

def substitute_avg_values(heads_sub,necks_sub):
    for i in range(len(heads_sub)):
        if not np.all(heads_sub[i]):
            head_values=heads_sub[i]
            n_non_zero = np.count_nonzero(heads_sub[i])
            if n_non_zero !=0:
                avg_head = sum(head_values) / n_non_zero
                for j in range(len(head_values)):
                    if head_values[j] == 0:
                        head_values[j]=avg_head
            else:
                for j in range(len(head_values)):
                   head_values[j]=necks_sub[i]

            heads_sub[i]=head_values
    return heads_sub

def handle_head_keypoints(p_2ds,neck_key,head_keys):
    heads = p_2ds[:,head_keys]
    necks = p_2ds[:,neck_key]

    heads_u = substitute_avg_values(copy.copy(heads[:,:,0]),copy.copy(necks[:,:,0]))
    heads_v = substitute_avg_values(copy.copy(heads[:,:,1]),copy.copy(necks[:,:,1]))
 

    heads[:,:,0]=heads_u
    heads[:,:,1]=heads_v

    p_2ds[:, head_keys] = heads
    return p_2ds

def openpose_interpolate(p_2ds):
    p_2ds=p_2ds.reshape(len(p_2ds),-1).T
    for i,j_line in enumerate(p_2ds):
        # find joints that contain 0 (error) through a sequence.
        if not np.all(j_line):
            # find pairs of frame index where error (start, end)
            all_pairs = find_start_end_of_errors(j_line)

            # apply linear interpolation on each pairs of indices
            p_2ds[i] = apply_interpolations(all_pairs,j_line,p_2ds)
    p_2ds=p_2ds.T
    p_2ds = p_2ds.reshape(len(p_2ds),-1,2)
    return p_2ds

def openpose_smoothing(p_2ds):
    p_2ds = p_2ds.reshape(len(p_2ds),-1).T
    for i,l in enumerate(p_2ds):
        p_2ds[i]= gaussian_filter1d(l,1.5)
    #print(p_2ds.shape)
    p_2ds=p_2ds.T
    p_2ds = p_2ds.reshape(len(p_2ds),-1,2)
    return p_2ds

def get_2ds_from_jsons(data_path): 
    json_files = os.listdir(data_path) 
    json_files.sort()
    json_files=json_files[1:]
    all_2ds = []
    for i,json_file in enumerate(json_files): 
        with open(data_path+json_file) as f:
            data = json.load(f)
        if len(data['people'])==0:
            p_2ds = np.zeros(72)
        else:
            contents = data['people'][0]
            p_2ds = np.array(contents['pose_keypoints_2d'])[3:]
        all_2ds.append(p_2ds)
    all_2ds=np.array(all_2ds)
    all_2ds=all_2ds.reshape(len(all_2ds),-1,3)[:,:,:-1]
    return all_2ds