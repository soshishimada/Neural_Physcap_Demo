import Utils.openpose_util as opp
import numpy as np 
import argparse 
 
def process(interpolate,smoothing):
    head_keys = [14, 15, 16, 17]
    neck_key = [0]
    print('processing .....') 
    p_2ds = opp.get_2ds_from_jsons(args.input_path) 
    p_2ds = opp.handle_head_keypoints(p_2ds, neck_key, head_keys) 
    if interpolate:
        p_2ds = opp.openpose_interpolate(p_2ds) 
    if smoothing:
        p_2ds = opp.openpose_smoothing(p_2ds) 
    print("Done")
    return p_2ds

def save_data(save_path,save_name,data):
    np.save(save_path+save_name,data)
    return 0
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for predictions')
    parser.add_argument('--input_path', default="")  
    parser.add_argument('--save_path', default="./sample_data/")
    parser.add_argument('--save_file_name', default='test.npy')
    args = parser.parse_args()
 
    interpolate=1
    smoothing=1 
    p_2ds = process(interpolate,smoothing)#
    save_data(args.save_path,args.save_file_name,p_2ds) 