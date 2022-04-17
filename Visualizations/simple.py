import numpy as np
import pybullet as p 
import rbdl 
import Utils.misc as ut
from Utils.initializer import InitializerConsistentHumanoid2
from Utils.angles import angle_util
import time
import copy
####
#### Initialization
####
id_simulator = p.connect(p.GUI)
p.configureDebugVisualizer(flag=p.COV_ENABLE_Y_AXIS_UP, enable=1)
p.configureDebugVisualizer(flag=p.COV_ENABLE_SHADOWS, enable=0)
target_joints = [ "head", "neck",   "left_knee", "left_ankle", "left_toe",  "right_knee",  "right_ankle", "right_toe",  "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
AU = angle_util()
ini = InitializerConsistentHumanoid2(1, target_joints) 
rbdl2bullet = ini.get_rbdl2bullet()

####
#### sample rotation and translation only for better visualization purpose (you need to provide correct R and T to see the results in a global frame if known.)
####
RT = np.array([0.0753311, 0.007644453, -0.9971293, 117.6134, -0.03987908, -0.9991937, -0.004647529, 1435.971, -0.9963609 ,0.0394145, 0.07557522, 4628.016 ,0, 0, 0 ,1 ]).reshape(4, 4)
R=RT[:3,:3].T
T=RT[:-1,3:].reshape(3)/1000
  
if __name__ == "__main__":  
    urdf_base_path = "./URDF/"  
    model0 = rbdl.loadModel((urdf_base_path+'manual.urdf').encode(), floating_base=True)
   
    id_robot = p.loadURDF(urdf_base_path+'manual.urdf',useFixedBase=False) 
    _, _, jointIds, jointNames = ut.get_jointIds_Names(id_robot)  
    q=copy.copy(np.load("./results/q_iter_dyn.npy", mmap_mode='r'))  
    q[:,6+27]=0
    q[:,6+35]=0
    q[:,6+36]=0
    q[:,6+37]=0
    q[:,6+38]=0  
    count=0 
    while(1): 
        print(count) 
        ut.visualization3D_multiple([id_robot], jointIds, rbdl2bullet, [q[count]], Rot=R,T=T , overlay=True) 

        time.sleep(0.005)

        count+=1
        if count >= len(q):
            count =0
