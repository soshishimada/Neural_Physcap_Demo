
import pybullet as p
import numpy as np
import torch
import math

import rbdl
class Initializer():
    def __init__(self,batch_size,target_joints):
        self.rbdl_joint_dic = {
            "base": 1, "left_hip": 3, "left_knee": 6, "left_ankle": 7, "left_toe": 11, "left_heel": 12, "right_hip": 13,
            "right_knee": 16, "right_ankle": 17, "right_toe": 21, "right_heel": 22, "neck": 37, "left_clavicle": 23, "left_shoulder": 25, "left_elbow": 27,
            "left_wrist": 29, "right_clavicle": 30, "right_shoulder": 32, "right_elbow": 34, "right_wrist": 36
        }
        self.rbdl_marker_dic = {
            "base": 1, "left_hip": 4, "left_knee": 6, "left_ankle": 7, "left_toe": 11, "left_heel": 12, "right_hip": 15,
            "right_knee": 16, "right_ankle": 17, "right_toe": 21, "right_heel": 22, "neck": 37, "left_clavicle": 23, "left_shoulder": 25, "left_elbow": 27,
            "left_wrist": 29, "right_clavicle": 30, "right_shoulder": 32, "right_elbow": 34, "right_wrist": 36
        }
        self.target_joints = target_joints
        self.target_joint_ids = [self.rbdl_joint_dic[key] for key in self.target_joints]
        self.target_marker_ids = [self.rbdl_marker_dic[key] for key in self.target_joints]
        self.scaler= 2000
        self.K = np.array([752.6881, 0, 517.431, 0, 0, 752.8311, 500.631, 0, 0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 4)#[:3, :3]
        #self.RT = np.array([ -0.0121125, 0.0119637, -0.9998551, -708.5256/self.scaler, 0.07398777, -0.9971766, -0.0128279, 887.3783/self.scaler, -0.9971856, -0.07413242, 0.01119314, 3340.92/self.scaler, 0, 0, 0, 1 ]).reshape(4,4)
        self.RT = np.array([1, 0, 0, -0, 0, 1, 0, 0, 0, 0, 1,0, 0, 0, 0, 1 ]).reshape(4,4)
        self.batch_size=batch_size
        self.rbdl2bullet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 28, 29,  30, 31, 32, 33, 34, 35, 36, 20, 21, 22]
        self.bullet2rbdl = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 34, 35, 36, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    def get_rbdl_dic(self):
        return self.rbdl_joint_dic

    def change_color_bullet(self,id_robot,color):
        for j in range(p.getNumJoints(id_robot)):
            p.changeVisualShape(id_robot, j, rgbaColor=color)
        return 0
    def remove_collision(self,robot1,robot2):
        for i in range(p.getNumJoints(robot1)):
            for j in range(p.getNumJoints(robot2)):
                p.setCollisionFilterPair(robot1, robot2, i, j, 0)
    def get_rbdl2bullet(self):
        return self.rbdl2bullet

    def get_lthrth_ids(self):
        return np.array([self.rbdl_joint_dic[key] for key in ["left_toe","left_heel","right_toe","right_heel",]])

    def get_target_joint_ids(self):
        return self.target_joint_ids

    def get_target_marker_ids(self):
        return self.target_marker_ids

    def get_P_tensor(self,P):
        P_tensor = torch.zeros(self.batch_size, 3 * len(self.target_joint_ids), 4 * len(self.target_joint_ids))
        for i in range(int(P_tensor.shape[1] / 3)):
            P_tensor[:, i * 3:(i + 1) * 3, i * 4:(i + 1) * 4] = P
        return P_tensor.type(torch.FloatTensor)

    def get_projection_and_deriv(self):
        P = torch.FloatTensor(np.dot(self.K, self.RT)[:3])
        P_tensor = self.get_P_tensor(P)
        get_ids_P = [x for x in range(P_tensor.shape[2]) if (x + 1) % 4 != 0]
        P_tensor_deriv = P_tensor[:, :, get_ids_P]
        return P_tensor,P_tensor_deriv

    def get_bone_length(self,model):
        init_q = np.zeros(model.q_size)
        init_q[-1] = 1
        target_joints = ["base", "head", "neck", "left_hip", "left_knee", "left_ankle", "left_toe", "right_hip", "right_knee", "right_ankle", "right_toe", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]
        joint_pos_dic = dict( [(key, rbdl.CalcBodyToBaseCoordinates(model, init_q, self.rbdl_joint_dic[key], np.zeros(3))) for key in  target_joints])
        joint_pairs = [("head", "neck"), ("neck", "base"), ("base", "left_hip"), ("base", "right_hip"),
                       ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_ankle", "left_toe"),
                       ("right_hip", "right_knee"), ("right_knee", "right_ankle"), ("right_ankle", "right_toe"),
                       ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
                       ("left_elbow", "left_wrist"),
                       ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")]
        bone_len_list = []
        for i, pairs in enumerate(joint_pairs):
            length = np.linalg.norm(joint_pos_dic[pairs[0]] - joint_pos_dic[pairs[1]])
            bone_len_list.append(length)
        return np.array(bone_len_list)

    def model_bone_length_getter(self,model_addresses):
        model_bone_len_dic = {}
        for key in model_addresses.keys():
            model_bone_len_dic[key] = self.get_bone_length(model_addresses[key])
        return model_bone_len_dic

class InitializerConsistentHumanoid(Initializer):
    def __init__(self,batch_size,target_joints):
        self.rbdl_joint_dic = {
            "base": 1, "head": 40, "neck": 37, "left_hip": 3, "left_knee": 7, "left_ankle": 8, "left_toe": 11,
            "left_heel": 12, "right_hip": 13, "right_knee": 17, "right_ankle": 18, "right_toe": 21, "right_heel": 22, "left_clavicle": 23,
            "left_shoulder": 25, "left_elbow": 27,  "left_wrist": 29, "right_clavicle": 30, "right_shoulder": 32, "right_elbow": 34, "right_wrist": 36
        }
        self.gt_mpi_dic = {
                "base": 4, "head": 7, "neck": 5, "left_hip": 18, "left_knee": 19, "left_ankle": 20, "left_toe": 22,
                "right_hip": 23,
                "right_knee": 24, "right_ankle": 25, "right_toe": 27, "left_clavicle": 8, "left_shoulder": 9, "left_elbow": 10,
                "left_wrist": 11, "right_clavicle": 13, "right_shoulder": 14, "right_elbow": 15, "right_wrist": 16
            }
        self.gt_human36M_dic={
            "base":0,"left_hip":6,"left_knee":7,"left_ankle":8, "left_toe":9,"right_hip":1,"right_knee":2,"right_ankle":3,"right_toe":4, "neck":13, "head":15,
            "left_shoulder":17,"left_elbow":18,"left_wrist":19,"right_shoulder":25, "right_elbow":26,"right_wrist":27
        }
        self.gt_DeepCap_dic = {
            "base": 14, "left_hip": 11, "left_knee": 12, "left_ankle": 13, "left_toe": 16, "right_hip": 8,
            "right_knee": 9, "right_ankle": 10, "right_toe": 15, "neck": 1, "head": 0,
            "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7, "right_shoulder": 2, "right_elbow": 3,
            "right_wrist": 4
        }
        self.target_joints = target_joints
        self.target_joint_ids = [self.rbdl_joint_dic[key] for key in self.target_joints]
        #self.target_marker_ids = [self.rbdl_marker_dic[key] for key in self.target_joints]
        self.scaler= 1000
        self.batch_size=batch_size

        """ caution!!!! this might be opposite!"""
        self.bullet2rbdl  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 34, 35, 36, 37, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        self.rbdl2bullet  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 20, 21, 22, 23]

    def get_max_min_anlges(self):
        max_angles=math.pi*torch.ones(self.batch_size,len(self.bullet2rbdl))
        min_angles=-math.pi*torch.ones(self.batch_size,len(self.bullet2rbdl))
        #print(max_angles.shape,min_angles)
        max_min_dic ={
                      'left_hip_X': (1, [-2.5, 1]),
                      'left_hip_Y': (2,[-math.pi/2,math.pi/2]),
                      'left_hip_Z': (3, [-0.8,1.57]),
                      'left_knee_Y': (4, [0, math.pi]),
                      'left_ankle_X': (5 ,[-0.72, 0.72]),
                      'left_ankle_Y': (6, [-math.pi/2,math.pi/2]),
                      'left_ankle_Z': (7, [-0.54, 0.54]),
                      'left_toe': (8, [0,0]),
                      'left_heel': (9, [0,0]),

                      'right_hip_X': (11, [-2.5,1]),
                      'right_hip_Y': (12, [-math.pi/2,math.pi/2]),
                      'right_hip_Z': (13, [-1.57, 0.8]),
                      'right_knee_Y': (14, [0, math.pi]),
                      'right_ankle_X': (15, [-0.72, 0.72]),
                      'right_ankle_Y': (16, [-math.pi / 2, math.pi / 2]),
                      'right_ankle_Z': (17, [-0.54, 0.54]),
                      'right_toe': (18, [0, 0]),
                      'right_heel': (19, [0, 0]),

                      'neck_X': (34, [-1, 1]),
                      'neck_Y': (35, [-math.pi / 2, math.pi / 2]),
                      'neck_Z': (36, [-0.6, 0.6]),
                      }
        for key in max_min_dic.keys():
            index = max_min_dic[key][0]
            min_value = max_min_dic[key][1][0]
            max_value = max_min_dic[key][1][1]
            max_angles[:,index]=max_value
            min_angles[:,index]=min_value 
        return max_angles, min_angles

class InitializerConsistentHumanoid2(Initializer):
    def __init__(self,batch_size,target_joints):
        self.rbdl_joint_dic = {
                "base": 1,  "head": 42, "neck": 39, "left_hip": 3, "left_knee": 7, "left_ankle": 8, "left_heel": 11, "left_toe": 12,"right_hip": 13, "right_knee": 17, "right_ankle": 18,  "right_heel": 21,"right_toe": 22,
                "left_clavicle": 23, "left_shoulder": 25, "left_elbow": 28, "left_wrist": 30, "right_clavicle": 31, "right_shoulder": 33,  "right_elbow": 36, "right_wrist": 38
            }

        self.gt_mpi_dic = {
                "base": 4, "head": 7, "neck": 5, "left_hip": 18, "left_knee": 19, "left_ankle": 20, "left_toe": 22,
                "right_hip": 23,
                "right_knee": 24, "right_ankle": 25, "right_toe": 27, "left_clavicle": 8, "left_shoulder": 9, "left_elbow": 10,
                "left_wrist": 11, "right_clavicle": 13, "right_shoulder": 14, "right_elbow": 15, "right_wrist": 16
            }
        self.gt_human36M_dic={
            "base":0,"left_hip":6,"left_knee":7,"left_ankle":8, "left_toe":9,"right_hip":1,"right_knee":2,"right_ankle":3,"right_toe":4, "neck":13, "head":15,
            "left_shoulder":17,"left_elbow":18,"left_wrist":19,"right_shoulder":25, "right_elbow":26,"right_wrist":27
        }
        self.gt_DeepCap_dic = {
            "base": 14, "left_hip": 11, "left_knee": 12, "left_ankle": 13, "left_toe": 16, "right_hip": 8,
            "right_knee": 9, "right_ankle": 10, "right_toe": 15, "neck": 1, "head": 0,
            "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7, "right_shoulder": 2, "right_elbow": 3,
            "right_wrist": 4
        }
        self.target_joints = target_joints
        self.target_joint_ids = [self.rbdl_joint_dic[key] for key in self.target_joints]
        #self.target_marker_ids = [self.rbdl_marker_dic[key] for key in self.target_joints]
        self.scaler= 1000
        self.batch_size=batch_size

        """ caution!!!! this might be opposite!"""
        self.bullet2rbdl  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,36,37,38,39,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]
        self.rbdl2bullet  = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,20,21,22,23]

    def get_max_min_anlges(self):
        max_angles=math.pi*torch.ones(self.batch_size,len(self.bullet2rbdl))
        min_angles=-math.pi*torch.ones(self.batch_size,len(self.bullet2rbdl))
        max_min_dic ={
                        'left_hip_X': (1, [-2.5, 1]),
                        'left_hip_Y': (2,[-math.pi/2,math.pi/2]),
                        'left_hip_Z': (3, [-0.8,1.57]),
                        'left_knee_Y': (4, [0, math.pi]),
                        'left_ankle_X': (5 ,[-0.72, 0.72]),
                        'left_ankle_Y': (6, [-math.pi/2,math.pi/2]),
                        'left_ankle_Z': (7, [-0.54, 0.54]),
                        'left_toe': (8, [0,0]),
                        'left_heel': (9, [0,0]),

                        'right_hip_X': (11, [-2.5,1]),
                        'right_hip_Y': (12, [-math.pi/2,math.pi/2]),
                        'right_hip_Z': (13, [-1.57, 0.8]),
                        'right_knee_Y': (14, [0, math.pi]),
                        'right_ankle_X': (15, [-0.72, 0.72]),
                        'right_ankle_Y': (16, [-math.pi / 2, math.pi / 2]),
                        'right_ankle_Z': (17, [-0.54, 0.54]),
                        'right_toe': (18, [0, 0]),
                        'right_heel': (19, [0, 0]),


                        'left_clavicle_ry': (20, [-0.37,0.37]),
                        'left_clavicle_rz': (21, [-2.2,0.6]),
                        'left_shoulder_rx': (22, [-1.59,1.59]),
                        'left_shoulder_ry': (23, [-1.6,0.66]),
                        'left_shoulder_rz': (24, [-2.2,0.4]),
                        'left_elbow_rx': (25, [-0.3,0.3]),
                        'left_elbow_ry' : (26, [-2.8,0.0]),

                        'right_clavicle_ry': (28, [-0.37,0.37]),
                        'right_clavicle_rz': (29, [-0.6,2.2]),
                        'right_shoulder_rx': (30, [-1.59,1.59]),
                        'right_shoulder_ry': (31, [-0.66,1.6]),
                        'right_shoulder_rz': (32, [-0.4,2.2]),
                        'right_elbow_rx': (33, [-0.3,0.3]),
                        'right_elbow_ry' : (34, [0.0,2.8]),

                        'neck_X': (36, [-0.4, 0.4]), #'neck_X': (34, [-0.4, 0.4]),
                        'neck_Y': (37, [-1.1, 1.1]),#'neck_Y': (35, [-1.1, 1.1]),
                        'neck_Z': (38, [-0.36, 0.36]),#'neck_Z': (36, [-0.36, 0.36]),
                      }
        for key in max_min_dic.keys():
            index = max_min_dic[key][0]
            min_value = max_min_dic[key][1][0]
            max_value = max_min_dic[key][1][1]
            max_angles[:,index]=max_value
            min_angles[:,index]=min_value
        return max_angles, min_angles