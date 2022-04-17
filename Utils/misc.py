import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import pybullet as p
import math
import sys 
import rbdl
sys.path.append("./util")
from Utils.angles import angle_util
import copy
AU=angle_util()

def valid_flags_by_vnect_accuracy(th_error,th_count,data,gt):
    N, T, J, _ = gt.shape
    gt = gt.view(N, T, -1)
    data = data.view(N, T, -1)
    residual = gt - data
    norms = torch.norm(residual, dim=2) 
    labels = torch.zeros(norms.shape)
    higher_id = torch.nonzero(norms > th_error) 
    labels[higher_id[:, 0], higher_id[:, 1]] = 1

    labels_sequence_wise = torch.sum(labels, 1)
    higher_id_sequence_wise = torch.nonzero(labels_sequence_wise > th_count)
    valid_labels = torch.ones(labels_sequence_wise.shape)
    valid_labels[higher_id_sequence_wise[:, 0]] = 0
    return valid_labels

def valid_flags_by_frameout(gt_2Ds):
    tmp = gt_2Ds.clone()

    tmp[tmp > 1] = 100
    tmp[tmp < 0] = 100

    tmp[tmp != 100] = 0
    tmp[tmp == 100] = 1

    labels = tmp.sum(1).sum(1).sum(1)

    final_labels = torch.zeros(labels.shape)
    for i in range(len(final_labels)):
        if labels[i] == 0:
            final_labels[i]=1

    return final_labels#valid_indices

def valid_flags(th_error,th_count,data, gt):
    """
    th_error: threshold value of the projection error
    th_count: max # frames that exceeds the th_error
    data: predicted 2D keypoints (keypoints normalized between 0 and 1)
    gt  : GT 2D keypoints (keypoints normalized between 0 and 1)
    """

    valid_flags_vnect_acc = valid_flags_by_vnect_accuracy(th_error,th_count,data,gt)
    valid_flags_frameout = valid_flags_by_frameout(gt)
    final_labels = valid_flags_vnect_acc*valid_flags_frameout
    valid_indices = torch.FloatTensor(np.array([x for x in np.arange(len(final_labels)) if final_labels[x] == 1]))
    return valid_indices.numpy().astype(int)

def img_size_getter(sub_ids):
    labels = sub_ids.clone()

    DC_id =torch.nonzero(labels < 1)
    HM_id =torch.nonzero((labels < 20)*(labels>0))
    MPI_id =torch.nonzero(labels > 20)
    WI_id = torch.nonzero(labels >= 30)
    labels[DC_id[:, 0] ]=1024
    labels[HM_id[:, 0] ]=1000
    labels[MPI_id[:, 0] ]=2048
    labels[WI_id[:, 0]] = 1280
    return labels
def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    num_rows, num_cols = A.shape;

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape;
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1).reshape(3, 1)
    centroid_B = np.mean(B, axis=1).reshape(3, 1)
    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    # dot is matrix multiplication for array
    H = Am.dot(np.transpose(Bm))

    # sanity check
    if np.linalg.matrix_rank(H) < 3:
        raise ValueError("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A + centroid_B

    return R, t
def mat2zerodiag(data,K):
    """
    Input: data, K
    data BxTxC

    return BxTKxCK with zeros
    """
    B,T,C = data.shape
    data_pad = torch.zeros( B, T *K, C * K)###.cuda()
    for i in range(K): 
        data_pad[:, i * T:(i + 1) * T, i * C:(i + 1) * C] = data
    return data_pad


def reverse_labels(labels):
    labels += 1
    labels[labels==2]=0
    labels[labels==1]=1
    return labels

def reverse_labels_np(labels):
    reversed_labels = copy.copy(labels)
    reversed_labels += 1
    reversed_labels[reversed_labels==2]=0
    reversed_labels[reversed_labels==1]=1
    return reversed_labels
def get_WILD_labels(sub_ids):
    wild_labels = copy.copy(sub_ids)
    wild_labels[wild_labels<30]=0
    wild_labels[wild_labels==30]=1
    non_wild_labels = reverse_labels_np(wild_labels)
    return wild_labels,non_wild_labels
def pysinc(x):
    return torch.sin(math.pi*x)/(math.pi*x)

def exp2quat(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3 
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3) 
    theta = torch.norm(e,dim=1).view(-1, 1) 
    w = torch.cos(0.5 * theta).view(-1, 1) 
    xyz = 0.5 * pysinc(0.5 * theta / math.pi) * e
    return torch.cat((w, xyz), 1).view(original_shape)

def quat2exp(quat):
    half_theta = torch.acos(quat[:,0].view(-1,1))
    exp = (2 / torch.sin(half_theta)) * half_theta * torch.FloatTensor( [quat[:,1], quat[:,2], quat[:,3]]).view(-1, 3)###.cuda()

    return exp
def get_q_axis_from_quat_bullet(q):
    quat = q[:, 3:7]
    quat = AU.quat_bullet2general_b(quat)
    exp = quat2exp(quat)
    new_q = torch.cat((q[:,:3],exp.view(-1,3),q[:,7:]),1)
    return new_q

def get_q_axis_from_quat_rbdl(q):
    quat = torch.cat((q[:, -1].view(-1,1), q[:, 3:6]),1)
    exp = quat2exp(quat)
    new_q = torch.cat((q[:,:3],exp.view(-1,3),q[:,6:-1]),1)
    return new_q

def get_q_quat_from_axis_rbdl(q):
    exp=q[:,3:6]
    quat =  AU.expmap_to_quaternion(exp)
    new_q = torch.cat((q[:, :3], quat[:,1:].view(-1, 3), q[:, 6:],quat[:,0].view(-1,1)), 1)
    return  new_q

def get_mass_mat(model, q): 
    n_b, _ = q.shape
    M_np = np.zeros((n_b, model.qdot_size, model.qdot_size))
    [rbdl.CompositeRigidBodyAlgorithm(model, q[i].astype(float), M_np[i]) for i in range(n_b)]

    return torch.FloatTensor(M_np)##.cuda()
def get_mass_mat_cpu(model, q):
 
    n_b, _ = q.shape
    M_np = np.zeros((n_b, model.qdot_size, model.qdot_size))
    [rbdl.CompositeRigidBodyAlgorithm(model, q[i].astype(float), M_np[i]) for i in range(n_b)]

    return torch.FloatTensor(M_np)###.cuda() 

def clean_massMat(M_inv):
    M_inv[:, 6] = 0

    M_inv[:, 6 + 8] = 0
    M_inv[:, 6 + 9] = 0
    M_inv[:, 6 + 10] = 0

    M_inv[:, 6 + 18] = 0
    M_inv[:, 6 + 19] = 0

    M_inv[:, 6 + 27] = 0
    M_inv[:, 6 + 35] = 0
    M_inv[:, -1] = 0

    M_inv[:, :, 6] = 0

    M_inv[:, :, 6 + 8] = 0
    M_inv[:, :, 6 + 9] = 0
    M_inv[:, :, 6 + 10] = 0

    M_inv[:, :, 6 + 18] = 0
    M_inv[:, :, 6 + 19] = 0

    M_inv[:, :, 6 + 27] = 0
    M_inv[:, :, 6 + 35] = 0
    M_inv[:, :, -1] = 0
    return  M_inv
def vec2zerodiag(data):
    """
    data BxTxC

    return BxTxTC with zeros
    """
    B,T,C = data.shape
    data = torch.diag_embed(data.contiguous().view(B, -1) , offset=0, dim1=-2, dim2=-1)
    data = data.view(B, -1, C, T*C).sum(2)

    return data



def motion_update(id_robot, jointIds, qs):
    [p.resetJointState(id_robot, jid, q) for jid, q in zip(jointIds, qs)]
    return 0
def visualization3D(id_robot,id_robot_ref,jointIds,rbdl2bullet,q,  target_qs):
    q = np.squeeze(q.detach().cpu().numpy())
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]

    motion_update(id_robot, jointIds_reordered , q[6:-1])
    motion_update(id_robot_ref, jointIds_reordered, target_qs[7:]) 

    p.resetBasePositionAndOrientation(id_robot, [q[0], q[1], q[2]], [q[3], q[4], q[5], q[-1]]) 

    p.resetBasePositionAndOrientation(id_robot_ref, [target_qs[0], target_qs[1], target_qs[2] ],  target_qs[3:7]) 
    p.stepSimulation()
    return 0
def visualization3D_single(id_robot, jointIds,rbdl2bullet,q,Rot,T ):
    q = np.squeeze(q.cpu().numpy())

    jointIds_reordered = np.array(jointIds)[rbdl2bullet]
    out_pos = np.array([q[0], q[1], q[2]])
    out_quat = np.array([q[3], q[4], q[5], q[-1]])
    r1 = R.from_quat(out_quat)
    mat1 = r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot.T, mat1))
    out_quat = r12.as_quat()
    out_pos = np.dot(Rot.T, out_pos - T)

    motion_update(id_robot, jointIds_reordered , q[6:-1])
    p.resetBasePositionAndOrientation(id_robot,out_pos,out_quat)
    p.stepSimulation()
    return 0

def visualization3D_double(id_robot, id_robot2, jointIds,rbdl2bullet,q ,q2,Rot,T ): 
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]

    out_pos = np.array([q[0], q[1], q[2]])
    out_quat = np.array([q[3], q[4], q[5], q[-1]])
    r1 = R.from_quat(out_quat)
    mat1 = r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot.T, mat1))
    out_quat = r12.as_quat()
    out_pos = np.dot(Rot.T, out_pos - T)

    out_pos2 = np.array([q2[0], q2[1], q2[2]])
    out_quat2 = np.array([q2[3], q2[4], q2[5], q2[-1]])
    r2 = R.from_quat(out_quat2)
    mat2 = r2.as_matrix()
    r22 = R.from_matrix(np.dot(Rot.T, mat2))
    out_quat2 = r22.as_quat()
    out_pos2 = np.dot(Rot.T, out_pos2 - T)
    motion_update(id_robot, jointIds_reordered , q[6:-1])
    p.resetBasePositionAndOrientation(id_robot, out_pos, out_quat)
    motion_update(id_robot2, jointIds_reordered , q2[6:-1])
    out_pos2[2]+=1#0.5
    p.resetBasePositionAndOrientation(id_robot2, out_pos2, out_quat2)

    p.stepSimulation()
    return 0

def visualization3D_double_HM(id_robot, id_robot2, jointIds,rbdl2bullet,q ,q2,Rot,T ): 
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]

    out_pos = np.array([q[0], q[1], q[2]])
    out_quat = np.array([q[3], q[4], q[5], q[-1]])
    r1 = R.from_quat(out_quat)
    mat1 = r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot, mat1))
    out_quat = r12.as_quat()
    out_pos = np.dot(Rot, out_pos) + T

    out_pos2 = np.array([q2[0], q2[1], q2[2]])
    out_quat2 = np.array([q2[3], q2[4], q2[5], q2[-1]])
    r2 = R.from_quat(out_quat2)
    mat2 = r2.as_matrix()
    r22 = R.from_matrix(np.dot(Rot, mat2))
    out_quat2 = r22.as_quat()
    out_pos2 = np.dot(Rot, out_pos2) + T
    motion_update(id_robot, jointIds_reordered , q[6:-1])
    p.resetBasePositionAndOrientation(id_robot, out_pos, out_quat)
    motion_update(id_robot2, jointIds_reordered , q2[6:-1])
    #out_pos2[2]+=0.5
    p.resetBasePositionAndOrientation(id_robot2, out_pos2, out_quat2)

    p.stepSimulation()
    return 0
def visualization3D_multiple(robot_ids, jointIds,rbdl2bullet,qs ,Rot=[],T=[] ,overlay=False,visu=False): 
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]
    if len(Rot)==0:
        Rot = np.eye(3)
        T=np.zeros(3)
    for i,(id_robot, q) in enumerate(zip(robot_ids,qs)):
 
        if visu:

            out_pos, out_quat = get_transformed_root_visualisation(np.array([q[0], q[1], q[2]]), np.array([q[3], q[4], q[5], q[-1]]), Rot,T)
        else:

            out_pos ,out_quat =get_transformed_root( np.array([q[0], q[1], q[2] ]), np.array([q[3], q[4], q[5], q[-1]]), Rot,T)

        motion_update(id_robot, jointIds_reordered , q[6:-1])
        if not overlay:
            out_pos[2]+=(out_pos[2]+i )
        p.resetBasePositionAndOrientation(id_robot, out_pos, out_quat)
    p.stepSimulation()
    return 0

def visualization3D_multiple_invTrans(robot_ids, jointIds,rbdl2bullet,qs ,Rot,T ,overlay=False): 
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]
    for i,(id_robot, q) in enumerate(zip(robot_ids,qs)):
 
        out_pos ,out_quat =get_transformed_root_inv( np.array([q[0], q[1], q[2] ]), np.array([q[3], q[4], q[5], q[-1]]), Rot,T)
        motion_update(id_robot, jointIds_reordered , q[6:-1])
        if not overlay:
            out_pos[0]+=(out_pos[0]+i )
        p.resetBasePositionAndOrientation(id_robot, out_pos, out_quat)
    p.stepSimulation()
    return 0
def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)
def get_transformed_root(pos,ori,Rot,T):
    r1 = R.from_quat(ori)
    mat1 = r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot.T, mat1))
    ori = r12.as_quat()
    pos = np.dot(Rot.T, pos - T)
    return pos,ori

def get_transformed_root_visualisation(pos,ori,Rot,T):
    r_adjust = R.from_euler('xyz', [-0.2, 0, 0])
    R2 = r_adjust.as_matrix()



    r1 = R.from_quat(ori)
    mat1 = r1.as_matrix()
    mat1 = np.dot(R2, mat1) 
    mat1 = np.dot(Rot.T, mat1) 
    r12 = R.from_matrix(mat1)
    ori = r12.as_quat()
    pos = np.dot(Rot.T, pos - T)
    return pos,ori

def get_transformed_root_inv(pos,ori,Rot,T): 
    r1 = R.from_quat(ori)
    mat1 = r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot, mat1))
    ori = r12.as_quat()
    pos = np.dot(Rot, pos).reshape(3,1) + T.reshape(3,1)
    return pos,ori

def visualization3D_triple(id_robot, id_robot2, id_robot3, jointIds,rbdl2bullet,q ,q2,q3,Rot,T ): 
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]
    out_pos,out_quat=get_transformed_root( np.array([q[0], q[1], q[2]]), np.array([q[3], q[4], q[5], q[-1]]), Rot,T)
    out_pos2,out_quat2=get_transformed_root( np.array([q2[0], q2[1], q2[2]]), np.array([q2[3], q2[4], q2[5], q2[-1]]), Rot,T)
    out_pos3,out_quat3=get_transformed_root( np.array([q3[0], q3[1], q3[2]]), np.array([q3[3], q3[4], q3[5], q3[-1]]), Rot,T)

    """
    out_quat = np.array([q[3], q[4], q[5], q[-1]])
    r1 = R.from_quat(out_quat)
    mat1 = r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot.T, mat1))
    out_quat = r12.as_quat()
    out_pos = np.dot(Rot.T, out_pos - T)

    out_pos2 = np.array([q2[0], q2[1], q2[2]])
    out_quat2 = np.array([q2[3], q2[4], q2[5], q2[-1]])
    r2 = R.from_quat(out_quat2)
    mat2 = r2.as_matrix()
    r22 = R.from_matrix(np.dot(Rot.T, mat2))
    out_quat2 = r22.as_quat()
    out_pos2 = np.dot(Rot.T, out_pos2 - T)
    """

    motion_update(id_robot, jointIds_reordered , q[6:-1])
    p.resetBasePositionAndOrientation(id_robot, out_pos, out_quat)
    motion_update(id_robot2, jointIds_reordered , q2[6:-1])
    p.resetBasePositionAndOrientation(id_robot2, out_pos2, out_quat2)
    motion_update(id_robot3, jointIds_reordered , q3[6:-1])
    p.resetBasePositionAndOrientation(id_robot3, out_pos3, out_quat3)
    p.stepSimulation()
    return 0
def visualization3D_bb(id_robot,id_robot_ref,jointIds,rbdl2bullet,q,  target_qs):
    q = np.squeeze(q.detach().cpu().numpy())
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]

    motion_update(id_robot, jointIds_reordered , q[6:-1])
    motion_update(id_robot_ref, jointIds_reordered, target_qs[6:-1]) 

    p.resetBasePositionAndOrientation(id_robot, [q[0], q[1], q[2]], [q[3], q[4], q[5], q[-1]]) 
    p.resetBasePositionAndOrientation(id_robot_ref, [target_qs[0], target_qs[1], target_qs[2]], [target_qs[3], target_qs[4], target_qs[5], target_qs[-1]]) 
    p.stepSimulation()
    return 0

def visualization3DCam2World_bb3(id_robot,id_robot_ref,id_robot_ref2,jointIds,rbdl2bullet,q,  target_qs, ref_qs, Rot, T):
    q = np.squeeze(q.detach().cpu().numpy())
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]
 
    motion_update(id_robot, jointIds_reordered , q[6:-1])
    motion_update(id_robot_ref, jointIds_reordered, target_qs[6:-1])
    motion_update(id_robot_ref2, jointIds_reordered, ref_qs[6:-1]) 
    out_pos = np.array([q[0], q[1], q[2]])
    out_quat = np.array([q[3], q[4], q[5], q[-1]])
    out_pos = np.dot(Rot.T, out_pos-T)
    gt_pos = np.array([target_qs[0], target_qs[1], target_qs[2] ])
    gt_pos = np.dot(Rot.T,gt_pos-T) 
    gt_quat= np.array([target_qs[3], target_qs[4], target_qs[5], target_qs[-1]])


    ref_pos = np.array([ref_qs[0], ref_qs[1], ref_qs[2] ])
    ref_pos = np.dot(Rot.T,ref_pos-T)
    ref_pos+=np.array([0,0,1])
    ref_quat= np.array([ref_qs[3], ref_qs[4], ref_qs[5], ref_qs[-1]])


    r1 = R.from_quat(out_quat)
    mat1=r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot.T,mat1))
    out_quat = r12.as_quat()

    r2 = R.from_quat(gt_quat)
    mat2=r2.as_matrix()
    r22 = R.from_matrix(np.dot(Rot.T,mat2))
    gt_quat = r22.as_quat()

    r3 = R.from_quat(ref_quat)
    mat3=r3.as_matrix()
    r32 = R.from_matrix(np.dot(Rot.T,mat3))
    ref_quat = r32.as_quat()

    p.resetBasePositionAndOrientation(id_robot, out_pos, out_quat) 
    p.resetBasePositionAndOrientation(id_robot_ref,gt_pos , gt_quat)
    p.resetBasePositionAndOrientation(id_robot_ref2,ref_pos , ref_quat) 
    p.stepSimulation()
    return 0

def visualization3DCam2World_bb(id_robot,id_robot_ref,jointIds,rbdl2bullet,q,  target_qs, Rot, T):
    q = np.squeeze(q.detach().cpu().numpy())
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]

    #quaternion=r.as_quaternion()
    #AU.qmul()
    motion_update(id_robot, jointIds_reordered , q[6:-1])
    motion_update(id_robot_ref, jointIds_reordered, target_qs[6:-1]) 
    out_pos = np.array([q[0], q[1], q[2]])
    out_quat = np.array([q[3], q[4], q[5], q[-1]])
    out_pos = np.dot(Rot.T, out_pos-T)
    gt_pos = np.array([target_qs[0], target_qs[1], target_qs[2] ])
    gt_pos = np.dot(Rot.T,gt_pos-T)
    #gt_pos+=np.array([0,0,1])
    gt_quat= np.array([target_qs[3], target_qs[4], target_qs[5], target_qs[-1]])

    r1 = R.from_quat(out_quat)
    mat1=r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot.T,mat1))
    out_quat = r12.as_quat()

    r2 = R.from_quat(gt_quat)
    mat2=r2.as_matrix()
    r22 = R.from_matrix(np.dot(Rot.T,mat2))
    gt_quat = r22.as_quat()

    p.resetBasePositionAndOrientation(id_robot, out_pos, out_quat) 

    p.resetBasePositionAndOrientation(id_robot_ref,gt_pos , gt_quat) 
    p.stepSimulation()
    return 0

def visualization3DCam2World_three(id_robot,id_robot_ref,id_robot_target,jointIds,rbdl2bullet,q,q_target,  target_qs, Rot, T):
    q = np.squeeze(q.detach().cpu().numpy())
    jointIds_reordered = np.array(jointIds)[rbdl2bullet] 
    motion_update(id_robot, jointIds_reordered , q[6:-1])
    motion_update(id_robot_target, jointIds_reordered , q_target[6:-1])
    motion_update(id_robot_ref, jointIds_reordered, target_qs[7:]) 
    out_pos = np.array([q[0], q[1], q[2]])
    out_quat = np.array([q[3], q[4], q[5], q[-1]])
    out_pos = np.dot(Rot.T,out_pos-T)

    target_pos = np.array([q_target[0], q_target[1], q_target[2]])
    target_quat = np.array([q_target[3], q_target[4], q_target[5], q_target[-1]])
    target_pos = np.dot(Rot.T,target_pos-T)
    gt_pos = np.array([target_qs[0], target_qs[1], target_qs[2] ])
    gt_pos = np.dot(Rot.T,gt_pos-T) 
    gt_quat= target_qs[3:7]

    r1 = R.from_quat(out_quat)
    mat1=r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot.T,mat1))
    out_quat = r12.as_quat()

    r2 = R.from_quat(gt_quat)
    mat2=r2.as_matrix()
    r22 = R.from_matrix(np.dot(Rot.T,mat2))
    gt_quat = r22.as_quat()

    r3 = R.from_quat(target_quat)
    mat3=r3.as_matrix()
    r32 = R.from_matrix(np.dot(Rot.T,mat3))
    target_quat = r32.as_quat()



    p.resetBasePositionAndOrientation(id_robot, out_pos, out_quat)

    p.resetBasePositionAndOrientation(id_robot_target, target_pos, target_quat) 
    p.resetBasePositionAndOrientation(id_robot_ref,gt_pos , gt_quat) 
    p.stepSimulation()
    return 0

def visualization3DCam2World(id_robot,id_robot_ref,jointIds,rbdl2bullet,q,  target_qs, Rot, T):
    q = np.squeeze(q.detach().cpu().numpy())
    jointIds_reordered = np.array(jointIds)[rbdl2bullet]
 
    motion_update(id_robot, jointIds_reordered , q[6:-1])
    motion_update(id_robot_ref, jointIds_reordered, target_qs[7:]) 
    out_pos = np.array([q[0], q[1], q[2]])
    out_quat = np.array([q[3], q[4], q[5], q[-1]])
    out_pos = np.dot(Rot.T,out_pos-T)
    gt_pos = np.array([target_qs[0], target_qs[1], target_qs[2] ])
    gt_pos = np.dot(Rot.T,gt_pos-T) 
    gt_quat= target_qs[3:7]

    r1 = R.from_quat(out_quat)
    mat1=r1.as_matrix()
    r12 = R.from_matrix(np.dot(Rot.T,mat1))
    out_quat = r12.as_quat()

    r2 = R.from_quat(gt_quat)
    mat2=r2.as_matrix()
    r22 = R.from_matrix(np.dot(Rot.T,mat2))
    gt_quat = r22.as_quat()

    p.resetBasePositionAndOrientation(id_robot, out_pos, out_quat) 

    p.resetBasePositionAndOrientation(id_robot_ref,gt_pos , gt_quat) 
    p.stepSimulation()
    return 0

def get_qs_with_euler(qs):
    qs_new = []
    for q in qs:
        quat = q[3:7]
        r = R.from_quat(quat)
        euler = r.as_euler("zyx")
        # quat = np.quaternion(quat[-1], quat[0], quat[1], quat[2])  # p.getEulerFromQuaternion(axis_ori)
        # rotvec = quaternion.as_rotation_vector(quat)
        root_config = np.concatenate((q[:3], euler), 0)
        qs_new.append(np.concatenate((root_config, q[7:]), 0))
    return np.array(qs_new)

def target_correction(error): 
    mask = torch.ones(error.shape[0])

    larger_id = torch.nonzero((error > math.pi) * mask).detach()
    smaller_id = torch.nonzero((error < - math.pi) * mask).detach()


    error[larger_id] = error[larger_id] - 2 * math.pi
    error[smaller_id] = error[smaller_id] + 2 * math.pi
    if len(larger_id)!=0 or len(larger_id)!=0:
        print("I'm in!!!!")
    return error

def get_jointIds_Names( id_robot):
    jointNamesAll = []
    jointIdsAll = []
    jointNames = []
    jointIds = []
    for j in range(p.getNumJoints(id_robot)):
        info = p.getJointInfo(id_robot, j)
        p.changeDynamics(id_robot, j, linearDamping=0, angularDamping=0)
        jointName = info[1]
        jointType = info[2]
        jointIdsAll.append(j)
        jointNamesAll.append(jointName)
        if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
            jointIds.append(j)
            jointNames.append(jointName)
    return jointIdsAll, jointNamesAll, jointIds, jointNames

