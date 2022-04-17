import sys
import os
import torch
import cvxpy as cp
sys.path.append("../util")
from Utils.angles import angle_util
import rbdl
from cvxpylayers.torch import CvxpyLayer
AU = angle_util()
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class CoreUtils():
    def __init__(self,q_size,delta_t):
        n, m  = q_size, q_size
        Q_sqrt = cp.Parameter((n, n))
        q = cp.Parameter(n)
        G = cp.Parameter((6, n))
        h = cp.Parameter(6)  # 6 + 43)
        x = cp.Variable(n)
        obj = cp.Minimize(1e-10*cp.norm(x, p=1) + 0.5 * cp.sum_squares(Q_sqrt * x) + q.T @ x)
        cons = [G @ x >= h]  # , G @ x <= h]
        prob = cp.Problem(obj, cons)
        self.layer = CvxpyLayer(prob, parameters=[Q_sqrt, q, G, h], variables=[x]) #, G, h
        self.q_size=q_size
        self.selecton = torch.zeros(q_size, q_size)#.cuda()
        remain_id = np.arange(6, q_size)
        #print(remain_id)
        for id in remain_id:
            self.selecton[id][id] = 1
        self.selecton[0][0] = 1
        self.selecton[1][1] = 1
        self.selecton[2][2] = 1
        L = torch.mm(self.selecton, delta_t * torch.eye(q_size))#.cuda())
        self.LTL = torch.mm(L.T, L)
        self.delta_t_eye = delta_t * torch.eye(q_size)##.cuda()
        self.label_mat_base=torch.eye(6)##.cuda().view(2,3,6)
 
        self.label_mat_base_cpu=torch.eye(6).view(2,3,6) 

    def get_contact_jacobis6D(self,model, q, ids):
        jacobis = np.zeros((len(q), len(ids), 6, model.qdot_size))  
        for batch in range(len(q)):
            for i, id in enumerate(ids):
                rbdl.CalcPointJacobian6D(model, q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
        jacobis = torch.FloatTensor(jacobis).view(len(q), -1, model.qdot_size)#.cuda()
        return jacobis
    def get_contact_jacobis6D_cpu(self,model, q, ids):
        jacobis = np.zeros((len(q), len(ids), 6, model.qdot_size)) 
        for batch in range(len(q)):
            for i, id in enumerate(ids):
                rbdl.CalcPointJacobian6D(model, q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
        jacobis = torch.FloatTensor(jacobis).view(len(q), -1, model.qdot_size)##.cuda()
        return jacobis

    def get_predictions_Kinematics_exf(self,TempConvNetArt,TempConvNetOri,TempConvNetTrans,input1,q0_ref,q0_ref_ref,depth0,external_force):

        n_b,_= q0_ref.shape
        ref_art = TempConvNetArt(q0_ref[:, 3:], input1, external_force)
        #ref_quat = TempConvNetOri(q0_ref[:, 3:], input1, external_force)
        ref_quat = TempConvNetOri(q0_ref [:, 3:], input1, external_force)
        #print(depth0.shape,input1.shape,external_force.shape)
        ref_trans = TempConvNetTrans(depth0, input1, external_force)
        ref_trans = torch.clamp(ref_trans, -5, 5)
        l = torch.sqrt(ref_quat.pow(2).sum(1)).view(ref_quat.shape[0], 1)
        ref_quat = ref_quat / l 
        q_ref = torch.cat((ref_trans, ref_quat[:, 1:], ref_art, ref_quat[:, 0].view(-1, 1)), 1)
        ref_trans = q_ref[:, :3].view(n_b, 3)  # #.cuda()
        return q_ref,ref_trans,ref_quat,ref_art
 
    def get_PD_errors(self,ref_quat,quat0,ref_trans,trans0,ref_art,art0): 
        errors_ori=self.get_ori_errors(ref_quat,quat0)
        errors_trans=self.get_trans_errors(ref_trans,trans0)
        errors_art=self.get_art_errors(ref_art,art0)
        return errors_trans,errors_ori,errors_art

    def get_PD_errors_cpu(self,ref_quat,quat0,ref_trans,trans0,ref_art,art0): 
        errors_ori=self.get_ori_errors_cpu(ref_quat,quat0)
        errors_trans=self.get_trans_errors(ref_trans,trans0)
        errors_art=self.get_art_errors_cpu(ref_art,art0)
        return errors_trans,errors_ori,errors_art
    def get_ori_errors_cpu(self,ref_quat,quat0):
        errors_ori = AU.qmul(ref_quat, AU.q_conj(quat0))
        errors_ori = AU.quat_shortest_path_cpu(errors_ori)
        l = torch.sqrt(errors_ori.pow(2).sum(1)).view(errors_ori.shape[0], 1)
        errors_ori = errors_ori / l
        errors_ori = errors_ori[:, 1:]
        return errors_ori
    def get_ori_errors(self,ref_quat,quat0):
        errors_ori = AU.qmul(ref_quat, AU.q_conj(quat0))
        errors_ori = AU.quat_shortest_path(errors_ori)
        l = torch.sqrt(errors_ori.pow(2).sum(1)).view(errors_ori.shape[0], 1)
        errors_ori = errors_ori / l
        errors_ori = errors_ori[:, 1:]
        return errors_ori

    def get_trans_errors(self,ref_trans,trans0):
        return ref_trans - trans0

    def get_art_errors(self,ref_art,art0):
        return AU.target_correction_batch(ref_art -art0)

    def get_art_errors_cpu(self,ref_art,art0):
        return AU.target_correction_batch_cpu(ref_art -art0)
  
    def get_tau(self,errors_trans,errors_ori,errors_art,qdot0,limit, small_z=1):
        tau_art = 324 * errors_art - 20 * qdot0[:, 6:]
        tau_ori = 6000 * errors_ori - 536 * qdot0[:, 3:6]
        tau_trans = 15000 * errors_trans - 678 * qdot0[:, :3]
        #tau = torch.cat((tau_trans, tau_ori, tau_art), 1)
        if small_z:
            tau_trans[:,2] = 3000 * errors_trans[:,2] - 2000 * qdot0[:,2]
        tau_art = torch.clamp(tau_art, -limit, limit)
        tau_ori = torch.clamp(tau_ori, -200, 200)
        tau_trans = torch.clamp(tau_trans, -400, 400)
        tau = torch.cat((tau_trans,tau_ori,tau_art),1)
        tau[:, 6 + 9] = 0
        tau[:, 6 + 8] = 0
        tau[:, 6 + 18] = 0
        tau[:, 6 + 19] = 0
        #tau = torch.clamp(tau, -25, 25)
        return tau
 

    def get_neural_development(self, errors_trans, errors_ori, errors_art, qdot0,gains, offset,limit_art,small_z = 0):
 
        tau_art = 324 * gains[:,6:]* errors_art - 20 * qdot0[:, 6:]+ offset[:,6:]
        tau_ori = 6000 * gains[:,3:6]*errors_ori - 536 * qdot0[:, 3:6]+ offset[:,3:6]
        #tau_ori = 6000 * errors_ori - 536 * qdot0[:, 3:6]#+ offset[:,3:6]
        tau_trans = 15000 * gains[:,:3]* errors_trans - 678 * qdot0[:, :3]+ offset[:,:3]
        #tau_trans = 15000 * errors_trans - 678 * qdot0[:, :3]
        if small_z:
            tau_trans[:,2] = 3000 * errors_trans[:,2] - 2000 * qdot0[:,2]
        #tau = torch.cat((tau_trans, tau_ori, tau_art), 1)

        tau_art = torch.clamp(tau_art, -limit_art, limit_art) 
        tau_ori = torch.clamp(tau_ori, -200, 200)
        tau_trans = torch.clamp(tau_trans, -400, 400)

        tau = torch.cat((tau_trans,tau_ori,tau_art),1)
        tau[:, 6 + 9] = 0
        tau[:, 6 + 8] = 0
        tau[:, 6 + 18] = 0
        tau[:, 6 + 19] = 0
        return tau

    def get_neural_development_cpu(self, errors_trans, errors_ori, errors_art, qdot0,gains, offset,limit_art,art_only=0,small_z = 0):
 
        tau_art = 324 * gains[:,6:]* errors_art - 20 * qdot0[:, 6:]+ offset[:,6:]
        if art_only:
            tau_ori = 6000 * errors_ori - 536 * qdot0[:, 3:6]
            tau_trans = 15000 * errors_trans - 678 * qdot0[:, :3]
        else:
            tau_ori = 6000 * gains[:, 3:6] * errors_ori - 536 * qdot0[:, 3:6] + offset[:, 3:6]
            tau_trans = 15000 * gains[:,:3]* errors_trans - 678 * qdot0[:, :3]+ offset[:,:3]
        if small_z:
            tau_trans[:,2] = 3000 * errors_trans[:,2] - 2000 * qdot0[:,2]
        #tau = torch.cat((tau_trans, tau_ori, tau_art), 1)

        tau_art = torch.clamp(tau_art, -limit_art, limit_art) 
        tau_ori = torch.clamp(tau_ori, -200, 200)
        tau_trans = torch.clamp(tau_trans, -400, 400)

        tau = torch.cat((tau_trans,tau_ori,tau_art),1)
        tau[:, 6 + 9] = 0
        tau[:, 6 + 8] = 0
        tau[:, 6 + 18] = 0
        tau[:, 6 + 19] = 0
        return tau 
 
    def pose_update_quat(self,qdot0,q0,quat0,delta_t,qddot,speed_limit,th_zero = 0):
        n_b,_=q0.shape
        qdot = qdot0 + delta_t * qddot
        qdot = torch.clamp(qdot, -speed_limit, speed_limit)
        #print(qdot)
        if th_zero:
            qdot[:,6+9]=0
            qdot[:,6+8]=0
            qdot[:,6+18]=0
            qdot[:,6+19]=0
            qdot[:,-1]=0
        q_trans = q0[:, :3] + delta_t * qdot[:, :3]
        q_trans= torch.clamp(q_trans, -50, 50)
        q_art = q0[:, 6:-1] + delta_t * qdot[:, 6:]

        if th_zero:
            q_art[:,9]=0
            q_art[:,8]=0
            q_art[:,18]=0
            q_art[:,19]=0

        angvel_quat = AU.get_angvel_quat(qdot[:, 3:6])
        quat = quat0.detach() + delta_t * AU.qmul(angvel_quat, quat0.detach()) / 2
        loss_unit = (quat.pow(2).sum(1) - 1).pow(2).mean()

        l = torch.sqrt(quat.pow(2).sum(1)).view(n_b, 1)
        quat = quat / l
        q = torch.cat((q_trans, quat[:, 1:], q_art, quat[:, 0].view(-1, 1)), 1)
        return quat,q,qdot,loss_unit

    def pose_update_quat_cpu(self,qdot0,q0,quat0,delta_t,qddot,speed_limit,th_zero = 0):
        n_b, _ = q0.shape
        qdot = qdot0 + delta_t * qddot
        qdot = torch.clamp(qdot, -speed_limit, speed_limit)
        # print(qdot)
        if th_zero:
            qdot[:, 6 + 9] = 0
            qdot[:, 6 + 8] = 0
            qdot[:, 6 + 18] = 0
            qdot[:, 6 + 19] = 0
            qdot[:, -1] = 0
        q_trans = q0[:, :3] + delta_t * qdot[:, :3]
        q_trans = torch.clamp(q_trans, -50, 50)
        q_art = q0[:, 6:-1] + delta_t * qdot[:, 6:]

        if th_zero:
            q_art[:, 9] = 0
            q_art[:, 8] = 0
            q_art[:, 18] = 0
            q_art[:, 19] = 0

        angvel_quat = AU.get_angvel_quat_cpu(qdot[:, 3:6])
        quat = quat0.detach() + delta_t * AU.qmul(angvel_quat, quat0.detach()) / 2 

        l = torch.sqrt(quat.pow(2).sum(1)).view(n_b, 1)
        quat = quat / l
        q = torch.cat((q_trans, quat[:, 1:], q_art, quat[:, 0].view(-1, 1)), 1)
        return quat, q, qdot, _

    def pose_update_quat_debug(self,qdot0,q0,quat0,delta_t,qddot,th_zero = 0):
        n_b,_=q0.shape
        qdot = qdot0 + delta_t * qddot

        if th_zero:
            qdot[:,6+9]=0
            qdot[:,6+8]=0
            qdot[:,6+18]=0
            qdot[:,6+19]=0

        q_trans = q0[:, :3] + delta_t * qdot[:, :3]
        q_art = q0[:, 6:-1] + delta_t * qdot[:, 6:]

        if th_zero:
            q_art[:,9]=0
            q_art[:,8]=0
            q_art[:,18]=0
            q_art[:,19]=0

        angvel_quat = AU.get_angvel_quat(qdot[:, 3:6])
        quat = quat0.detach() + delta_t * AU.qmul(angvel_quat, quat0.detach()) / 2
        loss_unit = (quat.pow(2).sum(1) - 1).pow(2).mean()

        l = torch.sqrt(quat.pow(2).sum(1)).view(n_b, 1)
        quat = quat / l
        q = torch.cat((q_trans, quat[:, 1:], q_art, quat[:, 0].view(-1, 1)), 1)
        return quat,q,qdot,loss_unit

    def foot_constraint(self,J,q0, qdot,label_mat,Rs):
        n_b,_=q0.shape
        E = torch.eye(43) .view(1,43,43)#.cuda()
        E = E.expand(n_b,-1,-1 )
        selection = self.selecton.view(1,43,43).expand(n_b,-1,-1 )
        delta_t_eye = self.delta_t_eye.view(1,43,43).expand(n_b,-1,-1 )


        #LTL = LTL.view(1,)
        # Q = E#torch.bmm(torch.transpose(J,1,2),J)+ 0.8*self.LTL+E#torch.eye(43).cuda()
        #k = torch.bmm( selection,(q0 -q_t )[:,:-1].view(n_b,-1,1))
        # P =   - qdot.view(n_b,1,-1)#torch.bmm(k.view(n_b,1,-1),delta_t_eye)
        qval =- qdot.view(n_b,-1) #Variable(P.view(n_b,-1), requires_grad=True).float().cuda()
        Q_sqrtval = E#Q #Variable(Q, requires_grad=True).cuda()  # .view(1, 43, 43)

        #select_J = torch.mm(A_select_mat, J)
        #select_J=torch.zeros(3,3).cuda()
        #select_J[0][0]=1
        #select_J[1][1]=1
        #select_J[2][2]=1
        #select_J_top=torch.cat((select_J,torch.zeros(3,3).cuda()),1)
        #select_J_bottom=torch.cat(( torch.zeros(3,3).cuda(),select_J),1)
        #select_J=torch.cat((select_J_top,select_J_bottom),0)
        #Aval = torch.bmm(label_mat,J) #torch.mm(select_J,J)# torch.cat((select_J, M1), 0)
        #zeros = torch.zeros(6, requires_grad=True).cuda()
        #bval = torch.zeros(n_b,6).cuda()
        #bval[2] = -0.1#*torch.ones(6).cuda()#torch.cat((zeros.view(-1, 1), Mqdot), 0).view(-1)
        #bval[5] = -0.1
        Rtranspose = torch.transpose(Rs,1,2)

        Rtranspose_mat_top = torch.cat((Rtranspose, torch.zeros(n_b,3, 3) ), 2)#.cuda()
        Rtranspose_mat_bottom = torch.cat((torch.zeros(n_b,3, 3) ,Rtranspose), 2)#.cuda()
        Rtranspose_mat = torch.cat((Rtranspose_mat_top, Rtranspose_mat_bottom), 1).view(n_b, 6, 6)
        #Rtranspose_mat = Rtranspose_mat.expand(n_b,-1,-1)
        Gval = torch.bmm(Rtranspose_mat, J)
        #print(Gval[0])
        Gval[:, 0] = 0
        Gval[:, 2] = 0
        Gval[:, 3] = 0
        Gval[:, 5] = 0


        Gval = torch.bmm(label_mat, Gval)
        #for l in Gval.detach().cpu().numpy():
        #    print(list(l))
        #print(Gval)
        hval = torch.zeros(n_b, 6)#.cuda()
        y, = self.layer(Q_sqrtval, qval, Gval, hval)
        return y

    def foot_constraint_cpu(self,J,q0, qdot,label_mat,Rs):
        n_b,_=q0.shape
        E = torch.eye(46).view(1,46,46)
        E = E.expand(n_b,-1,-1 ) 
        qval =- qdot.view(n_b,-1) #Variable(P.view(n_b,-1), requires_grad=True).float().cuda()
        Q_sqrtval = E#Q #Variable(Q, requires_grad=True).cuda()  # .view(1, 43, 43)

        Rtranspose = torch.transpose(Rs,1,2)

        Rtranspose_mat_top = torch.cat((Rtranspose, torch.zeros(n_b,3, 3)), 2)
        Rtranspose_mat_bottom = torch.cat((torch.zeros(n_b,3, 3),Rtranspose), 2)
        Rtranspose_mat = torch.cat((Rtranspose_mat_top, Rtranspose_mat_bottom), 1).view(n_b, 6, 6)

        Gval = torch.bmm(Rtranspose_mat, J)

        Gval[:, 0] = 0
        Gval[:, 2] = 0
        Gval[:, 3] = 0
        Gval[:, 5] = 0

        Gval = torch.bmm(label_mat, Gval)
        hval = torch.zeros(n_b, 6)
        y, = self.layer(Q_sqrtval, qval, Gval, hval )

        return y

    def pose_update_quat_hard_debug(self,qdot0, q0, quat0, delta_t, qddot, th_zero=0): 
        n_b, _ = q0.shape
        qdot = qdot0 + delta_t * qddot 
        if th_zero:
            qdot[:, 6 + 9] = 0
            qdot[:, 6 + 8] = 0
            qdot[:, 6 + 18] = 0
            qdot[:, 6 + 19] = 0 
 
        q_trans = q0[:, :3] + delta_t * qdot[:, :3]
        q_art = q0[:, 6:-1] + delta_t * qdot[:, 6:]
        if th_zero:
            q_art[:, 9] = 0
            q_art[:, 8] = 0
            q_art[:, 18] = 0
            q_art[:, 19] = 0
        angvel_quat = AU.get_angvel_quat(qdot[:, 3:6])
        quat = quat0.detach() + delta_t * AU.qmul(angvel_quat, quat0.detach()) / 2
        loss_unit = (quat.pow(2).sum(1) - 1).pow(2).mean()

        l = torch.sqrt(quat.pow(2).sum(1)).view(n_b, 1)
        quat = quat / l
        q = torch.cat((q_trans, quat[:, 1:], q_art, quat[:, 0].view(-1, 1)), 1)
        return quat, q, qdot, loss_unit#, 0

    def pose_update_quat_hard(self,qdot0,q0,quat0,delta_t,qddot,J, lr_labels,Rs, th_zero=0):
        error_flag= 0
        n_b,_=q0.shape
        qdot = qdot0 + delta_t * qddot
        qdot = torch.clamp(qdot, -50, 50)
        if th_zero:
            qdot[:, 6 + 9] = 0
            qdot[:, 6 + 8] = 0
            qdot[:, 6 + 18] = 0
            qdot[:, 6 + 19] = 0 
        label_mat=self.label_mat_base.expand(n_b,-1,-1,-1)
        label_mat = lr_labels.view(n_b, 2, 1, 1) * label_mat
        label_mat =label_mat.reshape(n_b,6,6)

        try:
            qdot2 = self.foot_constraint(J,q0, qdot,label_mat,Rs)
 
        except:
            print("failed")
            qdot2=qdot.clone()
            error_flag=1
            pass

        q_trans = q0[:, :3] + delta_t * qdot2[:, :3]
        q_art   = q0[:, 6:-1] + delta_t * qdot2[:, 6:]
        if th_zero:
            q_art[:, 9] = 0
            q_art[:, 8] = 0
            q_art[:, 18] = 0
            q_art[:, 19] = 0
 
        angvel_quat = AU.get_angvel_quat(qdot2[:, 3:6])
        quat = quat0.detach() + delta_t * AU.qmul(angvel_quat, quat0.detach()) / 2
        loss_unit = (quat.pow(2).sum(1) - 1).pow(2).mean()

        l = torch.sqrt(quat.pow(2).sum(1)).view(n_b, 1)
        quat = quat / l
        q = torch.cat((q_trans, quat[:, 1:], q_art, quat[:, 0].view(-1, 1)), 1)
        if error_flag:
            return quat, q, qdot2, loss_unit, 1
        else:
            return quat,q,qdot2,loss_unit, 0
    def pose_update_quat_hard_cpu(self,qdot0,q0,quat0,delta_t,qddot,J, lr_labels,Rs, th_zero=0):
        error_flag= 0
        n_b,_=q0.shape

        qdot = qdot0 + delta_t * qddot
        qdot = torch.clamp(qdot, -5, 5)
        if th_zero:
            qdot[:, 6 + 9] = 0
            qdot[:, 6 + 8] = 0
            qdot[:, 6 + 18] = 0
            qdot[:, 6 + 19] = 0 

        label_mat=self.label_mat_base_cpu.expand(n_b,-1,-1,-1)
        label_mat = lr_labels.view(n_b, 2, 1, 1) * label_mat
        label_mat =label_mat.reshape(n_b,6,6)

        try:
            qdot2 = self.foot_constraint_cpu(J,q0, qdot,label_mat,Rs)
            qdot2 = torch.clamp(qdot2, -5, 5)
 

        except:
            print("failed")
            qdot2=qdot.clone()
            error_flag=1
            pass

        q_trans = q0[:, :3] + delta_t * qdot2[:, :3]
        q_art   = q0[:, 6:-1] + delta_t * qdot2[:, 6:]
        if th_zero:
            q_art[:, 9] = 0
            q_art[:, 8] = 0
            q_art[:, 18] = 0
            q_art[:, 19] = 0
 
        angvel_quat = AU.get_angvel_quat_cpu(qdot2[:, 3:6])
        quat = quat0.detach() + delta_t * AU.qmul(angvel_quat, quat0.detach()) / 2
        loss_unit = (quat.pow(2).sum(1) - 1).pow(2).mean()

        l = torch.sqrt(quat.pow(2).sum(1)).view(n_b, 1)
        quat = quat / l
        q = torch.cat((q_trans, quat[:, 1:], q_art, quat[:, 0].view(-1, 1)), 1)
        if error_flag:
            return quat, q, qdot2, loss_unit, 1
        else:
            return quat,q,qdot2,loss_unit, 0
 
    def process_contact_info(self,labels):
        """
        labels:Bx4
        """
        left_flag = labels[:,0]+labels[:,1]
        left_flag[left_flag > 0] = 1
        right_flag = labels[:,2]+labels[:,3]
        right_flag[right_flag > 0] = 1

        return torch.cat((left_flag.view(-1,1),right_flag.view(-1,1)),1)

    def check_contact_presence(self,labels):
        """
        labels:Bx4
        """
        flags = labels[:,0]+labels[:,1]+labels[:,2]+labels[:,3]
        flags[flags > 0] = 1

        return flags