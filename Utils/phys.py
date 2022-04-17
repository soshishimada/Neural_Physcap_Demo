import torch
import rbdl
import numpy as np
from Utils.angles import angle_util
AU = angle_util()
 
class PyForwardKinematicsQuatSeemsWorking(torch.autograd.Function):
    def __init__(self, model_addresses, target_joint_ids,delta_t = 0.001):
        self.model_addresses = model_addresses

        self.qdot_size = model_addresses["1"].qdot_size
        self.target_joint_ids = target_joint_ids
        self.q = None
        self.delta_t = delta_t
        self.jacobi_inits = np.array([np.zeros((3, self.qdot_size)) for i in range(len(self.target_joint_ids))])

    # @staticmethod
    def forward(self,sub_ids, q):

        #self.q = q.cpu().data.numpy()
        #coords = torch.tensor([[rbdl.CalcBodyToBaseCoordinates(self.model, self.q[batch].flatten().astype(float), i, np.zeros(3)) for i in self.target_joint_ids] for batch in range(len(self.q))]).view(len(self.q), -1).float()

        self.q = q.cpu().data.numpy()
        self.models = [self.model_addresses[str(int(x))] for x in sub_ids.numpy()]
        coords = torch.tensor([[rbdl.CalcBodyToBaseCoordinates(self.models[batch], self.q[batch].flatten().astype(float), i, np.zeros(3)) for i in self.target_joint_ids] for batch in range(len(self.q))]).view(len(self.q), -1).float()
        coords = coords#.cuda()
        return coords

    # @staticmethod
    def backward(self, grad_coords):

        jacobis = np.zeros((len(self.q), len(self.target_joint_ids), 3, self.qdot_size))  # np.array([np.zeros((3, self.model.qdot_size)) for i in range(len(self.target_joint_ids))])
        for batch in range(len(self.q)):
            for i, id in enumerate(self.target_joint_ids):
                rbdl.CalcPointJacobian(self.models[batch], self.q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
        jacobis = torch.tensor(jacobis).view(len(self.q), -1, self.qdot_size).float()
        jacobis_root_ori = jacobis[:,:,3:6]

        quat = np.concatenate((self.q[:,3:6].flatten(),self.q[:,-1]),0)
        quat_inv=np.array([-quat[0],-quat[1],-quat[2],quat[3]])
        jac_angVel_quat = self.get_jacobi_anglularVel_quaternion(quat)
        jac_theta_angVelQuatForm=self.get_jacobi_theta_angVel(self.delta_t,order = "xyzw")


        quat_jac = torch.FloatTensor(np.dot(jacobis_root_ori,np.dot(jac_theta_angVelQuatForm,jac_angVel_quat)))

        final_jac=torch.cat((jacobis[:,:,:3],quat_jac[:,:,:3],jacobis[:,:,6:],quat_jac[:,:,-1].view(quat_jac.shape[0],quat_jac.shape[1],1)),2)#.cuda()
        #final_jac[:,:,6:-1]=0
        final_jac[:,:,:3]=0

    
        return None,torch.bmm(grad_coords.view(len(self.q), 1, -1), final_jac).view(len(self.q), -1)

    def get_jacobi_anglularVel_quaternion(self,quat):
        x=quat[0]
        y=quat[1]
        z=quat[2]
        w=quat[3]
        jac=(2/self.delta_t)*np.array([[w,z,-y,x],
                                  [-z,w,x,y],
                                  [y,-x,w,z],
                                  [-x,-y,-z,w]])

        return jac


    def get_jacobi_theta_angVel(self,delta_t,order):

        if order == "xyzw":
            jac = np.array([[delta_t,0,0,0],
                            [0, delta_t, 0, 0],
                            [0, 0, delta_t, 0],
                            ])
        elif order == "wxyz":
            jac = np.array([[0,delta_t,0,0],
                            [0, 0, delta_t, 0],
                            [0, 0, 0, delta_t],
                            ])
        else:
            print('not supported order of the quaternion')
        return jac


class PyForwardKinematicsExponentialMap(torch.autograd.Function):
    def __init__(self, model, target_joint_ids,delta_t=0.0001):
        self.model = model
        self.target_joint_ids = target_joint_ids
        self.q = None
        self.jacobi_inits = np.array([np.zeros((3, self.model.qdot_size)) for i in range(len(self.target_joint_ids))])
        self.delta_t=delta_t

    # @staticmethod
    def forward(self, q_exp):

        self.q = self.get_q_quat_from_axis_rbdl(q_exp)
        self.q = self.q.cpu().data.numpy() 

        coords = torch.tensor([[rbdl.CalcBodyToBaseCoordinates( self.model, self.q[batch].flatten().astype(float), i, np.zeros(3)) for i in self.target_joint_ids] for batch in range(len(self.q))]).view(len(self.q), -1).float()
        coords = coords#.cuda()
        return coords

    # @staticmethod
    def backward(self, grad_coords):

        jacobis = np.zeros((len(self.q), len(self.target_joint_ids), 3, self.model.qdot_size))  # np.array([np.zeros((3, self.model.qdot_size)) for i in range(len(self.target_joint_ids))])
        for batch in range(len(self.q)):
            for i, id in enumerate(self.target_joint_ids):
                rbdl.CalcPointJacobian(self.model, self.q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
        jacobis = torch.tensor(jacobis).view(len(self.q), -1, self.model.qdot_size).float()#.cuda()
        #print(jacobis.shape)
        jacobis[:,:,:3]=0
        jacobis[:,:,6:]=0



        """ 
        jacobis_root_ori = jacobis[:,:,3:6]


        quat = np.concatenate((self.q[:,3:6].flatten(),self.q[:,-1]),0)
        quat_inv = np.array([-quat[0],-quat[1],-quat[2],quat[3]])


        jac_angVel_quat = self.get_jacobi_anglularVel_quaternion(quat_inv)
        jac_theta_angVelQuatForm=self.get_jacobi_theta_angVel(self.delta_t,order = "xyzw")


        quat_jac = torch.FloatTensor(np.dot(jacobis_root_ori,np.dot(jac_theta_angVelQuatForm,jac_angVel_quat)))


        final_jac=torch.cat((jacobis[:,:,:3],quat_jac[:,:,:3],jacobis[:,:,6:],quat_jac[:,:,-1].view(quat_jac.shape[0],quat_jac.shape[1],1)),2).cuda()
        """
        return torch.bmm(grad_coords.view(len(self.q), 1, -1), jacobis).view(len(self.q), -1)

    def get_q_quat_from_axis_rbdl(self,q):
        exp = q[:, 3:6]
        quat = AU.expmap_to_quaternion(exp)
        new_q = torch.cat((q[:, :3], quat[:, 1:].view(-1, 3), q[:, 6:], quat[:, 0].view(-1, 1)), 1)
        return new_q


class PyForwardKinematicsQuaternionHuman36M(torch.autograd.Function):
 
    @staticmethod
    def forward(ctx, models,sub_ids, q): 
        ctx.q = q.cpu().data.numpy()
        ctx.models =  models 
        coords = torch.tensor([[rbdl.CalcBodyToBaseCoordinates(ctx.models[batch], ctx.q[batch].flatten().astype(float), i, np.zeros(3)) for i in ctx.target_joint_ids] for batch in range(len(self.q))]).view(len(self.q), -1).float()
        coords = coords#.cuda()
        return coords

    @staticmethod
    def backward(ctx, grad_coords):

        jacobis = np.zeros((len(ctx.q), len(ctx.target_joint_ids), 3, ctx.qdot_size))  
        for batch in range(len(ctx.q)):
            for i, id in enumerate(ctx.target_joint_ids):
                rbdl.CalcPointJacobian(ctx.models[batch], ctx.q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
        jacobis = torch.tensor(jacobis).view(len(ctx.q), -1,ctx.models[0].qdot_size).float()
        jacobis_root_ori = jacobis[:,:,3:6]


        quat = np.concatenate((ctx.q[:,3:6].flatten(),ctx.q[:,-1]),0)
        quat_inv = np.array([-quat[0],-quat[1],-quat[2],quat[3]])


        jac_angVel_quat =  PyForwardKinematicsQuaternionHuman36M.get_jacobi_anglularVel_quaternion(quat_inv)
        jac_theta_angVelQuatForm= PyForwardKinematicsQuaternionHuman36M.get_jacobi_theta_angVel(ctx.delta_t,order = "xyzw")

        quat_jac = torch.FloatTensor(np.dot(jacobis_root_ori,np.dot(jac_theta_angVelQuatForm,jac_angVel_quat)))

        final_jac=torch.cat((jacobis[:,:,:3],quat_jac[:,:,:3],jacobis[:,:,6:],quat_jac[:,:,-1].view(quat_jac.shape[0],quat_jac.shape[1],1)),2)#.cuda()

        return None,None,torch.bmm(grad_coords.view(len(ctx.q), 1, -1), final_jac).view(len(ctx.q), -1)

    @staticmethod
    def get_jacobi_anglularVel_quaternion(self,quat):
        x=quat[0]
        y=quat[1]
        z=quat[2]
        w=quat[3]
        jac=(2/self.delta_t)*np.array([[w,z,-y,x],
                                  [-z,w,x,y],
                                  [y,-x,w,z],
                                  [-x,-y,-z,w]])

        return jac

    @staticmethod
    def get_jacobi_theta_angVel(ctx,delta_t,order):

        if order == "xyzw":
            jac = np.array([[delta_t,0,0,0],
                            [0, delta_t, 0, 0],
                            [0, 0, delta_t, 0],
                            ])
        elif order == "wxyz":
            jac = np.array([[0,delta_t,0,0],
                            [0, 0, delta_t, 0],
                            [0, 0, 0, delta_t],
                            ])
        else:
            print('not supported order of the quaternion')
        return jac



class PyForwardKinematicsQuaternionHuman36M_test(torch.autograd.Function):

    @staticmethod
    def forward(ctx, models,target_joint_ids,delta_t,sub_ids, q): 
        ctx.delta_t=delta_t
        ctx.q = q.cpu().data.numpy()
        ctx.qdot_size=models[0].qdot_size
        ctx.target_joint_ids= target_joint_ids
        ctx.models =  models 
        
        coords = torch.tensor([[rbdl.CalcBodyToBaseCoordinates( models[batch],  ctx.q[batch].flatten().astype(float), i, np.zeros(3)) for i in  target_joint_ids] for batch in range(len(q))]).view(len(q), -1).float()
        coords = coords#.cuda()
        return coords

    @staticmethod
    def backward(ctx, grad_coords): 
        jacobis = np.zeros((len(ctx.q), len(ctx.target_joint_ids), 3, ctx.qdot_size)) 
        for batch in range(len(ctx.q)):
            for i, id in enumerate(ctx.target_joint_ids):
                rbdl.CalcPointJacobian(ctx.models[batch], ctx.q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
        jacobis = torch.tensor(jacobis).view(len(ctx.q), -1,ctx.models[0].qdot_size).float() 
        jacobis_root_ori = jacobis[:,:,3:6]


        quat = np.concatenate((ctx.q[:,3:6].flatten(),ctx.q[:,-1]),0)
        quat_inv = np.array([-quat[0],-quat[1],-quat[2],quat[3]])


        jac_angVel_quat =  PyForwardKinematicsQuaternionHuman36M.get_jacobi_anglularVel_quaternion(ctx.delta_t,quat_inv) 
        jac_theta_angVelQuatForm= PyForwardKinematicsQuaternionHuman36M.get_jacobi_theta_angVel(ctx.delta_t,order = "xyzw") 

        quat_jac = torch.FloatTensor(np.dot(jacobis_root_ori,np.dot(jac_theta_angVelQuatForm,jac_angVel_quat)))

        final_jac=torch.cat((jacobis[:,:,:3],quat_jac[:,:,:3],jacobis[:,:,6:],quat_jac[:,:,-1].view(quat_jac.shape[0],quat_jac.shape[1],1)),2).cuda() 
        return None,None,None,None,torch.bmm(grad_coords.view(len(ctx.q), 1, -1).cuda(), final_jac).view(len(ctx.q), -1)

    @staticmethod
    def get_jacobi_anglularVel_quaternion(ctx,delta_t,quat):
        x=quat[0]
        y=quat[1]
        z=quat[2]
        w=quat[3]
        jac=(2/ delta_t)*np.array([[w,z,-y,x],
                                  [-z,w,x,y],
                                  [y,-x,w,z],
                                  [-x,-y,-z,w]])

        return jac

    @staticmethod
    def get_jacobi_theta_angVel(ctx, delta_t,order):

        if order == "xyzw":
            jac = np.array([[delta_t,0,0,0],
                            [0, delta_t, 0, 0],
                            [0, 0, delta_t, 0],
                            ])
        elif order == "wxyz":
            jac = np.array([[0,delta_t,0,0],
                            [0, 0, delta_t, 0],
                            [0, 0, 0, delta_t],
                            ])
        else:
            print('not supported order of the quaternion')
        return jac


class PyForwardKinematicsQuaternion(torch.autograd.Function):
    def __init__(self, model, target_joint_ids,delta_t=0.0001):
        self.model = model
        self.target_joint_ids = target_joint_ids
        self.q = None
        self.jacobi_inits = np.array([np.zeros((3, self.model.qdot_size)) for i in range(len(self.target_joint_ids))])
        self.delta_t=delta_t

    @staticmethod
    def forward(self, q):
 
        self.q = q.cpu().data.numpy()

        coords = torch.tensor([[rbdl.CalcBodyToBaseCoordinates(self.model, self.q[batch].flatten().astype(float), i, np.zeros(3)) for i in self.target_joint_ids] for batch in range(len(self.q))]).view(len(self.q), -1).float()
        coords = coords#.cuda()
        return coords

    @staticmethod
    def backward(self, grad_coords):

        jacobis = np.zeros((len(self.q), len(self.target_joint_ids), 3, self.model.qdot_size)) 
        for batch in range(len(self.q)):
            for i, id in enumerate(self.target_joint_ids):
                rbdl.CalcPointJacobian(self.model, self.q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
        jacobis = torch.tensor(jacobis).view(len(self.q), -1, self.model.qdot_size).float()
        jacobis_root_ori = jacobis[:,:,3:6]


        quat = np.concatenate((self.q[:,3:6].flatten(),self.q[:,-1]),0)
        quat_inv = np.array([-quat[0],-quat[1],-quat[2],quat[3]])


        jac_angVel_quat = self.get_jacobi_anglularVel_quaternion(quat_inv)
        jac_theta_angVelQuatForm=self.get_jacobi_theta_angVel(self.delta_t,order = "xyzw")


        quat_jac = torch.FloatTensor(np.dot(jacobis_root_ori,np.dot(jac_theta_angVelQuatForm,jac_angVel_quat)))


        final_jac=torch.cat((jacobis[:,:,:3],quat_jac[:,:,:3],jacobis[:,:,6:],quat_jac[:,:,-1].view(quat_jac.shape[0],quat_jac.shape[1],1)),2)#.cuda()

        return torch.bmm(grad_coords.view(len(self.q), 1, -1), final_jac).view(len(self.q), -1)

    def get_jacobi_anglularVel_quaternion(self,quat):
        x=quat[0]
        y=quat[1]
        z=quat[2]
        w=quat[3]
        jac=(2/self.delta_t)*np.array([[w,z,-y,x],
                                  [-z,w,x,y],
                                  [y,-x,w,z],
                                  [-x,-y,-z,w]])

        return jac

    def get_jacobi_theta_angVel(self,delta_t,order):

        if order == "xyzw":
            jac = np.array([[delta_t,0,0,0],
                            [0, delta_t, 0, 0],
                            [0, 0, delta_t, 0],
                            ])
        elif order == "wxyz":
            jac = np.array([[0,delta_t,0,0],
                            [0, 0, delta_t, 0],
                            [0, 0, 0, delta_t],
                            ])
        else:
            print('not supported order of the quaternion')
        return jac


class PyPDController(torch.autograd.Function):
    @staticmethod
    def forward(ctx, action, kp,kd,q0,qdot0):

        tau = kp*(action-q0.view(-1))-kd*qdot0.view(-1)
        ctx.save_for_backward(kp,q0)

        return tau
    @staticmethod
    def backward(ctx, grad_output):

        kp,q0, = ctx.saved_tensors

        batch_size,dof,_ = q0.shape

        return  torch.bmm(grad_output.view(N,1,6),kp*torch.eye(dof).repeat(batch_size,1,1)) , None, None, None, None

class PyForwardDynamics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tau, M_inv):
        N, w, h = M_inv.shape
        ctx.save_for_backward(M_inv)

        return torch.bmm(M_inv, tau.view(N, -1, 1)).view(N, w)

    @staticmethod
    def backward(ctx, grad_output):

        M_inv, = ctx.saved_tensors
        N, w, h = M_inv.shape
        grad_input = grad_output.clone() 
        return  torch.bmm(grad_input.view(N, 1, w), M_inv).view(N, w), None


class PyPoseUpdate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, delta_t, q0, qdot0, qddot):
        ctx.save_for_backward(delta_t)

        qdot = qdot0 + delta_t*qddot #$torch.bmm(delta_t, qddot)
        print(q0.shape,delta_t.shape,qdot.shape)
        q = q0 + delta_t*qdot#torch.bmm(delta_t, qdot)
        return qdot, q

    @staticmethod
    def backward(ctx, grad_qdot, grad_q):

        delta_t, = ctx.saved_tensors
        grad_q_input = grad_q.clone() 
        return None, None, None,  torch.bmm(grad_q_input.view(len(grad_q), 1, -1), delta_t).view(len(grad_q), -1, 1)


class PyForwardKinematics(torch.autograd.Function):
    def __init__(self, model, target_joint_ids):
        self.model = model
        self.target_joint_ids = target_joint_ids
        self.q = None
        self.jacobi_inits = np.array([np.zeros((3, self.model.qdot_size)) for i in range(len(self.target_joint_ids))])

    #= @staticmethod
    def forward(self, q):
 
        self.q = q.cpu().data.numpy()

        coords = torch.tensor([[rbdl.CalcBodyToBaseCoordinates(self.model, self.q[batch].flatten().astype(float), i, np.zeros(3)) for i in self.target_joint_ids] for batch in
                               range(len(self.q))]).view(len(self.q), -1).float()#.cuda()
 
        return coords.view(len(self.q),len(self.target_joint_ids),-1)

    # @staticmethod
    def backward(self, grad_coords):

        jacobis = np.zeros((len(self.q), len(self.target_joint_ids), 3, self.model.qdot_size)) 
        for batch in range(len(self.q)):
            for i, id in enumerate(self.target_joint_ids):
                rbdl.CalcPointJacobian(self.model, self.q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
        jacobis = torch.tensor(jacobis).view(len(self.q), -1, self.model.qdot_size).float()#.cuda()
        return torch.bmm(grad_coords.view(len(self.q), 1, -1), jacobis).view(len(self.q), -1)


class PyProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P_tensor, p_3D): 
        N,_,=p_3D.shape
        p_3D = p_3D.view(N, -1, 3) # Nx15x3
        ones = torch.ones((N, p_3D.shape[1], 1))#.cuda() # Nx15x3
        p_3D = torch.cat((p_3D, ones), 2).view(N, -1)

        p_proj = torch.bmm(P_tensor, p_3D.view(p_3D.shape[0], p_3D.shape[1], 1))
        ctx.save_for_backward(P_tensor)

        return p_proj

    @staticmethod
    def backward(ctx, grad_output):
        P_tensor, = ctx.saved_tensors
        get_ids = [x for x in range(P_tensor.shape[2]) if (x + 1) % 4 != 0]
        P_tensor_deriv= P_tensor[:,:,get_ids]
        return None, torch.bmm(grad_output.view(len(grad_output), 1, -1), P_tensor_deriv).view(len(grad_output), -1).float()

def get_P_tensor(N, target_joint_ids, P):
    P_tensor = torch.zeros(N, 3 * len(target_joint_ids), 3 * len(target_joint_ids), device=device, dtype=dtype)
    for i in range(int(P_tensor.shape[1] / 3)):
        P_tensor[:, i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = P
    return P_tensor.type(torch.FloatTensor)

class PyPerspectivieDivision(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p_proj):

        ctx.save_for_backward(p_proj)
        batch_size, _, _ = p_proj.shape
        p_proj = p_proj.view(batch_size, -1, 3)

        p_proj /= p_proj[:, :, 2].view(batch_size, -1, 1).clone()
        p_proj = p_proj[:, :, :-1]
        p_proj = p_proj.reshape(batch_size, -1, 1)

        return p_proj

    @classmethod
    def backward(self, ctx, grad_output):
        p_proj, = ctx.saved_tensors

        p_proj = p_proj.view(p_proj.shape[0], -1, 3)
        batch_size, n_points, _ = p_proj.shape
        p_proj = p_proj.cpu().data.numpy()

        grads = np.array([[self.deriv_pd_comp(p_proj[i][j]) for j in range(n_points)] for i in range(batch_size)])
        grads = torch.tensor(grads).float()#.cuda()

        final_grads = torch.zeros(batch_size, n_points * 2, n_points * 3)#.cuda()
        for i in range(n_points):
            final_grads[:, i * 2:(i + 1) * 2, i * 3:(i + 1) * 3] = grads[:, i]
        final = torch.bmm(grad_output.view(batch_size, 1, -1), final_grads).view(batch_size, -1, 1)

        return final

    @staticmethod
    def deriv_pd_comp(coord):
        jacobi = [[1 / coord[2], 0, -coord[0] / (coord[2] ** 2)],
                  [0, 1 / coord[2], -coord[1] / (coord[2] ** 2)]]
        return np.array(jacobi)

