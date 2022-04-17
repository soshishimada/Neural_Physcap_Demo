import torch
import numpy as np 
import rbdl
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
        p_proj = p_proj.data.numpy()

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

class PyForwardKinematicsQuaternionHuman36M(torch.autograd.Function):
    def __init__(self, model_addresses, target_joint_ids,delta_t=0.0001):
        #self.model = model
        self.model_addresses=model_addresses

        self.qdot_size=model_addresses["1"].qdot_size
        self.target_joint_ids = target_joint_ids
        self.q = None
        self.jacobi_inits = np.array([np.zeros((3,  self.qdot_size)) for i in range(len( target_joint_ids))])
        self.delta_t=delta_t

    # @staticmethod
    def forward(self, sub_ids, q):
        self.q = q.data.numpy()
        self.models = [self.model_addresses[str(int(x))] for x in sub_ids.numpy()]
        coords = torch.tensor([[rbdl.CalcBodyToBaseCoordinates(self.models[batch], self.q[batch].flatten().astype(float), i, np.zeros(3)) for i in self.target_joint_ids] for batch in range(len(self.q))]).view(len(self.q), -1).float()

        return coords

    # @staticmethod
    def backward(self, grad_coords):

        jacobis = np.zeros((len(self.q), len(self.target_joint_ids), 3, self.qdot_size))  # np.array([np.zeros((3, self.model.qdot_size)) for i in range(len(self.target_joint_ids))])
        for batch in range(len(self.q)):
            for i, id in enumerate(self.target_joint_ids):
                rbdl.CalcPointJacobian(self.models[batch], self.q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
        jacobis = torch.tensor(jacobis).view(len(self.q), -1,self.models[0].qdot_size).float()
        jacobis_root_ori = jacobis[:,:,3:6]


        quat = np.concatenate((self.q[:,3:6].flatten(),self.q[:,-1]),0)
        quat_inv = np.array([-quat[0],-quat[1],-quat[2],quat[3]])

        jac_angVel_quat = self.get_jacobi_anglularVel_quaternion(quat_inv)
        jac_theta_angVelQuatForm=self.get_jacobi_theta_angVel(self.delta_t,order = "xyzw")

        quat_jac = torch.FloatTensor(np.dot(jacobis_root_ori,np.dot(jac_theta_angVelQuatForm,jac_angVel_quat)))

        final_jac=torch.cat((jacobis[:,:,:3],quat_jac[:,:,:3],jacobis[:,:,6:],quat_jac[:,:,-1].view(quat_jac.shape[0],quat_jac.shape[1],1)),2)#.cuda()

        return None,torch.bmm(grad_coords.view(len(self.q), 1, -1), final_jac).view(len(self.q), -1)

class PyProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, P_tensor, p_3D):
        #print(P_tensor.shape,p_3D.shape)
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
