import torch
import math

class angle_util():
    def pysinc(self,x):
        return torch.sin(math.pi * x) / (math.pi * x)
    def expmap_to_quaternion(self,e):
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
        theta = torch.norm(e, dim=1).view(-1, 1) 
        w = torch.cos(0.5 * theta).view(-1, 1) 
        xyz = 0.5 * self.pysinc(0.5 * theta / math.pi) * e
        return torch.cat((w, xyz), 1).view(original_shape)


    def quat_shortest_path(self,quat):
        n_b = quat.size()[0]
        masks = torch.ones(n_b)#.cuda()
        larger_id = torch.flatten(torch.nonzero((quat[:, 0] < 0) * masks).detach())
        quat[larger_id] = self.q_conj(quat[larger_id].view(larger_id.shape[0], 4))
        return quat

    def quat_shortest_path_cpu(self,quat):
        n_b = quat.size()[0]
        masks = torch.ones(n_b).float()##.cuda() 
        larger_id = torch.flatten(torch.nonzero((quat[:, 0] < 0).float() * masks).detach())
        quat[larger_id] = self.q_conj(quat[larger_id].view(larger_id.shape[0], 4))
        return quat
    def qmul(self,q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4

        original_shape = q.shape

        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape)

    def target_correction_batch(self,error):
        masks = torch.ones(error.size()[0], error.size()[1], )#.cuda()

        larger_id = torch.nonzero((error > math.pi) * masks).detach()
        smaller_id = torch.nonzero((error < - math.pi) * masks).detach()

        error[larger_id[:, 0], larger_id[:, 1]] = error[larger_id[:, 0], larger_id[:, 1]] - 2 * math.pi
        error[smaller_id[:, 0], smaller_id[:, 1]] = error[smaller_id[:, 0], smaller_id[:, 1]] + 2 * math.pi
        return error

    def target_correction_batch_cpu(self,error):
        masks = torch.ones(error.size()[0], error.size()[1], ).float()##.cuda()

        larger_id = torch.nonzero((error > math.pi).float() * masks).detach()
        smaller_id = torch.nonzero((error < - math.pi).float() * masks).detach()

        error[larger_id[:, 0], larger_id[:, 1]] = error[larger_id[:, 0], larger_id[:, 1]] - 2 * math.pi
        error[smaller_id[:, 0], smaller_id[:, 1]] = error[smaller_id[:, 0], smaller_id[:, 1]] + 2 * math.pi
        return error
    def angle_normalize_batch(self,q_all):

        q=q_all[:,3:-1].clone()
        masks = torch.ones(q.size()[0], q.size()[1])#.cuda()
        mod = torch.remainder(q, 2 * math.pi)
        larger_id = torch.nonzero((mod > math.pi) * masks).detach()#.cuda()
        smaller_id = torch.nonzero((mod <= math.pi) * masks).detach()#.cuda()
        q[larger_id[:, 0], larger_id[:, 1]] = mod[larger_id[:, 0], larger_id[:, 1]] - 2 * math.pi
        q[smaller_id[:, 0], smaller_id[:, 1]] = mod[smaller_id[:, 0], smaller_id[:, 1]]
        q_all[:, 3:-1]=q.clone()
        return q_all
    def angle_normalize_batch_cpu(self,q_all):

        q=q_all[:,3:-1].clone()
        masks = torch.ones(q.size()[0], q.size()[1]).float()##.cuda()
        mod = torch.remainder(q, 2 * math.pi)
        larger_id = torch.nonzero((mod > math.pi).float() * masks).detach()#.cuda()
        smaller_id = torch.nonzero((mod <= math.pi).float() * masks).detach()#.cuda()
        q[larger_id[:, 0], larger_id[:, 1]] = mod[larger_id[:, 0], larger_id[:, 1]] - 2 * math.pi
        q[smaller_id[:, 0], smaller_id[:, 1]] = mod[smaller_id[:, 0], smaller_id[:, 1]]
        q_all[:, 3:-1]=q.clone()
        return q_all
    def angle_normalize_art_batch(self,q_art):

        #q=q_all[:,3:-1].clone()
        masks = torch.ones(q_art.size()[0], q_art.size()[1])#.cuda()
        mod = torch.remainder(q_art, 2 * math.pi)
        larger_id = torch.nonzero((mod > math.pi) * masks).detach()#.cuda()
        smaller_id = torch.nonzero((mod <= math.pi) * masks).detach()#.cuda()
        q_art[larger_id[:, 0], larger_id[:, 1]] = mod[larger_id[:, 0], larger_id[:, 1]] - 2 * math.pi
        q_art[smaller_id[:, 0], smaller_id[:, 1]] = mod[smaller_id[:, 0], smaller_id[:, 1]]
        q_art[:, 3:-1]=q_art.clone()
        return q_art


    def angle_normalize_euler_batch(self,q_all):

        q=q_all[:,3: ].clone()
        masks = torch.ones(q.size()[0], q.size()[1])#.cuda()
        mod = torch.remainder(q, 2 * math.pi)
        larger_id = torch.nonzero((mod > math.pi) * masks).detach()#.cuda()
        smaller_id = torch.nonzero((mod <= math.pi) * masks).detach()#.cuda()
        q[larger_id[:, 0], larger_id[:, 1]] = mod[larger_id[:, 0], larger_id[:, 1]] - 2 * math.pi
        q[smaller_id[:, 0], smaller_id[:, 1]] = mod[smaller_id[:, 0], smaller_id[:, 1]]
        q_all[:, 3:]=q.clone()
        return q_all
    def angle_normalize_batch_exp(self,q_all):

        q=q_all[:,3:].clone()
        masks = torch.ones(q.size()[0], q.size()[1])#.cuda()
        mod = torch.remainder(q, 2 * math.pi)
        larger_id = torch.nonzero((mod > math.pi) * masks).detach()#.cuda()
        smaller_id = torch.nonzero((mod <= math.pi) * masks).detach()#.cuda()
        q[larger_id[:, 0], larger_id[:, 1]] = mod[larger_id[:, 0], larger_id[:, 1]] - 2 * math.pi
        q[smaller_id[:, 0], smaller_id[:, 1]] = mod[smaller_id[:, 0], smaller_id[:, 1]]
        q_all[:, 3:]=q.clone()
        return q_all

    def angle_clean(self,q):
        mod = q % (2 * math.pi)
        if mod >= math.pi:
            return mod - 2 * math.pi
        else:
            return mod

    def normalize_vector_prep(self,v):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        return v


    def normalize_vector(self,v):
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))#.cuda())
        v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
        v = v / v_mag
        return v

    def quat_bullet2general(self,q):
        return torch.FloatTensor([q[3],q[0],q[1],q[2]])

    def quat_bullet2general_b(self,q):
        return torch.cat((q[:,3].view(len(q),1),q[:,0].view(len(q),1),q[:,1].view(len(q),1),q[:,2].view(len(q),1)),1)

    def quat_bullet2general_b_s(self,q):
        B,T,C=q.shape
        return torch.cat((q[:,:,3].view(B,T,1),q[:,:,0].view(B,T,1),q[:,:,1].view(B,T,1), q[:,:,2].view(B,T,1)),2)

    def q_conj(self,q):
        w = q[:, 0]
        x = -q[:, 1]
        y = -q[:, 2]
        z = -q[:, 3]
        return torch.stack((w, x, y, z), dim=1)
#            angvel_quat = np.array([0,qdot[3],qdot[4],qdot[5]])

    def get_angvel_quat(self,qdot):
        return torch.stack((torch.zeros(len(qdot))  ,qdot[:,0],qdot[:,1],qdot[:,2]), dim=1)#.cuda()
    def get_angvel_quat_cpu(self,qdot):
        return torch.stack((torch.zeros(len(qdot)) ,qdot[:,0],qdot[:,1],qdot[:,2]), dim=1)
    def quat_conj(self,q):
        return torch.FloatTensor([q[0],-q[1],-q[2],-q[3]])

    def compute_rotation_matrix_from_quaternion(self,quaternion):

        batch = quaternion.shape[0]

        quat = self.normalize_vector(quaternion)

        qw = quat[..., 0].view(batch, 1)
        qx = quat[..., 1].view(batch, 1)
        qy = quat[..., 2].view(batch, 1)
        qz = quat[..., 3].view(batch, 1)

        # Unit quaternion rotation matrices computatation
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        xw = qx * qw
        yw = qy * qw
        zw = qz * qw

        row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
        row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
        row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

        matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

        return matrix#.detach()  # .numpy()





    def compute_rotation_matrix_from_quaternion_prep(self, quaternion):

        batch = quaternion.shape[0]

        quat = self.normalize_vector_prep(quaternion)

        qw = quat[..., 0].view(batch, 1)
        qx = quat[..., 1].view(batch, 1)
        qy = quat[..., 2].view(batch, 1)
        qz = quat[..., 3].view(batch, 1)

        # Unit quaternion rotation matrices computatation
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        xw = qx * qw
        yw = qy * qw
        zw = qz * qw

        row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
        row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
        row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

        matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

        return matrix  # .detach()  # .numpy()

    def cross_product(self,u, v):
        batch = u.shape[0]
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

        return out


    def compute_rotation_matrix_from_ortho6d(self,poses):
        x_raw = poses[:, 0:3]  # batch*3
        y_raw = poses[:, 3:6]  # batch*3

        x = self.normalize_vector(x_raw)  # batch*3
        z = self.cross_product(x, y_raw)  # batch*3
        z = self.normalize_vector(z)  # batch*3
        y = self.cross_product(z, x)  # batch*3

        x = x.view(-1, 3, 1)
        y = y.view(-1, 3, 1)
        z = z.view(-1, 3, 1)
        matrix = torch.cat((x, y, z), 2)  # batch*3*3
        return matrix


    def compute_rotation_matrix_loss(self,gt_rotation_matrix, predict_rotation_matrix):
        loss_function = torch.nn.MSELoss()
        loss = loss_function(predict_rotation_matrix, gt_rotation_matrix)
        return loss


    def get_44_rotation_matrix_from_33_rotation_matrix(self,m):
        batch = m.shape[0]

        row4 = torch.autograd.Variable(torch.zeros(batch, 1, 3))#.cuda())

        m43 = torch.cat((m, row4), 1)  # batch*4,3

        col4 = torch.autograd.Variable(torch.zeros(batch, 4, 1))#.cuda())
        col4[:, 3, 0] = col4[:, 3, 0] + 1

        out = torch.cat((m43, col4), 2)  # batch*4*4

        return out


    def get_44_rotation_matrix_from_33_rotation_matrix_prep(self,m):
        batch = m.shape[0]

        row4 = torch.autograd.Variable(torch.zeros(batch, 1, 3))

        m43 = torch.cat((m, row4), 1)  # batch*4,3

        col4 = torch.autograd.Variable(torch.zeros(batch, 4, 1))
        col4[:, 3, 0] = col4[:, 3, 0] + 1

        out = torch.cat((m43, col4), 2)  # batch*4*4
        out = out.view(4, 4).detach().cpu().numpy()

        return out


    def compute_euler_angles_from_rotation_matrices(self,rotation_matrices):
        batch = rotation_matrices.shape[0]
        R = rotation_matrices
        sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
        singular = sy < 1e-6
        singular = singular.float()

        x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        y = torch.atan2(-R[:, 2, 0], sy)
        z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

        xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
        ys = torch.atan2(-R[:, 2, 0], sy)
        zs = R[:, 1, 0] * 0

        out_euler = torch.autograd.Variable(torch.zeros(batch, 3))
        out_euler[:, 0] = x * (1 - singular) + xs * singular
        out_euler[:, 1] = y * (1 - singular) + ys * singular
        out_euler[:, 2] = z * (1 - singular) + zs * singular

        return out_euler


    def compute_quaternions_from_rotation_matrices(self,matrices):
        batch = matrices.shape[0]

        w = torch.sqrt(1.0 + matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]) / 2.0
        w = torch.max(w, torch.autograd.Variable(torch.zeros(batch)) + 1e-8)  # batch
        w4 = 4.0 * w
        x = (matrices[:, 2, 1] - matrices[:, 1, 2]) / w4
        y = (matrices[:, 0, 2] - matrices[:, 2, 0]) / w4
        z = (matrices[:, 1, 0] - matrices[:, 0, 1]) / w4

        quats = torch.cat((w.view(batch, 1), x.view(batch, 1), y.view(batch, 1), z.view(batch, 1)), 1)

        return quats

