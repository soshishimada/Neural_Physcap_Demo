import torch 
from Utils.angles import angle_util
class LossFunctions:

    def __init__(self):
        self.AU=angle_util()
    def compute_contact_loss(self,feet,pre_feet,labels):
        n_b,_=labels.shape
        #labels = torch.FloatTensor([[0, 1], [0, 0], [1, 1]])
        #feet = torch.arange(0, 36).view(n_b, 4, 3) - 18
       # pre_feet = 0.2 * (torch.arange(0, 36).view(n_b, 4, 3) - 18)
        residual_feet = (feet - pre_feet).pow(2).sum(2)

        residual_feet_l = residual_feet[:, :2]
        residual_feet_r = residual_feet[:, 2:]
        residual_final = torch.cat((residual_feet_l.sum(1).view(n_b, 1), residual_feet_r.sum(1).view(n_b, 1)), 1)
        loss = (labels * residual_final).mean()
        return loss

    def compute_trans_loss(self,q0,q_trans,frame_gt_qs):
        loss_trans = (q_trans - frame_gt_qs[:, :3]).pow(2).mean()
        loss_trans_smooth =0.3* (q_trans - q0[:,:3]).pow(2).mean()
        return loss_trans,loss_trans_smooth
    def compute_ori_loss(self,quat,frame_gt_rotMats):
        n_b,_=quat.shape
        out_mat = self.AU.compute_rotation_matrix_from_quaternion(quat)
        out_mat = self.AU.get_44_rotation_matrix_from_33_rotation_matrix(out_mat)  # (batch*joint_num)*4*4
        loss_rootOri = self.AU.compute_rotation_matrix_loss(out_mat, frame_gt_rotMats.view(n_b, 4, 4))  # .mean()

        return loss_rootOri

    def compute_ori_loss_quat_quat(self,quat,quat2):
        n_b,_=quat.shape
        out_mat = self.AU.compute_rotation_matrix_from_quaternion(quat)
        out_mat = self.AU.get_44_rotation_matrix_from_33_rotation_matrix(out_mat)  # (batch*joint_num)*4*4

        out_mat2 = self.AU.compute_rotation_matrix_from_quaternion(quat2)
        out_mat2= self.AU.get_44_rotation_matrix_from_33_rotation_matrix(out_mat2)  # (batch*joint_num)*4*4

        loss_rootOri = self.AU.compute_rotation_matrix_loss(out_mat, out_mat2.view(n_b, 4, 4))  # .mean()

        return loss_rootOri



    def compute_smooth_ori_loss(self,quat,quat_pre):
        n_b,_=quat.shape
        out_mat = self.AU.compute_rotation_matrix_from_quaternion(quat)
        out_mat = self.AU.get_44_rotation_matrix_from_33_rotation_matrix(out_mat)  # (batch*joint_num)*4*4

        out_mat_pre = self.AU.compute_rotation_matrix_from_quaternion(quat_pre)
        out_mat_pre = self.AU.get_44_rotation_matrix_from_33_rotation_matrix(out_mat_pre)  # (batch*joint_num)*4*4

        loss_rootOri = self.AU.compute_rotation_matrix_loss(out_mat,out_mat_pre)  # .mean()

        return loss_rootOri
    def compute_five_loss(self,out_quat,out_trans,q_art,p_3D_p,p_2D,gt_qs,gt_rotMats,gt_3Ds,gt_2Ds):
        n_b,_=out_quat.shape
        out_mat = self.AU.compute_rotation_matrix_from_quaternion(out_quat)
        out_mat = self.AU.get_44_rotation_matrix_from_33_rotation_matrix(out_mat)  # (batch*joint_num)*4*4
        loss_rootOri = self.AU.compute_rotation_matrix_loss(out_mat, gt_rotMats.view(n_b, 4, 4))/n_b

        #print(out_trans.shape,gt_qs.shape,out_quat.shape,gt_3Ds.shape,gt_2Ds.shape)
        loss_trans = (out_trans - gt_qs[:,:3]).pow(2).sum()/n_b
        loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[:,7:])).pow(2).sum() + (torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[:,7:])).pow(2).sum()) / 2
        loss_q /=n_b
        #print(p_3D_p.shape,gt_3Ds.shape)
        loss3D_p = ( p_3D_p  - gt_3Ds.view(n_b,-1) ).pow(2).sum()/n_b
        loss2D = (torch.squeeze(p_2D).view(n_b, -1, 2) - gt_2Ds).pow(2).sum()/n_b

        return loss_trans,loss_rootOri,loss_q,loss3D_p,loss2D

    def compute_three_loss(self,out_quat,out_trans,q_art,gt_qs,gt_rotMats):
        batch_size,_= out_quat.shape
        out_mat = self.AU.compute_rotation_matrix_from_quaternion(out_quat)
        out_mat = self.AU.get_44_rotation_matrix_from_33_rotation_matrix(out_mat)  # (batch*joint_num)*4*4
        loss_rootOri = self.AU.compute_rotation_matrix_loss(out_mat, gt_rotMats.view(batch_size, 4, 4))
        loss_trans = (out_trans - gt_qs[:3]).pow(2).sum()
        loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[6:])).pow(2).sum() + (torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[6:])).pow(2).sum()) / 2


        return loss_trans,loss_rootOri,loss_q

    def compute_q_p3d_p2d_loss(self,q_art,p_3D_p,p_2D,gt_qs,gt_3Ds,gt_2Ds):
        n_b,_= q_art.shape

        #loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[:,7:])).pow(2).sum() + (torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[:,7:])).pow(2).sum()) / 2
        loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[:, 7:])).pow(2)  + ( torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[:, 7:])).pow(2) ).mean()
        #loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[:, 7:])).pow(2)  + ( torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[:, 7:])).pow(2) ).mean()

        loss3D_p = ( p_3D_p  - gt_3Ds.view(n_b,-1) ).pow(2).mean()#/n_b

        loss2D = (torch.squeeze(p_2D) - gt_2Ds.view(n_b,-1)).pow(2).mean()#/n_b
        return loss_q,loss3D_p,loss2D

    def compute_part_q_p3d_p2d_loss(self, q_art, p_3D_p, p_2D, gt_qs, gt_3Ds, gt_2Ds):
        n_b, _ = q_art.shape
        gt_qs_art = gt_qs[:,7:]
        # loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[:,7:])).pow(2).sum() + (torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[:,7:])).pow(2).sum()) / 2
        loss_q_right_leg = ((torch.sin(q_art[:,[12,14]]) - torch.sin(gt_qs_art[:, [12,14]])).pow(2) + (torch.cos(q_art[:, [12,14]]) - torch.cos(gt_qs_art[:, [12,14]])).pow(2)).mean()
        # loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[:, 7:])).pow(2)  + ( torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[:, 7:])).pow(2) ).mean()

        loss3D_p = (p_3D_p - gt_3Ds.view(n_b, -1)).pow(2).mean()  # /n_b

        loss2D = (torch.squeeze(p_2D) - gt_2Ds.view(n_b, -1)).pow(2).mean()  # /n_b
        return loss_q_right_leg, loss3D_p, loss2D

    def compute_q_p3d_p2d_more_hand_loss(self,q_art,p_3D_p,p_2D,gt_qs,gt_3Ds,gt_2Ds,weights):
        n_b,_= q_art.shape

        #loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[:,7:])).pow(2).sum() + (torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[:,7:])).pow(2).sum()) / 2
        loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[:, 7:])).pow(2)  + ( torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[:, 7:])).pow(2) ).mean()


        residual_3D = ( p_3D_p  - gt_3Ds.view(n_b,-1) ).view(n_b,-1,3)#.pow(2).mean()#/n_b
        hand_loss = residual_3D[:,11].detach().clone().pow(2).mean()+residual_3D[:,14].detach().clone().pow(2).mean()
        residual_3D = weights*residual_3D
        loss3D_p = residual_3D.pow(2).mean()#/n_b
        residual_2D =(torch.squeeze(p_2D) - gt_2Ds.view(n_b,-1)).view(n_b,-1,2)
        residual_2D = weights * residual_2D
        loss2D = residual_2D.pow(2).mean()  # /n_b
        return loss_q,loss3D_p,loss2D,hand_loss

    def irregular_angle_loss(self,q_art, max_angles, min_angles):

        mask = torch.ones(q_art.shape).cuda()
        larger_lables = ((q_art > max_angles) * mask).detach()
        larger_loss = ((q_art - max_angles) * larger_lables).pow(2).mean()
        smaller_labels = ((q_art < min_angles) * mask).detach()
        smaller_loss = ((min_angles - q_art) * smaller_labels).pow(2).mean()
        return larger_loss + smaller_loss

    def compute_q_p3d_p2d_loss_self(self,q_art,p_3D_p,p_2D,q_ref,gt_3Ds,gt_2Ds):
        n_b,_= q_art.shape

        #loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(gt_qs[:,7:])).pow(2).sum() + (torch.cos(torch.squeeze(q_art)) - torch.cos(gt_qs[:,7:])).pow(2).sum()) / 2
        loss_q = ((torch.sin(torch.squeeze(q_art)) - torch.sin(q_ref[:, 6:-1])).pow(2)  + ( torch.cos(torch.squeeze(q_art)) - torch.cos(q_ref[:, 6:-1])).pow(2) ).mean()


        loss3D_p = ( p_3D_p  - gt_3Ds.view(n_b,-1) ).pow(2).mean()#/n_b

        loss2D = (torch.squeeze(p_2D) - gt_2Ds.view(n_b,-1)).pow(2).mean()#/n_b
        return loss_q,loss3D_p,loss2D