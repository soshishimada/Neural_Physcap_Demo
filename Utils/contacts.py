import numpy as np
import torch 
import rbdl 

def get_mass_mat(model, q):
    n_b, _ = q.shape
    M_np = np.zeros((n_b, model.qdot_size, model.qdot_size))
    [rbdl.CompositeRigidBodyAlgorithm(model, q[i].astype(float), M_np[i]) for i in range(n_b)]
    return M_np

def get_contact_point_in_ankle_frame(model, q, rbdl_dic, contact_link_name, ankle_name):
    point_local = np.zeros(3)
    point_ankle_world = rbdl.CalcBodyToBaseCoordinates(model, q, rbdl_dic[ankle_name], point_local)
    point_contact_world = rbdl.CalcBodyToBaseCoordinates(model, q, rbdl_dic[contact_link_name], point_local)
    anckle_rot = rbdl.CalcBodyWorldOrientation(model, q, rbdl_dic[ankle_name])
    contact_in_ankle = np.dot(anckle_rot, point_contact_world - point_ankle_world)
    return contact_in_ankle

def mat_concatenate( mat):
    out = None
    for i in range(len(mat)):
        if i == 0:
            out = mat[i]
        else:
            out = np.concatenate((out, mat[i]), 1)
    return out

def cross2dot_b(v):
    n_b,_=v.shape
    vx=v[:,0].view(-1,1)
    vy=v[:,1].view(-1,1)
    vz=v[:,2].view(-1,1)
    zeros = torch.zeros(n_b).view(-1,1)#.cuda()#.view(-1,1)
    mat_top = torch.cat((zeros,-vz,vy),1).view(n_b,1,3)
    mat_mid = torch.cat((vz,zeros,-vx),1).view(n_b,1,3)
    mat_bot = torch.cat((-vy,vx,zeros),1).view(n_b,1,3)
    return torch.cat((mat_top,mat_mid,mat_bot),1)
def cross2dot_b_cpu(v):
    n_b,_=v.shape
    vx=v[:,0].view(-1,1)
    vy=v[:,1].view(-1,1)
    vz=v[:,2].view(-1,1)
    zeros = torch.zeros(n_b).view(-1,1)
    mat_top = torch.cat((zeros,-vz,vy),1).view(n_b,1,3)
    mat_mid = torch.cat((vz,zeros,-vx),1).view(n_b,1,3)
    mat_bot = torch.cat((-vy,vx,zeros),1).view(n_b,1,3)
    return torch.cat((mat_top,mat_mid,mat_bot),1)
def cross2dot_convert( vectors):
    out = np.array(list(map( c2d_func,vectors)))
    out =  mat_concatenate(out)
    return out

def get_wrench( contact_in_ankle_frame):
    n_b,_=contact_in_ankle_frame.shape

    G_tau  = cross2dot_b(contact_in_ankle_frame) 
    G_lin = torch.eye(3).view(1,3,3)#.cuda()
    G_lin=G_lin.expand(n_b,-1,-1)

    return  torch.cat((G_tau,G_lin),1)
def get_wrench_cpu( contact_in_ankle_frame):
    n_b,_=contact_in_ankle_frame.shape

    G_tau  = cross2dot_b_cpu(contact_in_ankle_frame) 
    G_lin = torch.eye(3).view(1,3,3)
    G_lin=G_lin.expand(n_b,-1,-1)

    return  torch.cat((G_tau,G_lin),1)

def get_contact_jacobis6D(model, q, ids):
    jacobis = np.zeros((len(q), len(ids), 6, model.qdot_size))  
    for batch in range(len(q)):
        for i, id in enumerate(ids):
            rbdl.CalcPointJacobian6D(model, q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
    jacobis = torch.FloatTensor(jacobis).view(len(q), -1, model.qdot_size)#.cuda()
    return jacobis

def get_contact_jacobis6D_cpu(model, q, ids):
    jacobis = np.zeros((len(q), len(ids), 6, model.qdot_size))  
    for batch in range(len(q)):
        for i, id in enumerate(ids):
            rbdl.CalcPointJacobian6D(model, q[batch].flatten().astype(float), id, np.array([0., 0., 0.]), jacobis[batch][i])
    jacobis = torch.FloatTensor(jacobis).view(len(q), -1, model.qdot_size)
    return jacobis

def get_contact_wrench( model,qs,rbdl_dic,conF,con_labels):
    n_b,_=conF.shape
    qs=qs.detach().cpu().numpy().astype(float)

    l_toe_in_ankle = torch.FloatTensor([get_contact_point_in_ankle_frame(model,q,rbdl_dic,"left_toe","left_ankle") for q in qs])#.cuda()
    l_heel_in_ankle= torch.FloatTensor([get_contact_point_in_ankle_frame(model,q,rbdl_dic,"left_heel","left_ankle") for q in qs])#.cuda()
    r_toe_in_ankle = torch.FloatTensor([get_contact_point_in_ankle_frame(model,q,rbdl_dic,"right_toe","right_ankle") for q in qs])#.cuda()
    r_heel_in_ankle= torch.FloatTensor([get_contact_point_in_ankle_frame(model,q,rbdl_dic,"right_heel","right_ankle") for q in qs])#.cuda()

    l_toe_tau_lin_mat =  get_wrench(l_toe_in_ankle)
    l_heel_tau_lin_mat =  get_wrench(l_heel_in_ankle)
    r_toe_tau_lin_mat=  get_wrench(r_toe_in_ankle)
    r_heel_tau_lin_mat =  get_wrench(r_heel_in_ankle)

    l_toe_conF = conF[:,:3].view(n_b,3,1)
    l_heel_conF = conF[:,3:6].view(n_b,3,1)
    r_toe_conF = conF[:,6:9].view(n_b,3,1)
    r_heel_conF = conF[:,9:].view(n_b,3,1)

    l_toe_tau_lin_F = torch.bmm(l_toe_tau_lin_mat,l_toe_conF)
    l_heel_tau_lin_F = torch.bmm(l_heel_tau_lin_mat,l_heel_conF)
    r_toe_tau_lin_F = torch.bmm(r_toe_tau_lin_mat,r_toe_conF)
    r_heel_tau_lin_F = torch.bmm(r_heel_tau_lin_mat,r_heel_conF)

    #print(l_toe_tau_lin_F.shape,con_labels[:,0].shape)
    l_toe_tau_lin_F  = con_labels[:,0].view(n_b,1)*l_toe_tau_lin_F.view(n_b,6)
    l_heel_tau_lin_F = con_labels[:,1].view(n_b,1)*l_heel_tau_lin_F.view(n_b,6)
    r_toe_tau_lin_F  = con_labels[:,2].view(n_b,1)*r_toe_tau_lin_F.view(n_b,6)
    r_heel_tau_lin_F = con_labels[:,3].view(n_b,1)*r_heel_tau_lin_F.view(n_b,6)

    l_ankle_F = l_toe_tau_lin_F+l_heel_tau_lin_F
    r_ankle_F = r_toe_tau_lin_F+r_heel_tau_lin_F


    J = get_contact_jacobis6D(model, qs, [rbdl_dic["left_ankle"], rbdl_dic["right_ankle"]])  # ankles
    JT = torch.transpose(J, 1, 2)
    JT_l = JT[:,:,:6]
    JT_r = JT[:,:,6:]


    gen_l_F = torch.bmm(JT_l,l_ankle_F.view(n_b,6,1))
    gen_r_F = torch.bmm(JT_r,r_ankle_F.view(n_b,6,1)) 
    return (gen_l_F+gen_r_F).view(n_b,model.qdot_size) 


def get_contact_wrench_cpu(model, qs, rbdl_dic, conF, con_labels):
    n_b, _ = conF.shape
    qs = qs.numpy().astype(float)

    l_toe_in_ankle = torch.FloatTensor( [get_contact_point_in_ankle_frame(model, q, rbdl_dic, "left_toe", "left_ankle") for q in qs])
    l_heel_in_ankle = torch.FloatTensor( [get_contact_point_in_ankle_frame(model, q, rbdl_dic, "left_heel", "left_ankle") for q in qs])
    r_toe_in_ankle = torch.FloatTensor( [get_contact_point_in_ankle_frame(model, q, rbdl_dic, "right_toe", "right_ankle") for q in qs])
    r_heel_in_ankle = torch.FloatTensor( [get_contact_point_in_ankle_frame(model, q, rbdl_dic, "right_heel", "right_ankle") for q in qs])

    l_toe_tau_lin_mat = get_wrench_cpu(l_toe_in_ankle)
    l_heel_tau_lin_mat = get_wrench_cpu(l_heel_in_ankle)
    r_toe_tau_lin_mat = get_wrench_cpu(r_toe_in_ankle)
    r_heel_tau_lin_mat = get_wrench_cpu(r_heel_in_ankle)

    l_toe_conF = conF[:, :3].view(n_b, 3, 1)
    l_heel_conF = conF[:, 3:6].view(n_b, 3, 1)
    r_toe_conF = conF[:, 6:9].view(n_b, 3, 1)
    r_heel_conF = conF[:, 9:].view(n_b, 3, 1)

    l_toe_tau_lin_F = torch.bmm(l_toe_tau_lin_mat, l_toe_conF)
    l_heel_tau_lin_F = torch.bmm(l_heel_tau_lin_mat, l_heel_conF)
    r_toe_tau_lin_F = torch.bmm(r_toe_tau_lin_mat, r_toe_conF)
    r_heel_tau_lin_F = torch.bmm(r_heel_tau_lin_mat, r_heel_conF)
 
    l_toe_tau_lin_F = con_labels[:, 0].view(n_b, 1) * l_toe_tau_lin_F.view(n_b, 6)
    l_heel_tau_lin_F = con_labels[:, 1].view(n_b, 1) * l_heel_tau_lin_F.view(n_b, 6)
    r_toe_tau_lin_F = con_labels[:, 2].view(n_b, 1) * r_toe_tau_lin_F.view(n_b, 6)
    r_heel_tau_lin_F = con_labels[:, 3].view(n_b, 1) * r_heel_tau_lin_F.view(n_b, 6)

    l_ankle_F = l_toe_tau_lin_F + l_heel_tau_lin_F
    r_ankle_F = r_toe_tau_lin_F + r_heel_tau_lin_F

    J = get_contact_jacobis6D_cpu(model, qs, [rbdl_dic["left_ankle"], rbdl_dic["right_ankle"]])   
    JT = torch.transpose(J, 1, 2)
    JT_l = JT[:, :, :6]
    JT_r = JT[:, :, 6:]
    gen_l_F = torch.bmm(JT_l, l_ankle_F.view(n_b, 6, 1))
    gen_r_F = torch.bmm(JT_r, r_ankle_F.view(n_b, 6, 1))
     
    return (gen_l_F + gen_r_F).view(n_b, model.qdot_size) 