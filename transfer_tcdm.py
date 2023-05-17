# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, time
import argparse
from tcdm import suite
from tcdm.motion_util import to_quat, to_transform_mat, axis_angle_to_rot
from tcdm.envs.mujoco import physics_from_mjcf
from tcdm.envs import mj_models
import matplotlib.pyplot as plt
import mano
import imageio
import moviepy.editor as mp



def get_meshes(dorig, coarse_net, refine_net, rh_model, save=True, save_dir='out'):
    with torch.no_grad():

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        drec_cnet = coarse_net.sample_poses(dorig['bps_object'])
        rh_gen_cnet = rh_model(**drec_cnet, return_full_pose=True)
        verts_rh_gen_cnet = rh_gen_cnet.vertices
        joints_rh_gen_cnet = rh_gen_cnet.joints
        pose_rh_gen_cnet = rh_gen_cnet.full_pose.reshape(-1, 16, 3)

        _, h2o, _ = point2point_signed(verts_rh_gen_cnet, dorig['verts_object'].to(device))

        drec_cnet['trans_rhand_f'] = drec_cnet['transl']
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
        drec_cnet['verts_object'] = dorig['verts_object'].to(device)
        drec_cnet['h2o_dist'] = h2o.abs()

        drec_rnet = refine_net(**drec_cnet)
        rh_gen_rnet = rh_model(**drec_rnet, return_full_pose=True)
        verts_rh_gen_rnet = rh_gen_rnet.vertices
        joints_rh_gen_rnet = rh_gen_rnet.joints
        pose_rh_gen_rnet = rh_gen_rnet.full_pose.reshape(-1, 16, 3)

        jv_out = dict(verts_rh_gen_rnet=verts_rh_gen_rnet,joints_rh_gen_rnet=joints_rh_gen_rnet,verts_rh_gen_cnet=verts_rh_gen_cnet,joints_rh_gen_cnet=joints_rh_gen_cnet)
        jv_out['pose_rh_gen_cnet'] = pose_rh_gen_cnet; jv_out['verts_object'] = dorig['verts_object'];  jv_out['pose_rh_gen_rnet'] = pose_rh_gen_rnet
        jv_out = {k: v.detach().cpu().numpy() for k, v in jv_out.items()}
        jv_out['sample_rot_mats'] = np.array(dorig['rotmat'])
        gen_meshes = []
        for cId in range(0, len(dorig['bps_object'])):
            try:
                obj_mesh = dorig['mesh_object'][cId]
            except:
                obj_mesh = points2sphere(points=to_cpu(dorig['verts_object'][cId]), radius=0.002, vc=name_to_rgb['yellow'])

            hand_mesh_gen_rnet = Mesh(vertices=to_cpu(verts_rh_gen_rnet[cId]), faces=rh_model.faces, vc=[245, 191, 177])

            if 'rotmat' in dorig:
                rotmat = dorig['rotmat'][cId].T
                obj_mesh = obj_mesh.rotate_vertices(rotmat)
                hand_mesh_gen_rnet.rotate_vertices(rotmat)  
            gen_meshes.append([obj_mesh, hand_mesh_gen_rnet])
            if False:
                save_dir = 'demo_out'
                save_path = os.path.join(save_dir, str(cId))
                makepath(save_path)
                import ipdb; ipdb.set_trace()
                hand_mesh_gen_rnet.export(filename=save_path + '/rh_mesh_gen_%d.obj' % cId, filetype='obj')
                obj_mesh.export(filename=save_path + '/obj_mesh_%d.obj' % cId, filetype='obj')

        return gen_meshes, jv_out


def get_meshes_grasptta():
    grasp_save_dir = '/home/ishaans/grasp_tta_output_v9'
    grasp_fn = [g for g in os.listdir(grasp_save_dir) if not ('object.npy' in g) and not ('mano_output' in g)]
    grasp_paths = [os.path.join(grasp_save_dir,g) for g in grasp_fn]
    object_paths = [os.path.join(grasp_save_dir,g.split('.')[0] + '_object.npy') for g in grasp_fn]


    for i in range(len(grasp_paths)):
        object_vertices = np.load(object_paths[i])[0][:3]
        grasp_vertices = np.load(grasp_paths[i])[0]

        object_vertices = np.transpose(object_vertices)
        object_com = np.mean(object_vertices,axis=0)
        
        grasp_deltas = grasp_vertices - object_com

        
        fname = grasp_paths[i].replace('points.npy','')

        grasp_deltas_save_dir = os.path.join('/home/ishaans/grasp_deltas_tta_v9',fname)
        np.save(grasp_deltas_save_dir,grasp_deltas)



    

def debug_tcdm():
    
    path2 = os.path.join('/home/ishaans/grasp_tta_output_v9')
    path = os.path.join('/home/ishaans/grasp_deltas_tta_v9')
    grasp_list = os.listdir(path)
    
    gname = grasp_list[0]

    

    object_type,task = gname.split('_')
    task = task.split('.')[0]
    mano_fn = object_type + '_' + task + 'points_mano_output.npy'
    mano_out = np.load(os.path.join(path2,mano_fn),allow_pickle=True).item()
    vertices = mano_out['vertices'][0]
    joints = mano_out['joints'][0]
    pose = mano_out['pose'][0]
    betas = mano_out['betas'][0]
    global_orient = mano_out['global_orient'][0]

        

    object_path = object_type + '_' + task + 'points_object.npy'
    object_vertices= np.load(os.path.join(path2,object_path))[0][:3]
    object_vertices = np.transpose(object_vertices)
    object_com = np.mean(object_vertices,axis=0)
    fing_pos = joints.detach().cpu().numpy() - object_com
        
        
    #e = suite.load(object_type, task); e.reset()
    #e.physics.data.qpos[1] -= 1; e.physics.forward()

    object_list = dir(mj_models)
    obj_name = [o for o in object_list if o.lower() == object_type + 'object'][0]

    object_model_cls = getattr(mj_models,obj_name)
    env = mj_models.TableEnv()
    env.attach(mj_models.Adroit(limp=True))
    object_model = object_model_cls()
    env.attach(object_model)
    
    physics = physics_from_mjcf(env)
    for i in range(500):
        physics.step()

    
    
    com = physics.named.data.xipos[object_type + '/object'][None]
    
    fing_pos += physics.named.data.xipos[object_type + '/object'][None]

    images = []
    for i in range(16):
        env2 = mj_models.TableEnv()
        object_model = object_model_cls()
        env2.attach(object_model)
        env2.attach(mj_models.Adroit(limp=True))
        pos = np.array(physics.named.data.mocap_pos[i])
        pos[2] += 0.01
        env2.attach(mj_models.SphereDebugObject(pos=pos))
        #env2.attach(mj_models.SphereDebugObject(pos=fing_pos[0]))
        physics = physics_from_mjcf(env2)
        plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug1.png'),physics.render(camera_id=0, height=1080, width=1920))
        images.append(physics.render(camera_id=0, height=640, width=960))
    imageio.mimsave(os.path.join('sim_images','grasp_traj','joint_locations.gif'), images)
    clip = mp.VideoFileClip(os.path.join('sim_images','grasp_traj','joint_locations.gif'))
    clip.write_videofile(os.path.join('sim_images','grasp_traj','joint_locations.mp4'))

   

def start_tcdm():
    
    path2 = os.path.join('/home/ishaans/grasp_tta_output_v9')
    path = os.path.join('/home/ishaans/grasp_deltas_tta_v9')
    grasp_list = os.listdir(path)
    gname = grasp_list[0]

    object_type,task = gname.split('_')
    task = task.split('.')[0]
    mano_fn = object_type + '_' + task + 'points_mano_output.npy'
    mano_out = np.load(os.path.join(path2,mano_fn),allow_pickle=True).item()
    vertices = mano_out['vertices'][0]
    joints = mano_out['joints'][0]
    pose = mano_out['pose'][0]
    betas = mano_out['betas'][0]
    global_orient = mano_out['global_orient'][0]

    object_path = object_type + '_' + task + 'points_object.npy'
    object_vertices= np.load(os.path.join(path2,object_path))[0][:3]
    object_vertices = np.transpose(object_vertices)
    object_com = np.mean(object_vertices,axis=0)
    fing_pos = joints.detach().cpu().numpy() - object_com

    object_list = dir(mj_models)
    obj_name = [o for o in object_list if o.lower() == object_type + 'object'][0]
    object_model_cls = getattr(mj_models,obj_name)
    env = mj_models.TableEnv()
    env.attach(mj_models.Adroit(limp=True))
    object_model = object_model_cls()
    env.attach(object_model)
    #for i in range(len(fing_pos)):
    #env.attach(mj_models.SphereDebugObject(pos=fing_pos[i]))

    physics = physics_from_mjcf(env)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])
    for i in range(500):
        physics.step() #Makes the object fall on the table 

        
    #e = suite.load(object_type, task); e.reset()
    #e.physics.data.qpos[1] -= 1; e.physics.forward()
    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug1.png'),physics.render(camera_id=0, height=1080, width=1920))
    

    com = physics.named.data.xipos[object_type + '/object'][None]
    fing_pos += physics.named.data.xipos[object_type + '/object'][None]

    env2 = mj_models.TableEnv()
    env2.attach(mj_models.Adroit(limp=True))
    object_model = object_model_cls()
    env2.attach(object_model)
    
    for i in range(len(fing_pos)):
        env2.attach(mj_models.SphereDebugObject(pos=fing_pos[i]))
    physics = physics_from_mjcf(env2)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])

    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug2.png'),physics.render(camera_id=0, height=1080, width=1920))


    
    env3 = mj_models.TableEnv()
    object_model = object_model_cls()
    env3.attach(object_model)
    env3.attach(mj_models.Adroit(limp=True))
    physics = physics_from_mjcf(env3)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])
    for i in range(500):
        physics.step()


    
    start_pos = physics.body_poses.pos
    images = []
    linspaces = np.linspace(0, 1, num=500)
    for j in range(len(linspaces)):
        l = linspaces[j]
        target = l * fing_pos + (1 - l) * start_pos
        for i, p in enumerate(target):
            physics.named.data.mocap_pos['j{}_mocap'.format(i)] = p
        physics.step()

        images.append(physics.render(camera_id=0, height=128, width=128))

    #imageio.mimsave(os.path.join('sim_images','grasp_traj','grasp.gif'), images)
    #clip = mp.VideoFileClip(os.path.join('sim_images','grasp_traj','grasp.gif'))
    #clip.write_videofile(os.path.join('sim_images','grasp_traj','grasp.mp4'))

    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug3.png'),physics.render(camera_id=0, height=1080, width=1920))
    
    

    env4 = mj_models.TableEnv()
    object_model = object_model_cls()
    env4.attach(object_model)
    env4.attach(mj_models.Adroit(limp=True))
    physics = physics_from_mjcf(env4)
    


    cur_state = physics.data.qpos.copy()
    
    
    exp_data = np.load(os.path.join('/home/ishaans/TCDM_dev/trajectories', object_type + '_' + task + '.npz'), allow_pickle=True)
    exp_data = {k: v for k, v in exp_data.items()}
    exp_data['s_0'] = exp_data['s_0'][()]

    pg = cur_state.copy()
    pg[-6:] = exp_data['s_0']['initialized']['position'][-6:]
    physics.data.qpos[:] = pg
    physics.forward()

    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug4.png'),physics.render(camera_id=0, height=1080, width=1920))
    

    import ipdb
    ipdb.set_trace()
    mano_to_adroit = {}
    for i in range(len(16)):
        i,j = 0,0
    for i, p in enumerate(physics.body_poses.pos):
        physics.named.data.mocap_pos['j{}_mocap'.format(index)] = p
    for _ in range(150):
        physics.step()

    
    
    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug5.png'),physics.render(camera_id=0, height=1080, width=1920))


def temp_tcdm():
    
    path2 = os.path.join('/home/ishaans/grasp_tta_output_v9')
    path = os.path.join('/home/ishaans/grasp_deltas_tta_v9')
    grasp_list = os.listdir(path)
    gname = grasp_list[11]
    
    object_type,task = gname.split('_')
    task = task.split('.')[0]
    mano_fn = object_type + '_' + task + 'points_mano_output.npy'
    mano_out = np.load(os.path.join(path2,mano_fn),allow_pickle=True).item()
    vertices = mano_out['vertices'][0]
    joints = mano_out['joints'][0]
    pose = mano_out['pose'][0]
    betas = mano_out['betas'][0]
    global_orient = mano_out['global_orient'][0]

    object_path = object_type + '_' + task + 'points_object.npy'
    object_vertices= np.load(os.path.join(path2,object_path))[0][:3]
    object_vertices = np.transpose(object_vertices)
    object_com = np.mean(object_vertices,axis=0)
    fing_pos = joints.detach().cpu().numpy() - object_com

    object_list = dir(mj_models)
    obj_name = [o for o in object_list if o.lower() == object_type + 'object'][0]
    object_model_cls = getattr(mj_models,obj_name)
    env = mj_models.TableEnv()
    env.attach(mj_models.Adroit(limp=True))
    object_model = object_model_cls(pos=np.array([0.0,0.0,0.5]))
    env.attach(object_model)
    #for i in range(len(fing_pos)):
    #env.attach(mj_models.SphereDebugObject(pos=fing_pos[i]))

    physics = physics_from_mjcf(env)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])
    
    
    #e = suite.load(object_type, task); e.reset()
    #e.physics.data.qpos[1] -= 1; e.physics.forward()
    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug_v1.png'),physics.render(camera_id=0, height=1080, width=1920))
    

    com = physics.named.data.xipos[object_type + '/object'][None]
    fing_pos += physics.named.data.xipos[object_type + '/object'][None]

    env2 = mj_models.TableEnv()
    env2.attach(mj_models.Adroit(limp=True))
    object_model = object_model_cls(pos=np.array([0.0,0.0,0.5]))
    env2.attach(object_model)
    
    for i in range(len(fing_pos)):
        env2.attach(mj_models.SphereDebugObject(pos=fing_pos[i]))
    physics = physics_from_mjcf(env2)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])

    


    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug_v2.png'),physics.render(camera_id=0, height=1080, width=1920))


    env3 = mj_models.TableEnv()
    object_model = object_model_cls(pos=np.array([0.0,0.0,0.5]))
    env3.attach(object_model)
    env3.attach(mj_models.Adroit(limp=True))
    physics = physics_from_mjcf(env3)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])
    
    start_pos = physics.body_poses.pos
    
    images = []
    linspaces = np.linspace(0, 1, num=1000)
    for j in range(len(linspaces)):
        l = linspaces[j]
        target = l * fing_pos + (1 - l) * start_pos
        for i, p in enumerate(target):
            physics.named.data.mocap_pos['j{}_mocap'.format(i)] = p
        physics.step()

        images.append(physics.render(camera_id=0, height=128, width=128))
    import ipdb
    ipdb.set_trace()

    imageio.mimsave(os.path.join('sim_images','grasp_traj','grasp.gif'), images)
    clip = mp.VideoFileClip(os.path.join('sim_images','grasp_traj','grasp.gif'))
    clip.write_videofile(os.path.join('sim_images','grasp_traj','grasp.mp4'))

    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug_v3.png'),physics.render(camera_id=0, height=1080, width=1920))



def afford_tcdm():
    path = '/home/ishaans/afford_dex_pass/output/release/layout/cascade/recon/mocap'
    joint_files = os.listdir(path)
    joint_fn = '0000_07_s2_prediction_result.pkl'
    joints = np.load(os.path.join(path,joint_fn),allow_pickle=True)
    joints = joints['pred_output_list'][0]['right_hand']['pred_joints_smpl']
    #indices = [i for i in range(21) if i not in [4,8,12,16,20]]

    map_idx = {0:0,1:13,2:14,3:15,5:1,6:2,7:3,9:4,10:5,11:6,13:10,14:11,15:12,17:7,18:8,19:9}

    # map_idx = {0:0,1:1,2:2,3:3,5:4,6:5,7:6,9:7,10:8,11:9,13:10,14:11,15:12,17:13,18:14,19:15}
    inv_map = {v:k for k,v in map_idx.items()}
    

    new_joints = [joints[inv_map[i]] for i in range(len(inv_map))]
    #import ipdb
    #ipdb.set_trace()
    joints = np.array(new_joints)

    suffix = 'cup_drink'
    object_type,task = suffix.split('_')
    task = task.split('.')[0]

   
    fing_pos = joints - np.mean(joints)

    
    object_list = dir(mj_models)
    obj_name = [o for o in object_list if o.lower() == object_type + 'object'][0]
    object_model_cls = getattr(mj_models,obj_name)
    env = mj_models.TableEnv()
    env.attach(mj_models.Adroit(limp=True))
    object_model = object_model_cls(pos=np.array([0.0,0.0,0.5]))
    env.attach(object_model)
    #for i in range(len(fing_pos)):
    #env.attach(mj_models.SphereDebugObject(pos=fing_pos[i]))

    physics = physics_from_mjcf(env)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])

    #import ipdb
    #ipdb.set_trace()
    
    
    #e = suite.load(object_type, task); e.reset()
    #e.physics.data.qpos[1] -= 1; e.physics.forward()
    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug_v1_afford.png'),physics.render(camera_id=0, height=1080, width=1920))
    

    com = physics.named.data.xipos[object_type + '/object'][None]
    fing_pos += physics.named.data.xipos[object_type + '/object'][None]
    #fing_pos += np.array([0,0,0.01])

    env2 = mj_models.TableEnv()
    env2.attach(mj_models.Adroit(limp=True))
    object_model = object_model_cls(pos=np.array([0.0,0.0,0.5]))
    #env2.attach(object_model)
    
    for i in range(len(fing_pos)):
        env2.attach(mj_models.SphereDebugObject(pos=fing_pos[i]))
    physics = physics_from_mjcf(env2)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])

    


    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug_v2_afford.png'),physics.render(camera_id=0, height=1080, width=1920))


    env3 = mj_models.TableEnv()
    object_model = object_model_cls(pos=np.array([0.0,0.0,0.5]))
    env3.attach(object_model)
    env3.attach(mj_models.Adroit(limp=True))
    physics = physics_from_mjcf(env3)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])
    
    start_pos = physics.body_poses.pos
    images = []
    linspaces = np.linspace(0, 1, num=1000)
    for j in range(len(linspaces)):
        l = linspaces[j]
        target = l * fing_pos + (1 - l) * start_pos
        for i, p in enumerate(target):
            physics.named.data.mocap_pos['j{}_mocap'.format(i)] = p
        physics.step()

        images.append(physics.render(camera_id=0, height=128, width=128))

    import ipdb
    ipdb.set_trace()

    imageio.mimsave(os.path.join('sim_images','grasp_traj','grasp_afford.gif'), images)
    clip = mp.VideoFileClip(os.path.join('sim_images','grasp_traj','grasp_afford.gif'))
    clip.write_videofile(os.path.join('sim_images','grasp_traj','grasp_afford.mp4'))

    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug_v3_afford.png'),physics.render(camera_id=0, height=1080, width=1920))



def test_tcdm():
    
    path2 = os.path.join('/home/ishaans/grasp_tta_output_v9')
    path = os.path.join('/home/ishaans/grasp_deltas_tta_v9')
    grasp_list = os.listdir(path)
    gname = grasp_list[11]
    

    object_type,task = gname.split('_')
    task = task.split('.')[0]
    mano_fn = object_type + '_' + task + 'points_mano_output.npy'
    mano_out = np.load(os.path.join(path2,mano_fn),allow_pickle=True).item()
    vertices = mano_out['vertices'][0]
    joints = mano_out['joints'][0]
    pose = mano_out['pose'][0]
    betas = mano_out['betas'][0]
    global_orient = mano_out['global_orient'][0]

    object_path = object_type + '_' + task + 'points_object.npy'
    object_vertices= np.load(os.path.join(path2,object_path))[0][:3]
    object_vertices = np.transpose(object_vertices)
    object_com = np.mean(object_vertices,axis=0)
    fing_pos = joints.detach().cpu().numpy() - object_com

    beta = -np.pi/2
    rot_mat_y = np.array([[np.cos(beta),0,np.sin(beta)],
                         [0, 1, 0],
                         [-np.sin(beta), 0, np.cos(beta)]])

    alpha = np.pi/2
    rot_mat= np.array([[1,0,0],
                         [0, np.cos(alpha), -np.sin(alpha)],
                         [0, np.sin(alpha), np.cos(alpha)]])

    fing_pos = np.matmul(fing_pos,rot_mat)
    #fing_pos = np.matmul(fing_pos,rot_mat_y)

    object_list = dir(mj_models)
    obj_name = [o for o in object_list if o.lower() == object_type + 'object'][0]
    import ipdb
    ipdb.set_trace()
    object_model_cls = getattr(mj_models,obj_name)
    env = mj_models.TableEnv()
    env.attach(mj_models.Adroit(limp=True))
    object_model = object_model_cls(pos=np.array([0.0,0.0,0.5]))
    env.attach(object_model)
    #for i in range(len(fing_pos)):
    #env.attach(mj_models.SphereDebugObject(pos=fing_pos[i]))

    physics = physics_from_mjcf(env)
    for i in range(200):
        physics.step()
    
    
    #e = suite.load(object_type, task); e.reset()
    #e.physics.data.qpos[1] -= 1; e.physics.forward()
    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug_u3.png'),physics.render(camera_id=0, height=1080, width=1920))
    

    com = physics.named.data.xipos[object_type + '/object'][None]
    fing_pos += physics.named.data.xipos[object_type + '/object'][None]
    fing_pos += np.array([0,0,0.1])

    #env2 = mj_models.TableEnv()
    #env2.attach(mj_models.Adroit(limp=True))
    #object_model = object_model_cls(pos=np.array([0.0,0.0,0.5]))
    #env2.attach(object_model)
    
    for i in range(len(fing_pos)):
        env.attach(mj_models.SphereDebugObject(pos=fing_pos[i]))
    physics = physics_from_mjcf(env)
    #physics.model.opt.gravity = np.array([0.0,0.0,0.0])

    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug_u4.png'),physics.render(camera_id=0, height=1080, width=1920))

    #import ipdb
    #ipdb.set_trace()

    """
    env3 = mj_models.TableEnv()
    object_model = object_model_cls(pos=np.array([0.0,0.0,0.5]))
    env3.attach(object_model)
    env3.attach(mj_models.Adroit(limp=True))
    physics = physics_from_mjcf(env3)
    physics.model.opt.gravity = np.array([0.0,0.0,0.0])
    
    start_pos = physics.body_poses.pos
    images = []
    linspaces = np.linspace(0, 1, num=1000)
    for j in range(len(linspaces)):
        l = linspaces[j]
        target = l * fing_pos + (1 - l) * start_pos
        for i, p in enumerate(target):
            physics.named.data.mocap_pos['j{}_mocap'.format(i)] = p
        physics.step()

        images.append(physics.render(camera_id=0, height=128, width=128))

    imageio.mimsave(os.path.join('sim_images','grasp_traj','grasp2.gif'), images)
    clip = mp.VideoFileClip(os.path.join('sim_images','grasp_traj','grasp2.gif'))
    clip.write_videofile(os.path.join('sim_images','grasp_traj','grasp2.mp4'))

    plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug_u3.png'),physics.render(camera_id=0, height=1080, width=1920))
    """
    
        


def translate_tcdm():
    #_DEFAULT_GLOBAL = Quaternion(axis=[0,0,1], angle=np.pi)
    _MAX_TRIES = 10
    _STEPS_PER_TRY=2000
    _STOP_THRESH = 0.05
    _OBJ_DELTA_THRESH = 0.001
    _HUMAN_FPS = 120
    _ADROIT_FPS=25
    PARENTS = [None, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    path2 = os.path.join('/home/ishaans/grasp_tta_output_v9')
    path = os.path.join('/home/ishaans/grasp_deltas_tta_v9')
    grasp_list = os.listdir(path)

    for gname in grasp_list[:1]:

        object_type,task = gname.split('_')
        task = task.split('.')[0]
        mano_fn = object_type + '_' + task + 'points_mano_output.npy'
        mano_out = np.load(os.path.join(path2,mano_fn),allow_pickle=True).item()
        vertices = mano_out['vertices'][0]
        joints = mano_out['joints'][0]
        pose = mano_out['pose'][0]
        betas = mano_out['betas'][0]
        global_orient = mano_out['global_orient'][0]

        

        object_path = object_type + '_' + task + 'points_object.npy'
        object_vertices= np.load(os.path.join(path2,object_path))[0][:3]
        object_vertices = np.transpose(object_vertices)
        object_com = np.mean(object_vertices,axis=0)
        fing_pos = joints.detach().cpu().numpy() - object_com
        
        
        #e = suite.load(object_type, task); e.reset()
        #e.physics.data.qpos[1] -= 1; e.physics.forward()

        object_list = dir(mj_models)
        obj_name = [o for o in object_list if o.lower() == object_type + 'object'][0]

        object_model_cls = getattr(mj_models,obj_name)
        env = mj_models.TableEnv()
        env.attach(mj_models.Adroit(limp=True))
        object_model = object_model_cls()
        env.attach(object_model)
        
        physics = physics_from_mjcf(env)
        for i in range(500):
            physics.step()

        
        
        com = physics.named.data.xipos[object_type + '/object'][None]
        
        fing_pos += physics.named.data.xipos[object_type + '/object'][None]


        env2 = mj_models.TableEnv()
        object_model = object_model_cls()
        env2.attach(object_model)
        env2.attach(mj_models.Adroit(limp=True))
        for i in range(len(fing_pos)):
            env2.attach(mj_models.SphereDebugObject(pos=fing_pos[i]))
        physics = physics_from_mjcf(env2)

        import ipdb
        ipdb.set_trace()

        plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug1.png'),physics.render(camera_id=0, height=1080, width=1920))
        
        
        env3 = mj_models.TableEnv()
        object_model = object_model_cls()
        env3.attach(object_model)
        env3.attach(mj_models.Adroit(limp=True))
        physics = physics_from_mjcf(env3)
        for i in range(100):
            physics.step()




        physics.forward()
        start_pos = physics.body_poses.pos
        images = []
        linspaces = np.linspace(0, 1, num=500)
        for j in range(len(linspaces)):
            l = linspaces[j]
            target = l * fing_pos + (1 - l) * start_pos
            for i, p in enumerate(target):
                physics.named.data.mocap_pos['j{}_mocap'.format(i)] = p
            physics.step()

            images.append(physics.render(camera_id=0, height=128, width=128))

        imageio.mimsave(os.path.join('sim_images','grasp_traj','grasp.gif'), images)
        clip = mp.VideoFileClip(os.path.join('sim_images','grasp_traj','grasp.gif'))
        clip.write_videofile(os.path.join('sim_images','grasp_traj','grasp.mp4'))

        plt.imsave(os.path.join('sim_images','grasp_traj','sphere_debug2.png'),physics.render(camera_id=0, height=1080, width=1920))
        

        cur_state = physics.data.qpos.copy()
        
        #im_name = object_type + '_' + task + 'tcdm_image.png'
        #plt.imsave(os.path.join('sim_images',im_name),physics.render(camera_id=0, height=1080, width=1920))
        
        exp_data = np.load(os.path.join('/home/ishaans/TCDM_dev/trajectories', object_type + '_' + task + '.npz'), allow_pickle=True)
        exp_data = {k: v for k, v in exp_data.items()}
        exp_data['s_0'] = exp_data['s_0'][()]

        pg = cur_state.copy()
        pg[-6:] = exp_data['s_0']['initialized']['position'][-6:]

        physics.data.qpos[:] = pg
        physics.forward()
        mano_to_adroit = {}
        for i in range(len(16)):
            i,j = 0,0
        for i, p in enumerate(physics.body_poses.pos):
            if i == 0:
                index = 15
            else:
                index = i-1
            physics.named.data.mocap_pos['j{}_mocap'.format(index)] = p
        for _ in range(150):
            physics.step()

        
        
        im_name = object_type + '_' + task + 'tcdm_image.png'
        plt.imsave(os.path.join('sim_images',im_name),physics.render(camera_id=0, height=1080, width=1920))
        
        
        


def update_traj_file(afford_file,traj_file):
    import shutil
    traj_data = np.load(traj_file,allow_pickle=True)
    data = traj_data['s_0'].item()
    afford_qpos = np.load(afford_file,allow_pickle=True)
    data['afford_dex_pass'] = data['motion_planned'] #Could also use initialized here
    data['afford_dex_pass']['position'][:30] = afford_qpos[6:]
    new_npz = {k:traj_data[k] for k in traj_data}
    new_npz['s_0'] = np.array(data)
    shutil.copyfile(traj_file, traj_file.split('.npz')[0] + '_old.npz')
    np.savez(traj_file,**new_npz)


        


if __name__ == '__main__':
    #get_meshes_grasptta()
    
    #afford_tcdm()
    update_traj_file('sim_images/alarmclock_afford.npy','trajectories/alarmclock_lift_old_temp.npz')
    import ipdb
    ipdb.set_trace()