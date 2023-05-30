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
import copy
import shutil
sys.path.append('.')
sys.path.append('..')
import numpy as np
from nnutils.src.jutils.hand_utils import ManopthWrapper
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import os, time
import argparse
from tcdm import suite
from tcdm.motion_util import to_quat, to_transform_mat, axis_angle_to_rot
from tcdm.envs.mujoco import physics_from_mjcf
from tcdm.envs import mj_models
import matplotlib.pyplot as plt
import mano
import imageio
from validate_grasp import ValidateGrasp
import moviepy.editor as mp




class AffordTCDM():

    def __init__(self,task_name,trial,mocap_path):
        self.task_name = task_name
        self.trial = trial
        self.parent_dir = 'finetune_vizes'
        self.pc_path = '/home/ishaans/TCDM_dev/object_point_clouds_v12/'
        self.mocap_path = mocap_path
        self.open_palm = 0.5

        grasp_str = mocap_path.split('/')[-2]

        self.dir = f'finetune_grasp_{grasp_str}'
        
        self.save_dir = os.path.join(self.parent_dir,self.dir,self.task_name,str(self.trial))

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.validate_grasp = ValidateGrasp(self.task_name,trial,self.dir,self.parent_dir)



    def point_cloud(self):
        path = os.path.join(self.pc_path,self.task_name + 'points.npy')
        object_cloud = np.transpose(np.load(path,allow_pickle=True).T.astype(np.float32))
        object_cloud = object_cloud - np.mean(object_cloud,axis=0) 
        min_z = np.min(object_cloud,axis=0)[-1]
        object_cloud -= np.array([0,0,min_z])
        return object_cloud

    def plot_pc(self,point_cloud):
        env2 = mj_models.TableEnv()
        env2.attach(mj_models.Adroit(limp=False))
        
        for i in range(len(object_cloud)):
            env2.attach(mj_models.SmallSphereDebugObject(pos=object_cloud[i]))
        physics = physics_from_mjcf(env2)
        physics.data.qpos[1] -= 1; physics.forward()
        physics.model.opt.gravity = np.array([0.0,0.0,0.0])
        plt.imsave(os.path.join(self.save_dir,f'pc_spheres.png'),physics.render(camera_id=0, height=1080, width=1920))

        
    def step1(self,traj_data,add_object=True):
        task = self.task_name
        env3 = mj_models.TableEnv()
        object_model2 = self.object_model_cls(pos=np.array([0.0,0.0,0.1]))
        object_model = mj_models.get_object(self.task_name.split('_')[0])()

        
        if add_object:
            env3.attach(object_model)
        env3.attach(mj_models.Adroit(limp=True))
        physics = physics_from_mjcf(env3)

        physics.data.qpos[:6] = traj_data['motion_planned']['position'][30:]

        physics.step()

        old_qpos = copy.deepcopy(physics.data.qpos[:6])

        old_com = copy.deepcopy(physics.named.data.xipos[self.object_type + '/object'][None])

        
        #self.fing_pos = self.compute_final_pos(start_pos,self.fing_pos)
        #self.fing_pos += np.array([0,0,0.1])
        images = []
        size=1000
        linspaces = np.linspace(0, 1, num=size)
        for dim in range(2):
            start_pos = physics.body_poses.pos
            z_fingpos = np.mean(self.fing_pos,axis=0)[2]
            z_startpos = np.mean(start_pos,axis=0)[2]
            end_pos = self.fing_pos - np.array([0,0,z_fingpos - z_startpos - 0.1])
            if dim == 1:
                end_pos = self.fing_pos
            for j in range(len(linspaces)):
                l = linspaces[j]
                target = l * end_pos + (1 - l) * start_pos
                for i, p in enumerate(target):
                    physics.named.data.mocap_pos['j{}_mocap'.format(i)] = p
                physics.step()
                if j%40 == 0:
                    images.append(physics.render(camera_id=0, height=128, width=128))
        
        for idx in range(1000):
            for i, p in enumerate(self.fing_pos):
                physics.named.data.mocap_pos['j{}_mocap'.format(i)] = p
            physics.step()
            if idx%40 == 0:
                images.append(physics.render(camera_id=0, height=128, width=128))

        if add_object:
            imageio.mimsave(os.path.join(self.save_dir,'grasp_obj.gif'), images)
            clip = mp.VideoFileClip(os.path.join(self.save_dir,f'grasp_obj.gif'))
            clip.write_videofile(os.path.join(self.save_dir,f'grasp_obj.mp4'))

        new_com = physics.named.data.xipos[self.object_type + '/object'][None]

        diff = new_com - old_com

        im1 = physics.render(camera_id=0, height=1080, width=1920)
        obj_str = f'grasp_w_object.png' if add_object else  f'grasp.png'
        plt.imsave(os.path.join(self.save_dir,obj_str),im1)
        out = physics.data.qpos[:]

        #physics.data.contact.dist
        return out


    """
    def step2(self,get_com=False):
        domain,task = self.task_name.split('_')
        if get_com:
            task_kwargs = {}
        else:
            task_kwargs = {'pregrasp': 'afford_dex_pass'}
        env = suite.load(domain, task, task_kwargs, {})  
        env.reset()
        
        plt.imsave(os.path.join(self.dir2,self.task_name,f'pregrasp_w_task_{self.trial}.png'),env.physics.render(camera_id=0, height=1080, width=1920))
    """

    def process_joints(self,joints):
        task = self.task_name
        map_idx = {0:0,1:13,2:14,4:15,5:1,6:2,8:3,9:4,10:5,12:6,13:10,14:11,16:12,17:7,18:8,20:9}
        inv_map = {v:k for k,v in map_idx.items()}
        new_joints = [joints[inv_map[i]] for i in range(len(inv_map))]
        joints = np.array(new_joints)
        fing_pos = joints - np.mean(joints)
        fing_pos[:,1] = -fing_pos[:,1]
        fing_pos[:,2] = -fing_pos[:,2]

        return fing_pos

    def extract_joints(self,path):
        
        mano_dict = np.load(os.path.join(path,self.task_name + f'_s{self.trial}_prediction_result.pkl'),allow_pickle=True)

        mano_dict = mano_dict['pred_output_list'][0]

        if 'pred_joints_smpl' in mano_dict['right_hand']:
            hand_key = 'right_hand'
        elif 'pred_joints_smpl' in mano_dict['left_hand']:
            hand_key = 'left_hand'
        else:
            return None
        
        joints = mano_dict[hand_key]['pred_joints_smpl']
        pose = mano_dict[hand_key]['pred_hand_pose']

        if hand_key == 'right_hand':
            pose = torch.tensor(pose)
            hand_wrapper = ManopthWrapper()

            rot = pose[:,:3]
            hA = pose[:,3:]
           

            
            #hA = hA + hand_wrapper.hand_mean

            hA = (self.open_palm*(hA + hand_wrapper.hand_mean))

            
            mesh, pJoints, = hand_wrapper(None, torch.cat([rot, hA], -1))
            
            trans = -pJoints[:, 5]
            mesh = mesh.update_padded(mesh.verts_padded() + trans.unsqueeze(1))
            pJoints += trans.unsqueeze(1)

            joints = pJoints[0].numpy()
            
            """
            mesh, joints, _ = hand_wrapper.to_palm(rot, hA, add_pca=True)
            joints = joints[0].numpy()
            """
            

        elif hand_key == 'left_hand':
            
            mean_x = np.mean(joints[:,0])
            joints[:,0] = -joints[:,0] + 2*mean_x

        return joints
    


    def run(self):
        #original hand is at [0,-0.2,0.2]

        #if not os.path.exists(os.path.join(self.dir2,self.task_name)):
        #os.makedirs(os.path.join(self.dir2,self.task_name))
        traj_data = np.load(os.path.join('/home/ishaans/TCDM_dev','trajectories',self.task_name + '.npz'),allow_pickle=True)
        traj_data = traj_data['s_0'].item()

        
        
        path = os.path.join(self.mocap_path,'recon/mocap')

        joints = self.extract_joints(path)

        if joints is None:
            return 

        path2 =  os.path.join(self.mocap_path,'recon/rendered')
        src_im = os.path.join(path2,f'{self.task_name}_s{self.trial}.jpg')
        dest_im = os.path.join(self.save_dir,f'pred_grasp.png')
        shutil.copyfile(src_im,dest_im)
        self.fing_pos = self.process_joints(joints)

        suffix = self.task_name
        self.object_type,task = suffix.split('_')
        task_type = task.split('.')[0]

        if self.object_type in ['dhand','dmanus']:
            self.object_type = task_type
        
        object_list = dir(mj_models)
        obj_name = [o for o in object_list if o.lower() == self.object_type + 'object'][0]
        self.object_model_cls = getattr(mj_models,obj_name)

        #com_mj = self.step2(get_com=True) 

        pc = self.point_cloud()

        com_pc = np.mean(pc,axis=0) 
        max_pc = np.max(pc,axis=0) 
        min_pc = np.min(pc,axis=0) 

        offset = np.array([com_pc[0],com_pc[1],max_pc[2]]) - np.mean(self.fing_pos,axis=0)
        
        self.fing_pos = self.fing_pos + offset        
        
        qpos = self.step1(traj_data,add_object=True)
        
        
        save_path = os.path.join(self.save_dir,f'qpos_v{self.trial}')
        np.save(save_path,qpos)
        traj_file = os.path.join('/home/ishaans/TCDM_dev','trajectories/' + self.task_name + '.npz')
        
        self.update_traj_file(save_path + '.npy',traj_file)

        self.validate_grasp.run()


        
    def update_traj_file(self,afford_file,traj_file):
        import copy
        import shutil
        if not os.path.exists(traj_file.split('.npz')[0] + '_old.npz'):
            shutil.copyfile(traj_file, traj_file.split('.npz')[0] + '_old.npz')
        traj_data = np.load(traj_file,allow_pickle=True)
        data = traj_data['s_0'].item()
        afford_qpos = np.load(afford_file,allow_pickle=True)
        data[f'{self.dir}_{self.trial}'] = copy.deepcopy(data['motion_planned']) #Could also use initialized here
        data[f'{self.dir}_{self.trial}']['position'][:30] = afford_qpos[6:]
        new_npz = {k:traj_data[k] for k in traj_data}
        new_npz['s_0'] = np.array(data)
        
        np.savez(traj_file,**new_npz)

       


if __name__ == '__main__':

    frames_path = '/home/ishaans/grasp_outputs/object_frames_back'

    tasks = os.listdir(frames_path)
    tasks = [t.split('.')[0] for t in tasks]
    tasks.remove('dhand_cup')
    tasks.remove('dhand_waterbottle')
    tasks.remove('dmanus_coffeecan')
    tasks.remove('dhand_binoculars')
    tasks.remove('dmanus_sim2real')
    tasks.remove('dhand_alarmclock')
    tasks.remove('dhand_elephant')
    tasks.remove('dmanus_crackerbox')
    tasks.remove('hammer_strike')
    tasks.remove('door_open')
    

    #tasks = ['alarmclock_lift','elephant_pass1','duck_lift','knife_chop1']
    tasks = ['alarmclock_lift','elephant_pass1','lightbulb_pass1','mouse_lift','watch_lift']

    mocap_paths = ['/home/ishaans/afford_dex_pass/output/release/layout/cascade_hijack_masked/',
                '/home/ishaans/afford_dex_pass/output/release/layout/cascade_size_06_ratio_1/',
                '/home/ishaans/afford_dex_pass/output/release/layout/cascade_size_06_ratio_1732/',
                '/home/ishaans/afford_dex_pass/output/release/layout/cascade_size_06_ratio_05774/',
                '/home/ishaans/afford_dex_pass/output/release/layout/cascade_size_075_ratio_1/',
                '/home/ishaans/afford_dex_pass/output/release/layout/cascade_size_075_ratio_1732/',
                '/home/ishaans/afford_dex_pass/output/release/layout/cascade_size_075_ratio_05774/']

    #mocap_paths = ['/home/ishaans/afford_dex_pass/output/release/layout/cascade_hijack_masked/',
    #'/home/ishaans/afford_dex_pass/output/release/layout/cascade_size_06_ratio_1/']

    
    for task in tasks:
        print(task)
        for mpath in mocap_paths:
            for trial in range(0,3):
                tcdm = AffordTCDM(task,trial,mpath)
                tcdm.run()