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
import shutil
sys.path.append('.')
sys.path.append('..')
import numpy as np
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
import moviepy.editor as mp




class AffordTCDM():

    def __init__(self,task_name,trial):
        self.task_name = task_name
        self.trial = trial
        self.threshold = 0.02
        self.dir = 'sim_images2'
        self.pc_path = '/home/ishaans/grasp_outputs/grasp_tta_output_v10'
        self.mocap_path = '/home/ishaans/afford_dex_pass/output/release/layout/cascade_hijack_masked/'


    def step1(self):
        #Spawn the object and find its coordinates
        task = self.task_name

        env = mj_models.TableEnv()
        env.attach(mj_models.Adroit(limp=True))
        object_model = self.object_model_cls(pos=np.array([0.0,0.0,0.1]))
        env.attach(object_model)
        
        physics = physics_from_mjcf(env)
        physics.model.opt.gravity = np.array([0.0,0.0,0.0])

        #plt.imsave(os.path.join('sim_images2',task,'sphere_debug_v1_afford.png'),physics.render(camera_id=0, height=1080, width=1920))

        com = physics.named.data.xipos[self.object_type + '/object'][None]

        start_pos = physics.body_poses.pos

        return com,start_pos

    

    def validate_joint_indices(self,start_pos,index=2):
        task = self.task_name
        #Spawn the object and find its coordinates
        env = mj_models.TableEnv()
        env.attach(mj_models.Adroit(limp=True))
        env.attach(mj_models.SphereDebugObject(pos=start_pos[index]))
        
        physics = physics_from_mjcf(env)
        physics.model.opt.gravity = np.array([0.0,0.0,0.0])

        plt.imsave(os.path.join(self.dir,task,'val_joint_indices.png'),physics.render(camera_id=0, height=1080, width=1920))

        
    def step2(self):
        task = self.task_name
        #add red spheres at the joint locations 

        env2 = mj_models.TableEnv()
        env2.attach(mj_models.Adroit(limp=True))
        object_model = self.object_model_cls(pos=np.array([0.0,0.0,0.1]))
        env2.attach(object_model)
        
        for i in range(len(self.fing_pos)):
            pass
            #env2.attach(mj_models.SphereDebugObject(pos=self.fing_pos[i]))
        physics = physics_from_mjcf(env2)
        physics.model.opt.gravity = np.array([0.0,0.0,0.0])

        plt.imsave(os.path.join(self.dir,task,'only_spheres.png'),physics.render(camera_id=0, height=1080, width=1920))


    def step3(self,add_object=True):
        task = self.task_name

        env3 = mj_models.TableEnv()
        object_model = self.object_model_cls(pos=np.array([0.0,0.0,0.1]))
        if add_object:
            env3.attach(object_model)
        env3.attach(mj_models.Adroit(limp=True))
        physics = physics_from_mjcf(env3)
        #physics.model.opt.gravity = np.array([0.0,0.0,0.0])
        
        start_pos = physics.body_poses.pos
        images = []
        linspaces = np.linspace(0, 1, num=1000)
        for j in range(len(linspaces)):
            l = linspaces[j]
            target = l * self.fing_pos + (1 - l) * start_pos
            for i, p in enumerate(target):
                physics.named.data.mocap_pos['j{}_mocap'.format(i)] = p
            physics.step()


            if j%4 == 0:
                images.append(physics.render(camera_id=0, height=128, width=128))

        

        if add_object:
            imageio.mimsave(os.path.join(self.dir,task,f'grasp_obj_v{self.trial}.gif'), images)
            clip = mp.VideoFileClip(os.path.join(self.dir,task,f'grasp_obj_v{self.trial}.gif'))
            clip.write_videofile(os.path.join(self.dir,task,f'grasp_obj_v{self.trial}.mp4'))


        im1 = physics.render(camera_id=0, height=1080, width=1920)

        obj_str = f'grasp_w_object_v{self.trial}.png' if add_object else  f'grasp_v{self.trial}.png'
        plt.imsave(os.path.join(self.dir,task,obj_str),im1)

        out = physics.data.qpos[:]

        
        return out,physics.body_poses.pos,im1

    
    

    
    def step4(self):
        out,last_pos,im = self.step3(add_object=True)
        task = self.task_name
        env3 = mj_models.TableEnv()
        env3.attach(mj_models.Adroit(limp=False))
        physics = physics_from_mjcf(env3)
        im1 = physics.render(camera_id=0, height=1080, width=1920)

        dists = np.sum(np.abs(self.fing_pos - last_pos),axis=1)
        for i in range(len(dists)):
            if dists[i] > self.threshold:
                env3.attach(mj_models.SphereDebugObject(pos=self.fing_pos[i]))
                pass
            else:
                env3.attach(mj_models.SphereDebuggObject(pos=self.fing_pos[i]))
                pass
        physics = physics_from_mjcf(env3)
        #physics.model.opt.gravity = np.array([0.0,0.0,0.0])
        im2 = physics.render(camera_id=0, height=1080, width=1920)

        v = im2 - im1
        v[v > 180] = 0
        mask = np.expand_dims(np.max(v > 0,axis=2),axis=2)
        
        new_im = im*(mask == 0) + v*(mask)
        plt.imsave(os.path.join(self.dir,task,f'grasp_w_spheres_v{self.trial}.png'),new_im)
        
        return out

    def hand_radius(self,qpos):
        """
        0 : Wrist
        1 : Thumb_00
        2 : Thumb_01
        3 : Thumb_02
        4 : Thumb_03
        5 : Index_00
        6 : Index_01
        7 : Index_02
        8 : Index_03
        9 : Middle_00
        10 : Middle_01
        11 : Middle_02
        12 : Middle_03
        13 : Ring_00
        14 : Ring_01
        15 : Ring_02
        16 : Ring_03
        17 : Little_00
        18 : Little_01
        19 : Little_02
        20 : Little_03
        """

        """
        0: wrist
        1 - 3: index
        4 - 6: middle
        7 - 9: little
        10 - 12: ring
        13 - 15: thumb
        """

        indices = [0,3,6,9,12]

        diff = []
        for i in range(15):
            diff.append(np.abs(qpos[i] - qpos[i+1]))
        
        diff = np.array(diff)
        return diff    


    def step5(self):

        domain,task = self.task_name.split('_')
        
        task_kwargs = {'pregrasp': 'afford_dex_pass'}

        env = suite.load(domain, task, task_kwargs, {})  
        env.reset()

        # shift hand down by one meter
        #env.physics.data.qpos[1] -= 1; 
        #env.physics.forward()


        plt.imsave(os.path.join(self.dir,self.task_name,'pregrasp_w_task.png'),env.physics.render(camera_id=0, height=1080, width=1920))
    
    def get_joints(self,joints):


        task = self.task_name
        map_idx = {0:0,1:13,2:14,4:15,5:1,6:2,8:3,9:4,10:5,12:6,13:10,14:11,16:12,17:7,18:8,20:9}

        # map_idx = {0:0,1:1,2:2,3:3,5:4,6:5,7:6,9:7,10:8,11:9,13:10,14:11,15:12,17:13,18:14,19:15}
        inv_map = {v:k for k,v in map_idx.items()}

        new_joints = [joints[inv_map[i]] for i in range(len(inv_map))]
        #import ipdb
        #ipdb.set_trace()
        joints = np.array(new_joints)
        fing_pos = joints - np.mean(joints)

        #theta_x,theta_y,theta_z = 0,-np.pi,-np.pi 
        theta_x,theta_y,theta_z = 0,0,0

        #TODO: Check rotation makes sense - is the origin correct
        #TODO: Run without guidance on object-frames back

        rot_mat_x = np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x)],[0,np.sin(theta_x),np.cos(theta_x)]])
        rot_mat_y = np.array([[np.cos(theta_y),0,np.sin(theta_y)],[0,1,0],[-np.sin(theta_y), 0, np.cos(theta_y)]])
        rot_mat_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0],[np.sin(theta_z), np.cos(theta_z), 0], [0,0,1]])

        fing_pos = np.matmul(fing_pos,rot_mat_x)
        fing_pos = np.matmul(fing_pos,rot_mat_y)
        fing_pos = np.matmul(fing_pos,rot_mat_z)

        

        fing_pos[:,1] = -fing_pos[:,1]
        fing_pos[:,2] = -fing_pos[:,2]
        
        return fing_pos

    def run(self):
        #original hand is at [0,-0.2,0.2]

        if not os.path.exists(os.path.join(self.dir,self.task_name)):
            os.makedirs(os.path.join(self.dir,self.task_name))
        
        path = os.path.join(self.mocap_path,'recon/mocap')

        joint_files = os.listdir(path)
        joints = np.load(os.path.join(path,self.task_name + f'_s{self.trial}_prediction_result.pkl'),allow_pickle=True)

        if not ('pred_joints_smpl' in joints['pred_output_list'][0]['right_hand']):
            joints = joints['pred_output_list'][0]['left_hand']['pred_joints_smpl']
            mean_x = np.mean(joints[:,0])
            joints[:,0] = -joints[:,0] + 2*mean_x

        else:
            joints = joints['pred_output_list'][0]['right_hand']['pred_joints_smpl']

        path2 =  os.path.join(self.mocap_path,'recon/rendered')
        src_im = os.path.join(path2,f'{self.task_name}_s{self.trial}.jpg')
        dest_im = os.path.join(self.dir,self.task_name,f'pred_grasp_{self.trial}.png')
        shutil.copyfile(src_im,dest_im)
        self.fing_pos = self.get_joints(joints)
        #indices = [i for i in range(21) if i not in [4,8,12,16,20]]

    
        suffix = self.task_name
        self.object_type,task = suffix.split('_')
        task_type = task.split('.')[0]

        if self.object_type in ['dhand','dmanus']:
            self.object_type = task_type
        
        
        object_list = dir(mj_models)
        
        
        obj_name = [o for o in object_list if o.lower() == self.object_type + 'object'][0]
        
       
        self.object_model_cls = getattr(mj_models,obj_name)

        com,start_pos = self.step1() 
        self.fing_pos += com
        #self.fing_pos += np.array([0.04,0.03,0.07])
        #self.validate_joint_indices(start_pos)
        #self.step2()
        #self.point_cloud()

        self.step3(add_object=False)
        out = self.step4()
        save_path = os.path.join(self.dir,self.task_name, self.task_name + f'_qpos_v{self.trial}')
        #np.save(save_path,out)
        traj_file = os.path.join('trajectories/' + self.task_name + '.npz')

        self.step5()
        
        diff_fing_pos = self.hand_radius(self.fing_pos)

        #self.update_traj_file(save_path + '.npy',traj_file)


        
    def update_traj_file(self,afford_file,traj_file):
        import copy
        import shutil
        if not os.path.exists(traj_file.split('.npz')[0] + '_old.npz'):
            shutil.copyfile(traj_file, traj_file.split('.npz')[0] + '_old.npz')
        traj_data = np.load(traj_file,allow_pickle=True)
        data = traj_data['s_0'].item()
        afford_qpos = np.load(afford_file,allow_pickle=True)
        data['afford_dex_pass'] = copy.deepcopy(data['motion_planned']) #Could also use initialized here
        data['afford_dex_pass']['position'][:30] = afford_qpos[6:]
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

    

    tasks = ['alarmclock_lift']

    for task in tasks:
        print(task)
        for trial in range(0,3):
            tcdm = AffordTCDM(task,trial)
            tcdm.run()