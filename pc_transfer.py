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
        self.dir = 'transfer_pc'
        self.pc_path = '/home/ishaans/TCDM_dev/object_point_clouds_v12/'
        self.mocap_path = '/home/ishaans/afford_dex_pass/output/release/layout/cascade_hijack_masked/'
        
    def point_cloud(self):
        path = os.path.join(self.pc_path,self.task_name + 'points.npy')

        #object_cloud = np.transpose(np.load(path,allow_pickle=True)[0].astype(np.float32))[:,:3]

        object_cloud = np.transpose(np.load(path,allow_pickle=True).T.astype(np.float32))
        
        env2 = mj_models.TableEnv()
        env2.attach(mj_models.Adroit(limp=False))
        

        random_idx = np.arange(len(object_cloud))
        np.random.shuffle(random_idx)
        object_cloud = object_cloud[random_idx][:500]

        object_cloud = object_cloud - np.mean(object_cloud,axis=0) 

        min_z = np.min(object_cloud,axis=0)[-1]
        object_cloud -= np.array([0,0,min_z])

        print('com',np.mean(object_cloud,axis=0))


        """
        quat2 = np.array([-0.33141357403559174, -0.1913417161825449, 0.4619397662556433, 0.8001031451912656])
        quat1 = np.array([0.866025404, 0.5, 0, 0])

        quat = quat2 - quat1
        rot = Rotation.from_quat(quat)
        theta_x, theta_y, theta_z = rot.as_euler('xyz', degrees=False)
        theta_x, theta_y, theta_z = 0,0,0


        #theta_z = -np.pi/2 + np.pi/6
        rot_mat_x = np.array([[1,0,0],[0,np.cos(theta_x),-np.sin(theta_x)],[0,np.sin(theta_x),np.cos(theta_x)]])
        rot_mat_y = np.array([[np.cos(theta_y),0,np.sin(theta_y)],[0,1,0],[-np.sin(theta_y), 0, np.cos(theta_y)]])
        rot_mat_z = np.array([[np.cos(theta_z),-np.sin(theta_z),0],[np.sin(theta_z), np.cos(theta_z), 0], [0,0,1]])

        object_cloud = np.matmul(object_cloud,rot_mat_x)
        object_cloud = np.matmul(object_cloud,rot_mat_y)
        object_cloud = np.matmul(object_cloud,rot_mat_z)
        """
       
        for i in range(len(object_cloud)):
            env2.attach(mj_models.SmallSphereDebugObject(pos=object_cloud[i]))
        physics = physics_from_mjcf(env2)
        physics.data.qpos[1] -= 1; physics.forward()
        physics.model.opt.gravity = np.array([0.0,0.0,0.0])
        plt.imsave(os.path.join(self.dir,task,f'pc_spheres_{self.trial}.png'),physics.render(camera_id=0, height=1080, width=1920))

        return physics.render(camera_id=0, height=1080, width=1920)

       
    def step2(self):
        object_str, task_str = self.task_name.split('_')
        #add red spheres at the joint locations 

        e = suite.load(object_str, task_str); e.reset()

        # shift hand down by one meter
        e.physics.data.qpos[1] -= 1; e.physics.forward()

        plt.imsave(os.path.join(self.dir,task,'only_spheres.png'),e.physics.render(camera_id=0, height=1080, width=1920))

        return e.physics.render(camera_id=0, height=1080, width=1920)


    
    def run(self):
        #original hand is at [0,-0.2,0.2]

        if not os.path.exists(os.path.join(self.dir,self.task_name)):
            os.makedirs(os.path.join(self.dir,self.task_name))
        
        path = os.path.join(self.mocap_path,'recon/mocap')

       
        path2 =  os.path.join(self.mocap_path,'recon/rendered')

       
        src_im = os.path.join(path2,f'{self.task_name}_s{self.trial}.jpg')
        dest_im = os.path.join(self.dir,self.task_name,f'pred_grasp_{self.trial}.png')
        shutil.copyfile(src_im,dest_im)
        
        #indices = [i for i in range(21) if i not in [4,8,12,16,20]]

    
        suffix = self.task_name
        self.object_type,task = suffix.split('_')
        task_type = task.split('.')[0]

    
        object_list = dir(mj_models)
        
        
        obj_name = [o for o in object_list if o.lower() == self.object_type + 'object'][0]
        
       
        self.object_model_cls = getattr(mj_models,obj_name)

        im = self.step2()
        im2 = self.point_cloud()

        diff_im = im - im2

        diff_im[diff_im < 20] = 0

        diff_im[diff_im > 20] = 100
   
        plt.imsave(os.path.join(self.dir,self.task_name,'diff_im.png'),diff_im)

       


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
   
    #tasks = ['knife_chop1','lightbulb_pass1']

    for task in tasks:
        print(task)
        for trial in range(0,1):
            tcdm = AffordTCDM(task,trial)
            tcdm.run()