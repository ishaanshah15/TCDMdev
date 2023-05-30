## -*- coding: utf-8 -*-
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




class ValidateGrasp():

    def __init__(self,task_name,trial,dir,parent_dir):
        self.parent_dir = parent_dir
        self.task_name = task_name
        self.trial = trial
        self.dir = dir
        self.save_dir = os.path.join(self.parent_dir,self.dir,self.task_name,str(self.trial))

    
    def grasp_object(self):

        domain,task = self.task_name.split('_')
        
        task_kwargs = {'pregrasp': f'{self.dir}_{self.trial}'}
        env = suite.load(domain, task, task_kwargs)  
        env.reset()
    
        plt.imsave(os.path.join(self.save_dir,f'pregrasp.png'),env.physics.render(camera_id=0, height=1080, width=1920))

        images = []

        for i in range(100):
            env.physics.named.data.ctrl[6:] = 1
            env.physics.step()

            if i%7 == 0:
                images.append(env.physics.render(camera_id=0, height=128, width=128))
        plt.imsave(os.path.join(self.save_dir,f'closed_grasp.png'),env.physics.render(camera_id=0, height=1080, width=1920))


        # shift hand down by one meter
        #env.physics.data.qpos[1] -= 1; 
        #env.physics.forward()
        
        
        ctrl_lift = [ 0.25, -0.2, 0.12755218, -0.87627903, -0.39146814, 0.59359401]

        for i in range(500):
            
            for j in range(len(ctrl_lift)):
                
                env.physics.named.data.ctrl[j] = ctrl_lift[j]
            env.physics.named.data.ctrl[6:] = 1
            env.physics.step()

            if i%7 == 0:
                images.append(env.physics.render(camera_id=0, height=128, width=128))

        tokens = str(env.physics.named.data.ctrl).split(' ')
        tokens = [t.replace('/','_') for t in tokens if 'adroit' in t]
        
        imageio.mimsave(os.path.join(self.save_dir,f'lift_object.gif'), images)
        clip = mp.VideoFileClip(os.path.join(self.save_dir,f'lift_object.gif'))
        clip.write_videofile(os.path.join(self.save_dir,f'lift_object.mp4'))
        
        plt.imsave(os.path.join(self.save_dir,f'lift_object.png'),env.physics.render(camera_id=0, height=1080, width=1920))


    
    
    def run(self):
        self.grasp_object()
        

       


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
    


    #tasks = ['airplane_fly1']
    
    for task in tasks:
        print(task)
        
        tcdm = ValidateGrasp(task)
        tcdm.run()