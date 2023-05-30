#!/usr/bin/env python3
import os
import math
import ipdb 
import numpy as np
import itertools
from PIL import Image as PIL_Image
from tcdm import suite
import open3d as o3d
import open3d
import robosuite.utils.transform_utils as T
from camera_utils_v2 import *

def generatePointCloud(sim,cam_id):

    img_width = 640
    img_height = 480

    pixl2cam = get_camera_intrinsic_matrix_id(sim,cam_id,img_height,img_width)
    cam2world = get_camera_extrinsic_matrix_id(sim,cam_id)
    world2pixl = get_camera_transform_matrix_id(
        sim=sim,
        cam_id=cam_id,
        camera_height=img_height,
        camera_width=img_width,
    )

    depth,segmentation = captureImage(sim,img_width,img_height,cam_id)
   
    pixl2world = np.linalg.inv(world2pixl)

    pixl_idx = [x for x in itertools.product(np.arange(img_height),np.arange(img_width))]


    tokens = np.unique(segmentation)

    
    pixl = []
    for t in tokens:
        if t > 69 and t < 80 :
            pixl += [x for x in pixl_idx if segmentation[x[0],x[1], 0] == t]
            
    pixl = np.array(pixl)

    if len(pixl) > 0:
        points = transform_from_pixels_to_world(pixl,depth,pixl2world)
    else:
        points = []

    return points,depth,segmentation

# Render and process an image
def captureImage(sim,img_width,img_height,cam_id):
    depth = sim.render(width=img_width, height=img_height, camera_id=cam_id, depth=True)
    segmentation = sim.render(width=img_width,height=img_height,camera_id=cam_id,segmentation=True)
    return depth,segmentation

def run():
    path = os.path.join('trajectories')
    objects = os.listdir(path)
   
    for o in objects:
        try_obj = True
        try:
            object_str,task_str = o.split('_')
            task_str,_ = task_str.split('.')
        except:
            try_obj = False

        if try_obj:
            write_image(object_str,task_str)
        print(object_str,task_str)
       


def write_image(object_str,task_str):
    e = suite.load(object_str, task_str); e.reset()
    e.physics.data.qpos[1] -= 1; e.physics.forward()
    sim = e.physics

    point_list = []
    for cam_id in range(0,5):
        points,depth,segmentation = generatePointCloud(sim,cam_id)
        point_list.append(points)
        print('num points',len(points))

        dir_name = 'object_point_clouds_v12/'
        path2 = dir_name + object_str + '_' + task_str + str(cam_id) + '.npy'
        path3 = dir_name + object_str + '_' + task_str  +  'segment' + str(cam_id) +'.npy'
        
        np.save(path2,depth)
        np.save(path3,segmentation)
        
    path = dir_name + object_str + '_' + task_str  + 'points' + '.npy'
    all_points = []
    for x in point_list:
        all_points += list(x)
    np.save(path,np.array(all_points))


if __name__ == '__main__':
    run()