#!/usr/bin/env python3
import os
import math
import ipdb 
import numpy as np

from PIL import Image as PIL_Image
from tcdm import suite
import open3d as o3d
import open3d

def generatePointCloud(sim):

    img_width = 640
    img_height = 480

    #aspect_ratio = img_width/img_height
    # sim.model.cam_fovy[0] = 60
    #fovy = math.radians(60)
    #fovx = 2 * math.atan(math.tan(fovy / 2) * aspect_ratio)
    #fx = 1/math.tan(fovx/2.0)
    #fy = 1/math.tan(fovy/2.0)
    #cx = img_width/2
    #cy = img_height/2
    #cam_mat = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)

    cam_mat = get_cam_mat(img_width,img_height)

    depth,segmentation = captureImage(sim,img_width,img_height)
    #depth = 1 - (depth - np.min(depth))/(np.max(depth) - np.min(depth))
    #print(np.max(depth),np.min(depth))
    #depth = 230.0*np.flip(depth, axis=0)
    #depth = 250.0*depth
    o3d_cloud = segment_object(depth,segmentation,cam_mat)
    #o3d_cloud = scaleCloudXY(o3d_cloud)
    return o3d_cloud,depth,segmentation
    #o3d.visualization.draw_geometries([o3d_cloud])

def get_cam_mat(img_width,img_height,angle=60):
    fovy = math.radians(angle)
    f = img_height / (2 * math.tan(fovy / 2))
    cx = img_width/2
    cy = img_height/2
    cam_mat = o3d.camera.PinholeCameraIntrinsic(img_width, img_height, f, f, cx, cy)
    return cam_mat


# Render and process an image
def captureImage(sim,img_width,img_height):
    depth = sim.render(width=img_width, height=img_height, camera_id=1, depth=True)
    segmentation = sim.render(width=img_width,height=img_height,camera_id=1,segmentation=True)
    # 480x640 np array
    #depth = np.loadtxt("depth_image_rendered.npy").astype(np.float32)

    #flipped_depth = np.flip(depth, axis=0)
    #real_depth = depthimg2Meters(flipped_depth)
    return depth,segmentation

def segment_object(depth,smap,cam_mat):
    dmap = np.array(depth)
    dmap[smap[:,:,0] != 72] = 100
    o3d_depth = o3d.geometry.Image(dmap)
    cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, cam_mat)
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(np.asarray(cloud.points)[np.asarray(cloud.points)[:,2] < 100])
    #import ipdb
    #ipdb.set_trace()
    return new_cloud


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

    pc,depth,segmentation = generatePointCloud(sim)
    #import ipdb
    #ipdb.set_trace()
    dir_name = 'object_point_clouds_v6/'
    path = dir_name + object_str + '_' + task_str  + '.pcd'
    path2 = dir_name + object_str + '_' + task_str  + '.npy'
    path3 = dir_name + object_str + '_' + task_str  +  'segment' + '.npy'
    open3d.io.write_point_cloud(path,pc)
    np.save(path2,depth)
    np.save(path3,segmentation)


if __name__ == '__main__':
    run()