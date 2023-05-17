import cv2
from tcdm import suite
import os 
import numpy as np
from scipy.spatial.transform import Rotation
#TODO: Save out point clouds 


def run():
    path = os.path.join('trajectories')
    objects = os.listdir(path)
    count = 0

    for o in objects:
        count += 1
        try:
            object_str,task_str = o.split('_')
            task_str,_ = task_str.split('.')
            write_image(object_str,task_str)
            print(object_str,task_str)
        except:
            pass
        if count > np.inf:
            break


def write_image(object_str,task_str):
    e = suite.load(object_str, task_str); e.reset()

    # shift hand down by one meter
    e.physics.data.qpos[1] -= 1; e.physics.forward()


    # render and save out image
    img = e.physics.render(camera_id=2)

    cv2.imwrite('object_frames_back/' + object_str + '_' + task_str + '.png', img[:,:,::-1])


def get_camera_pose():
    e = suite.load('alarmclock', 'see1'); e.reset()
    e.physics.data.qpos[1] -= 1; e.physics.forward()

    import ipdb
    ipdb.set_trace()

    camera_values = ['cam_bodyid', 'cam_fovy', 'cam_ipd', 'cam_mat0', 'cam_mode', 'cam_pos', 'cam_pos0', 'cam_poscom0', 'cam_quat', 'cam_targetbodyid', 'cam_user']
    cam_pose = {}
    for v in camera_values:
        cam_pose[v] = getattr(e.physics.model,v)[1]
    np.save('camera_1_pose.npy',cam_pose)

    import ipdb
    ipdb.set_trace()


def replace_cascade_names(cascade_path,frames_path):
    cascade_paths = []
    cascade_paths.append(os.path.join(cascade_path,'inp'))
    cascade_paths.append(os.path.join(cascade_path,'mask'))
    cascade_paths.append(os.path.join(cascade_path,'mask_param'))
    cascade_paths.append(os.path.join(cascade_path,'overall'))
    cascade_paths.append(os.path.join(cascade_path,'recon','mocap'))
    cascade_paths.append(os.path.join(cascade_path,'recon','rendered'))
    cascade_paths.append(os.path.join(cascade_path,'superres'))


    for c in cascade_paths:
        replace_fnames(c,frames_path)


def replace_fnames(cascade_path,frames_path):
    fnames = os.listdir(frames_path)
    fnames = [f.split('.')[0] for f in fnames]
    fnames.sort()
    cnames = os.listdir(cascade_path)
    cnames = [c[:7] for c in cnames]
    cnames.sort()
    cnames = np.unique(cnames)

    
   
    assert len(fnames) == len(cnames)
    fc_map = {c:f for c,f in zip(cnames,fnames)}


    c0_names = os.listdir(cascade_path)

    for c0 in c0_names:
        c1 = fc_map[c0[:7]]
        c1 += c0[7:]
        os.rename(os.path.join(cascade_path,c0),os.path.join(cascade_path,c1))










def quat_to_euler():
    
    quat = [-0.33141357403559174, -0.1913417161825449, 0.4619397662556433, 0.8001031451912656]
    rot = Rotation.from_quat(quat)
    rot_euler = rot.as_euler('xyz', degrees=True)
    rot_euler2 = [180,0,60]
    r = Rotation.from_euler('xyz',rot_euler2,degrees=True)
    u = r.as_quat()
    rot2 = Rotation.from_quat(u)
    rot_euler3 = rot2.as_euler('xyz', degrees=True)

    print(u)
    import ipdb
    ipdb.set_trace()
    


    

if __name__ == '__main__':
    #quat_to_euler()
    #get_camera_pose()
    #run()
    replace_cascade_names('/home/ishaans/afford_dex_pass/output/release/layout/cascade',
                          '/home/ishaans/TCDM_dev/object_frames_back')