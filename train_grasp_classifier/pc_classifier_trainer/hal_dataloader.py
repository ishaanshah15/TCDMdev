'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class HalDataset3D(Dataset):
    def __init__(self, datapath, num_points, split='train', use_uniform_sample=True,use_params=False,process_data=True):
        root = 'data_3d'
        self.npoints = num_points
        self.uniform = use_uniform_sample
        self.num_category = 2
        self.use_params = use_params

         #[(c_1,path_1,param1),(c_2,path_2,param2) ..]
        
        
        
        if not process_data:
            self.points,self.labels,self.tasks,self.finger_pos,self.datapath = np.load(f'data/{split}_data_hal_v5.npy',allow_pickle=True)
        else:
            self.datapath = datapath
            self.labels = np.array([int(d[0]) for d in datapath])
            print('The size of %s data is %d' % (split, len(self.datapath)))
            self.points = []
            self.labels = []
            self.tasks = []
            self.finger_pos = []
            for index in range(len(self.datapath)):
                print(index)
                pc,lab,task,params = self._get_item(index)
                params = params.reshape((48,))
                self.points.append(pc)
                self.labels.append(lab)
                self.tasks.append(task)
                self.finger_pos.append(params)
                #os.makedirs(os.path.join('saved_pc',str(task),str(index)))
                #np.save(os.path.join('saved_pc',str(task),str(index)),[pc,lab])
        
            np.save(f'data/{split}_data_hal_v5.npy',[self.points,self.labels,self.tasks,self.finger_pos,self.datapath])
        

    def __len__(self):
        return len(self.datapath)

    
    
    def _get_item(self, index):
        
        fn = self.datapath[index]
        cls = self.datapath[index][0]
        label = np.array([cls]).astype(np.int32)
        params = np.load(fn[1])
        fing_pc = finger_pc(params)
        point_set = np.load(fn[2]).astype(np.float32)

        fing_pc = add_offset(fing_pc,point_set)

        point_set = np.concatenate([point_set,np.zeros((len(point_set),1))],axis=1)
        fing_pc = np.concatenate([fing_pc,np.ones((len(fing_pc),1))],axis=1)
        

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            np.random.shuffle(point_set)
            point_set = point_set[0:self.npoints]
    
        point_set = np.concatenate([point_set,fing_pc])
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        task = self.datapath[index][1].split('/')[-3]

        return point_set,label[0],task,params



    def __getitem__(self, index):
        
        return {'points': self.points[index][:,:4].astype(np.float32),
                'labels': self.labels[index],
                'tasks': self.tasks[index],
                'finger_params':self.finger_pos[index],
                'datapath':self.datapath[index][1]}

    


def add_p(joint1,joint2,radius=0.006,num_points=50):
    points = []
    for _ in range(num_points):
        alpha = np.random.uniform()
        point = joint1*alpha + (1-alpha)*joint2
        points.append(point + 2*radius*(np.random.uniform(size=(1,3)) - 0.5))
    points = np.concatenate(points)
    return points


def add_offset(points,pc):
    com_pc = np.mean(pc,axis=0) 
    max_pc = np.max(pc,axis=0) 
    min_pc = np.min(pc,axis=0) 

    offset = np.array([com_pc[0],com_pc[1],max_pc[2]])
    points += offset
    return points


def subsample(pc,n_points,use_farthest=True):
    if use_farthest:
        obj_pc = farthest_point_sample(pc,n_points)
    else:
        np.random.shuffle(pc)
        obj_pc = obj_pc[:n_points]
    return obj_pc




def finger_pc(params):
    center = 0
    index = [1,2,3]
    middle = [4,5,6]
    ring = [10,11,12]
    little = [7,8,9]
    thumb = [13,14,15]

    fingers = [index,middle,ring,little,thumb]
    points = []
    for i in range(len(fingers)):
        finger = fingers[i]
        points += [add_p(params[center],params[finger[0]])]
        
        for j in range(len(finger)-1):
            points += [add_p(params[finger[j]],params[finger[j+1]])]

    points = np.concatenate(points)
    return points



if __name__ == '__main__':
    train_dataset3d = HalDataset3D(None,0,process_data=False)
    import ipdb
    ipdb.set_trace()
    #data = Dataset3D('/data/modelnet40_normal_resampled/', split='train')
    #DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    