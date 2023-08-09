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


class Dataset3D(Dataset):
    def __init__(self, datapath, num_points, split='train', use_uniform_sample=False,use_params=False):
        root = 'data_3d'
        self.npoints = num_points
        self.uniform = use_uniform_sample
        self.num_category = 2
        self.use_params = use_params

         #[(c_1,path_1,param1),(c_2,path_2,param2) ..]
        self.datapath = datapath
        self.labels = np.array([int(d[0]) for d in datapath])
        print('The size of %s data is %d' % (split, len(self.datapath)))

        self.points = []
        self.labels = []
        for index in range(len(self.datapath)):
            pc,lab = self._get_item(index)
            self.points.append(pc)
            self.labels.append(lab)


    def __len__(self):
        return len(self.datapath)

    
    def register_points(self,index):
        self.list_points,self.target = self._get_item(index)


    def _get_item(self, index):
        
        fn = self.datapath[index]
        cls = self.datapath[index][0]
        label = np.array([cls]).astype(np.int32)
        point_set = np.load(fn[1]).astype(np.float32)
        

        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.use_params:
            param = np.load(fn[2])
            return (point_set,param),label[0]
        else:
            return point_set,label[0]



    def __getitem__(self, index):
        return self.points[index],self.labels[index]


if __name__ == '__main__':
    import torch

    data = Dataset3D('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)