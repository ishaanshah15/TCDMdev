'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import glob

from tqdm import tqdm
from torch.utils.data import Dataset
from dataloader_3d import Dataset3D
from hal_dataloader import HalDataset3D
import torch



def get_all_test_split(paths,paths2,test1=True,test2=True):
    test_classes = []
    
    for i in range(len(paths)):
        test1_path = os.path.join(paths[i],'passed_test1.txt')
        test2_path = os.path.join(paths[i],'passed_test2.txt')
        
        pc_path = os.path.join(paths2[i],'point_cloud.npy')
        if test1 and test2:
            passed_test = os.path.exists(test1_path) and os.path.exists(test2_path)
        elif test1:
            passed_test = os.path.exists(test1_path)
        elif test2:
            passed_test = os.path.exists(test2_path)
        else:
            passed_test=False

        test_classes.append((passed_test,pc_path))

    return test_classes



def get_all_test_split_hal(test1=True,test2=True):
    parent_path = '/scratch/ishaans/grasp_data/finetune_all'
    object_pc_path = '/home/ishaans/TCDM_dev/object_point_clouds_v12'
    test_classes = []
    paths,_ = search(parent_path,search_str='fing_pos.npy',str_val1='finetune_all')
    
    for i in range(len(paths)):
        test1_path = os.path.join(paths[i],'passed_test1.txt')
        test2_path = os.path.join(paths[i],'passed_test2.txt')
        param_path = os.path.join(paths[i],'fing_pos.npy')
        task = paths[i].split('/')[-2]
        pc_path = os.path.join(object_pc_path,task + 'points.npy')
        if test1 and test2:
            passed_test = os.path.exists(test1_path) and os.path.exists(test2_path)
        elif test1:
            passed_test = os.path.exists(test1_path)
        elif test2:
            passed_test = os.path.exists(test2_path)
        else:
            passed_test=False
        test_classes.append((passed_test,param_path,pc_path))

    return test_classes


def get_test_paths():
    #parent_path1 = '/home/ishaans/TCDM_dev/grasp_gen_scripts/finetune_all'
    parent_path2 = '/home/ishaans/TCDM_dev/grasp_gen_scripts/pregrasp_cloud'
    paths2,paths1 = search(parent_path2)
    return paths1,paths2
    

def search(rootdir,str_val1='pregrasp_cloud',str_val2 = 'finetune_all',search_str='point_cloud.npy'):
    file_list = []
    path1 = []
    for root, directories, file in os.walk(rootdir):
        for file in file:
            if file == search_str:
                file_list.append(root)
                dir2 = root.replace(str_val1,str_val2)
                path1.append(dir2)
    return file_list,path1


def get_dataset3d(num_points):
    paths1,paths2 = get_test_paths()
    dataset = get_all_test_split(paths1,paths2)

    labels = np.array([int(d[0]) for d in dataset])
    indices = np.arange(len(labels))
    zero_indices = indices[labels == 0]
    one_indices = indices[labels == 1]

    np.random.shuffle(zero_indices)
    np.random.shuffle(one_indices)

    min_len = min(len(zero_indices),len(one_indices))

    

    zero_indices = zero_indices[:min_len]
    one_indices = one_indices[:min_len]

    total_indices = list(zero_indices) + list(one_indices)

    balanced_dataset = [dataset[i] for i in total_indices]
    unbalanced_dataset = [dataset[i] for i in indices if not (i in total_indices)]



    np.random.shuffle(balanced_dataset)
    num_samples = len(balanced_dataset)
    index = int(0.3*num_samples)
    train_samples = balanced_dataset[index:]
    #train_samples += unbalanced_dataset
    
    val_test_samples = balanced_dataset[:index]

    np.random.shuffle(val_test_samples)
    index2 = int(0.5*len(val_test_samples))
    test_samples = val_test_samples[:index2]
    val_samples = val_test_samples[index2:]


    np.random.shuffle(train_samples)





    train_dataset3d = Dataset3D(train_samples,num_points)
    val_dataset3d = Dataset3D(val_samples,num_points)
    test_dataset3d = Dataset3D(test_samples,num_points)




    return train_dataset3d,test_dataset3d


def get_dataloader_3d(batch_size,num_points):
    train_dataset3d,test_dataset3d = get_dataset3d(num_points)
    train_loader = torch.utils.data.DataLoader(train_dataset3d, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(train_dataset3d, batch_size=batch_size, shuffle=True)
    return train_loader,test_loader
    

def get_dataset_hal():
    dataset = get_all_test_split_hal()
    
    
    labels = np.array([int(d[0]) for d in dataset])
    
    indices = np.arange(len(labels))
    zero_indices = indices[labels == 0]
    one_indices = indices[labels == 1]

    np.random.shuffle(zero_indices)
    np.random.shuffle(one_indices)

    min_len = min(len(zero_indices),len(one_indices))

    zero_indices = zero_indices[:min_len]
    one_indices = one_indices[:min_len]
    
    total_indices = list(zero_indices) + list(one_indices)

    balanced_dataset = [dataset[i] for i in total_indices]
    unbalanced_dataset = [dataset[i] for i in indices if not (i in total_indices)]

    np.random.shuffle(balanced_dataset)
   

    num_samples = len(dataset)
    index = int(0.2*num_samples)
    train_samples = balanced_dataset[index:]
    #train_samples += unbalanced_dataset
    val_test_samples = balanced_dataset[:index]


    np.random.shuffle(val_test_samples)
    index2 = int(0.5*len(val_test_samples))
    val_samples = val_test_samples[:index2]
    test_samples = val_test_samples[index2:]

    

    

    np.random.shuffle(train_samples)

    num_points = 1000
    train_dataset3d = HalDataset3D(train_samples,num_points)
    test_dataset3d = HalDataset3D(test_samples,num_points,split='test')
    val_dataset3d = HalDataset3D(val_samples,num_points,split='val')

    return train_dataset3d,val_dataset3d,test_dataset3d





if __name__ == '__main__':
    #parent_path2 = '/home/ishaans/TCDM_dev/grasp_gen_scripts/pregrasp_cloud'
    #train_dataset3d,test_dataset3d = get_dataset2d()
    #im,lab = train_dataset3d.__getitem__(0)
    #test_classes = get_all_test_split2D()
    train_dataset3d,test_dataset3d = get_dataset_hal()
    import ipdb
    ipdb.set_trace()
    
