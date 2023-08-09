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
from rgb_dataset import Dataset2D
import torch





def get_all_test_split2D(test1=True,test2=True):
    #parent_path = '/home/ishaans/TCDM_dev/grasp_gen_scripts/finetune_all'
    parent_path = '/scratch/ishaans/grasp_data/finetune_all'
    
    test_classes = []
    paths,_ = search(parent_path,search_str='pred_grasp.png',str_val1='finetune_all')
    
    
    for i in range(len(paths)):
        test1_path = os.path.join(paths[i],'passed_test1.txt')
        test2_path = os.path.join(paths[i],'passed_test2.txt')
        pc_path = os.path.join(paths[i],'pred_grasp.png')
        param_path = os.path.join(paths[i],'fing_pos.npy')
        if test1 and test2:
            passed_test = os.path.exists(test1_path) and os.path.exists(test2_path)
        elif test1:
            passed_test = os.path.exists(test1_path)
        elif test2:
            passed_test = os.path.exists(test2_path)
        else:
            passed_test=False
        
        task = pc_path.split('/')[-3]
        test_classes.append((passed_test,pc_path,param_path,os.path.exists(test1_path),
                            os.path.exists(test2_path),task))

    return test_classes


def get_folder_test_split2D(folder,test1=True,test2=True):
    #parent_path = '/home/ishaans/TCDM_dev/grasp_gen_scripts/finetune_all'
    parent_path = f'/scratch/ishaans/grasp_data/finetune_all/{folder}'
    
    test_classes = []
    paths,_ = search(parent_path,search_str='pred_grasp.png',str_val1='finetune_all')

    
    
    for i in range(len(paths)):
        test1_path = os.path.join(paths[i],'passed_test1.txt')
        test2_path = os.path.join(paths[i],'passed_test2.txt')
        pc_path = os.path.join(paths[i],'pred_grasp.png')
        param_path = os.path.join(paths[i],'fing_pos.npy')
        if test1 and test2:
            passed_test = os.path.exists(test1_path) and os.path.exists(test2_path)
        elif test1:
            passed_test = os.path.exists(test1_path)
        elif test2:
            passed_test = os.path.exists(test2_path)
        else:
            passed_test=False
        
        task = pc_path.split('/')[-3]
        test_classes.append((passed_test,pc_path,param_path,os.path.exists(test1_path),
                            os.path.exists(test2_path),task))

    return test_classes



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


def get_all_contents(rootdir):
    file_list = []
    for root, directories, file in os.walk(rootdir):
        for file in file:
            file_list.append([root,directories,file])
    return file_list


    

def get_dataset2d(use_params=False):
    dataset = get_all_test_split2D()

    """
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

    dataset = [dataset[i] for i in total_indices]
    """

    np.random.shuffle(dataset)

    num_samples = len(dataset)
    index = int(0.7*num_samples)
    train_samples = dataset[:index]
    val_test_samples = dataset[index:]

    index2 = int(0.5*len(val_test_samples))
    val_samples = val_test_samples[:index2]
    test_samples = val_test_samples[index2:]

    train_dataset3d = Dataset2D(train_samples,use_params=use_params)
    val_dataset3d = Dataset2D(val_samples,use_params=use_params)
    test_dataset3d = Dataset2D(test_samples,use_params=use_params)

    
    return train_dataset3d,val_dataset3d,test_dataset3d

def get_test_folders(test_folders,use_params=False):
    dataset = []
    for f in test_folders:
        dataset += get_folder_test_split2D(f)

    
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

    dataset = [dataset[i] for i in total_indices]
  
    
    test_dataset3d = Dataset2D(dataset,use_params=use_params)
    
    return test_dataset3d


def get_dataset2d_balanced(use_params=False):

    
    dataset = get_all_test_split2D()
  
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


    train_dataset3d = Dataset2D(train_samples,use_params=use_params)
    val_dataset3d = Dataset2D(val_samples,use_params=use_params)
    test_dataset3d = Dataset2D(test_samples,use_params=use_params)

    
    
    return train_dataset3d,val_dataset3d,test_dataset3d




if __name__ == '__main__':
    parent_path2 = '/home/ishaans/TCDM_dev/grasp_gen_scripts/pregrasp_cloud'
    train_dataset3d,test_dataset3d = get_dataset2d()
    im,lab = train_dataset3d.__getitem__(0)
    
