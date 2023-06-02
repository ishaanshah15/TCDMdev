import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def get_grasp_dict(path):
    sample_dict = {}
    grasp_test = {}
    afford_folders = os.listdir(path)
    tasks = [x for x in os.listdir(os.path.join(path,afford_folders[0])) if os.path.isdir(os.path.join(path,afford_folders[0],x))] 
    for t in tasks:
        grasp_test[t] = {}
        for af in afford_folders:
            grasp_test[t][af] = {}

    for af in afford_folders:
        for t in tasks:
            trials = [x for x in os.listdir(os.path.join(path,af,t)) if os.path.isdir(os.path.join(path,af,t,x))]
            for trial in trials:
                run = os.path.join(path,af,t,trial)
                run_files = os.listdir(run)
                
                grasp_test[t][af][trial] = {}

                if 'passed_test1.txt' in run_files:
                    grasp_test[t][af][trial]['test1'] = True
                elif 'failed_test1.txt' in run_files:
                    grasp_test[t][af][trial]['test1'] = False
                else:
                    grasp_test[t][af][trial]['test1'] = None

                if 'passed_test2.txt' in run_files:
                    grasp_test[t][af][trial]['test2'] = True
                elif 'failed_test2.txt' in run_files:
                    grasp_test[t][af][trial]['test2'] = False
                else:
                    grasp_test[t][af][trial]['test2'] = None

                grasp_test[t][af][trial]['path'] = run
                sample_dict[run] = [t,af,grasp_test[t][af][trial]['test1'],grasp_test[t][af][trial]['test2']]
    
    return grasp_test,sample_dict
        

def sample_grasps(task):
    parent_path = 'finetune_all_tests'
    grasp_dict,sample_dict = get_grasp_dict(parent_path)
    positive_paths,negative_paths = [],[]
    
    for p in sample_dict.keys():
        if sample_dict[p][0] != task:
            continue
        if sample_dict[p][2] and sample_dict[p][3]:
            positive_paths.append(p)
        elif sample_dict[p][2] and (sample_dict[p][3] == False):
            negative_paths.append(p)
    
    
    np.random.shuffle(positive_paths)
    np.random.shuffle(negative_paths)

    return positive_paths,negative_paths


def build_task(target_tasks):
    positive_paths,negative_paths = [],[]
    min_pos_paths,min_neg_paths = [],[]
    extra_pos_paths,extra_neg_paths = [],[]

    for t in target_tasks:
        pos_paths,neg_paths = sample_grasps(t)
        min_pos_paths += pos_paths[:3]
        min_neg_paths += neg_paths[:2]
        extra_pos_paths += pos_paths[3:8]
        extra_neg_paths += neg_paths[2:7]

    positive_paths += min_pos_paths
    negative_paths += min_neg_paths

    np.random.shuffle(extra_pos_paths)
    np.random.shuffle(extra_neg_paths)


    positive_paths += extra_pos_paths[:10]
    negative_paths += extra_neg_paths[:8]

    all_paths = positive_paths + negative_paths

    labels = [1]*len(positive_paths) + [0]*len(negative_paths)

    all_keys = [path_to_key(p) for p in all_paths]

    all_tasks = [path_to_task(p) for p in all_paths]

    train_grasps = [all_paths,all_keys,all_tasks,labels]

    return train_grasps


def path_to_key(path):
    splits = path.split('/')
    pkey = f'{splits[1]}_{splits[-1]}'
    return pkey


def path_to_task(path):
    task = path.split('/')[2]
    task = task.replace('_','-') 
    return task


def run_scripts(train_grasps,start=0,end=2):
    df = pd.DataFrame(train_grasps)
    df = df.transpose()
    df.columns = ['path','gkey','task','passed_tests']
    project_name = 'corl_experiments'
    for i in range(start,end):
        row = df.iloc[[i]]
        gkey = row['gkey']
        task = row['task']
        os.system(f"python train.py exp_name={gkey}  wandb.project={project_name} env.name={task} env.task_kwargs.pregrasp={gkey} &")




if __name__ == '__main__':
    target_tasks = ['mug_drink3','toothbrush_lift','lightbulb_pass1','banana_pass1','stamp_stamp1']
    #target_tasks = ['toothbrush_lift']
    train_grasps = build_task(target_tasks)
    run_scripts(train_grasps)

    
    
    
    
