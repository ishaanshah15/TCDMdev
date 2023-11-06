import os
import numpy as np
import pandas as pd
#from sklearn.utils import shuffle
import time
from argparse import ArgumentParser

"""
parser = ArgumentParser(description="Example code for loading pre-trained policies")
parser.add_argument('--start',type=int)
parser.add_argument('--end',type=int)
"""

def get_grasp_dict(path):
    sample_dict = {}
    grasp_test = {}
    afford_folders = os.listdir(path)
    tasks = [x for x in os.listdir(os.path.join(path,afford_folders[-2])) if os.path.isdir(os.path.join(path,afford_folders[-2],x))] 
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
    parent_path = 'grasp_gen_scripts/finetune_all'
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

        print(t,len(pos_paths),len(neg_paths))

    positive_paths += min_pos_paths
    negative_paths += min_neg_paths

    np.random.shuffle(extra_pos_paths)
    np.random.shuffle(extra_neg_paths)


    positive_paths += extra_pos_paths[:4]
    negative_paths += extra_neg_paths[:4]

    all_paths = positive_paths + negative_paths

    labels = [1]*len(positive_paths) + [0]*len(negative_paths)

    all_keys = [path_to_key(p) for p in all_paths]

    all_tasks = [path_to_task(p) for p in all_paths]

    train_grasps = [all_paths,all_keys,all_tasks,labels]

    print('#runs',len(train_grasps[0]))
    print(positive_paths)

    np.save('train_grasps',train_grasps)


def path_to_key(path):
    splits = path.split('/')
    pkey = f'{splits[-3]}_{splits[-1]}'
    return pkey


def path_to_task(path):
    task = path.split('/')[-2]
    task = task.replace('_','-') 
    return task


def run_scripts(start,end):
    train_grasps = np.load('train_grasps_2_objs.npy')
    df = pd.DataFrame(train_grasps)
    df = df.transpose()
    df.columns = ['path','gkey','task','passed_tests']
    project_name_pre = 'corl_experiments'
    lines = []

    for i in range(start,end):
        row = df.iloc[[i]]
        gkey = row['gkey'].item()
        task = row['task'].item() 
        
        passed_test = 'passed' if int(row['passed_tests'].item()) ==1 else 'failed'
        project_name = f'{project_name_pre}_{passed_test}'
        
        lines.append(f"python train.py exp_name={task}_{gkey}  wandb.project={project_name} env.name={task} env.task_kwargs.pregrasp={gkey} & \n")
        lines.append("sleep 120 \n")

    
    if os.path.exists(f'run_{start}_{end}.sh'):
        os.remove(f'run_{start}_{end}.sh')
    with open (f'run_{start}_{end}.sh', 'w') as rsh:
        rsh.writelines(lines)



if __name__ == '__main__':
    #args = parser.parse_args()
    target_tasks = ['mug_drink3','flute_pass1','headphones_pass1','stapler_lift']
    build_task(target_tasks)
    #run_scripts(0,10)
    #run_scripts(10,15)
    #run_scripts(24,36)
    

    
    
    
    
