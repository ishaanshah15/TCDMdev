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

def get_grasp(task,pred_true=True):
    path = os.path.join('/home/ishaans/TCDM_dev/train_grasp_classifier/analysis/tasks',task)
    afford_folders = os.listdir(path)
    grasp_paths = []
    for af in afford_folders:
        trials = os.listdir(os.path.join(path,af))
        for t in trials:
            fpaths = os.listdir(os.path.join(path,af,t))
            for fp in fpaths:
                grasp_paths += [os.path.join(path,af,t,fp)]

    

    overall_path = '/scratch/ishaans/grasp_data/finetune_all'
    final_paths = []
    for gp in grasp_paths:
        gpath = os.path.join(overall_path,'/'.join(gp.split('/')[-3:]))
        
        gp_con = os.listdir(gp)
        label = 1 if 'passed_test.txt' in gp_con else 0
        pred = 1 if 'pred_true.txt' in gp_con else 0
        if not pred_true:
            pred = 1 - pred
        if pred:
            final_paths.append([gpath,path_to_key(gpath),path_to_task(gpath),label,pred])

    
    np.random.shuffle(final_paths)


    
    return final_paths
    


def get_all_runs(tasks):
    runs = []
    for t in tasks:
        
        grasps = get_grasp(t)
        np.random.shuffle(grasps)
        runs += grasps[:5]

    

    print(len(runs))
    
    np.save('classified_grasps',runs)
    
    
    
def path_to_key(path):
    splits = path.split('/')
    pkey = f'{splits[-3]}_{splits[-1]}'
    return pkey


def path_to_task(path):
    task = path.split('/')[-2]
    task = task.replace('_','-') 
    return task


def run_scripts(start,end,unified=True):
    train_grasps = np.load('classified_grasps.npy')
    train_grasps = [list(t) for t in train_grasps]
    
    df = pd.DataFrame(train_grasps)
    
    #df = df.transpose()
    df.columns = ['path','gkey','task','passed_tests','classification']
    project_name_pre = 'icra_exp_unified_5_gen_pass'
    lines = []

    print('df size',len(df))
    
    if unified:
        task_keys = {}
        for i in range(start,end):
            row = df.iloc[[i]]
            gkey = row['gkey'].item()
            task = row['task'].item() 
            if task in task_keys:
                task_keys[task] += f'*{gkey}'
            else:
                task_keys[task] = gkey

        for task in task_keys:
            project_name = f'{project_name_pre}'
            lines.append(f"python train.py exp_name={task}  wandb.project={project_name} env.name={task} env.task_kwargs.pregrasp={task_keys[task]} & \n")
            lines.append("sleep 120 \n")

        if os.path.exists(f'run_{task}.sh'):
            os.remove(f'run_{task}.sh')
        with open (f'run_{task}.sh', 'w') as rsh:
            rsh.writelines(lines)

    else:
        for i in range(start,end):
            print(i)
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
    target_tasks = ['alarmclock_lift', 
             'duck_lift',
             'flashlight_lift',
             'mouse_lift',
             'spheremedium_lift',
             'stapler_lift',
             'toothbrush_lift',
             'toothpaste_lift',
             'watch_lift',
             'waterbottle_lift']


    target_tasks = ['airplane_pass1',
                    'banana_pass1',
                    'binoculars_pass1',
                    'elephant_pass1',
                    'eyeglasses_pass1',
                    'headphones_pass1',
                    'lightbulb_pass1']

    #target_tasks = np.random.choice(tasks,5,replace=False)
    
    get_all_runs(target_tasks)

    
    for i in range(1):
        run_scripts(0,35)
    
    #run_scripts(8,16)
    

    
    
    
    
