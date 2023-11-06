# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import glob, yaml, os, imageio, cv2, shutil
import sys
sys.path.append('/home/ishaans/TCDM_dev/grasp_selection_scripts')
from grasp_for_rl import get_grasp
 
# setting path
from tcdm import suite
import torch
from stable_baselines3 import PPO
from argparse import ArgumentParser
import sys
#sys.path.append('/home/ishaans/TCDM_dev/')
from mj_pc_robosuite import get_pc
from rollout_generalization import grasp_helper,rollout
from dagger_simple.trainer import train_policy_2d
from dagger_simple.action_dataset import preprocess_frames,preprocess_all
from dagger_simple.policy_net import ResNetPolicy,ParamFC,CombinedNet
from dagger_simple.trainer import STATE,RGB,COMBINED
parser = ArgumentParser(description="Example code for loading pre-trained policies")

                                     
parser.add_argument('--render', action="store_true", help="Supply flag to render mp4")
parser.add_argument('--tasks', default='mouse_lift', help="tasks to train with - comma separated")


WEIGHTS_PATH = 'dagger_simple/saved_models/best_model_2d'
NUM_EPS=50
NOISE_FACTOR=0.0

MODEL_TYPE = 1

"""
Script to perform policy distillation from expert policy that uses privileged state information to a policy that uses only RGB images and goal coordinates as input.
"""


def render(writer, physics, AA=2, height=256, width=256):
    if writer is None:
        return
    img = physics.render(camera_id=0, height=height * AA, width=width * AA)
    writer.append_data(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))




def rollout_single_episode(save_folder, writer, pregrasp, policy, model, noise_factor=0.0, use_policy=False):

    buffer = []
    config =  yaml.safe_load(open(os.path.join(save_folder, 'exp_config.yaml'), 'r'))
    o, t = config['env']['name'].split('-')
    config['env']['task_kwargs']['pregrasp'] = pregrasp
    env = suite.load(o, t, config['env']['task_kwargs'], gym_wrap=True)
    

    # rollout the policy and print total reward
    s, done, total_reward = env.reset(), False, 0
    render(writer, env.wrapped.physics)
    counter = 0
    while not done:
        counter += 1

        images = []
        for id in range(7):
            im = env.wrapped.physics.render(camera_id=id, height=224, width=224)
            images.append(im)
        
        inp_rgb,inp_state,inp_comb = preprocess_all(images[:4],224,s)
       

        if type(model) == CombinedNet:
            inp = inp_comb
        elif type(model) == ResNetPolicy:
            inp = inp_rgb
        else:
            inp = inp_state

        #inp = s['state'][None]

        
        model_action = model(inp[0],inp[1])
        
        #model_action = model.forward(inp)

        model_action = model_action[0].cpu().detach().numpy()
        policy_action, _ = policy.predict(s['state'], deterministic=True)

        action = model_action
        
        
        
        #print(counter,np.linalg.norm(model_action-policy_action))
        #action = policy_action

        if use_policy:
            action = policy_action

        #action += noise_factor*(np.random.rand((len(policy_action))) - 0.5)
        s, r, done, info = env.step(action)

        render(writer, env.wrapped.physics)
        total_reward += r
        buffer.append({'images':images, 'policy_action':policy_action,  'task':t, 'rl_state':s, 'model_action':model_action
        })

    success_rate = info['obj_success']

    print('pregrasp',pregrasp)
    print('Total reward:', total_reward)
    print('Success rate:', success_rate)
    print('Episode len',counter)
    return buffer,success_rate
    


def rollout_manager(writer, ep_pregrasps, path, noise_factor=0.0,use_policy=False):
    if MODEL_TYPE == RGB:
        model = ResNetPolicy()
    elif MODEL_TYPE == STATE:
        model = ParamFC()
    else:
        model = CombinedNet()
    model.load_state_dict(torch.load(WEIGHTS_PATH)['model'])
    

    model = model.to('cuda')
    model.eval()
    episodes = len(ep_pregrasps)
    

    buffer = []

    avg_sr = 0
    for i in range(episodes):
        print('episode',i)
        

        task,pregrasp = ep_pregrasps[i]['task'], ep_pregrasps[i]['pregrasp']

        policy = PPO.load(f'/home/ishaans/TCDM_dev/unified_saved_weights/general_10_grasps/{task}/restore_checkpoint.zip')
        save_folder = f'pretrained_agents/{task}'
        ep_buffer,success_rate = rollout_single_episode(save_folder, writer, pregrasp, policy, model, noise_factor=noise_factor, use_policy=use_policy)
        avg_sr += success_rate
        print('ep buffer size', len(ep_buffer))
        buffer += ep_buffer
        #buffer.append(noisy_rollout(save_folder,writer,policy,pregrasp,task,noise_factor=noise_factor))
    
    
    np.save(path,buffer)
    return avg_sr/episodes



def dagger(writer,tasks):
    paths = []
    path_pref = '/home/ishaans/TCDM_dev/dagger_simple/buffers/'

    for i in range(0,10):
        ep_pregrasps = get_pregrasps(tasks,episodes=NUM_EPS)
        save_path = os.path.join(path_pref,f'{str(tasks[0])}_exp_buffer_dagger_iter_simple_{i}_{MODEL_TYPE}.npy')
        paths.append(save_path)
        sr = rollout_manager(writer, ep_pregrasps, save_path, noise_factor=NOISE_FACTOR,use_policy=True)
    
    for i in range(10,50):
        ep_pregrasps = get_pregrasps(tasks,episodes=NUM_EPS)
        save_path = os.path.join(path_pref,f'{str(tasks[0])}_exp_buffer_dagger_iter_simple_{i}_{MODEL_TYPE}.npy')
        paths.append(save_path)
        sr = rollout_manager(writer, ep_pregrasps, save_path, noise_factor=NOISE_FACTOR)
        train_policy_2d(paths,WEIGHTS_PATH,MODEL_TYPE)
        





def get_pregrasps(tasks,episodes=100):
    tasks = tasks

    ep_pregrasps = []

    for i in range(episodes):
        task_idx = np.random.randint(len(tasks)) #i//10 
        task = tasks[task_idx]
        
        
        gen10grasps = grasp_helper(task,gen10=True)

        gen10grasps = sorted(gen10grasps)
        
        pregrasp =  gen10grasps[np.random.randint(len(gen10grasps))]

        ep_pregrasps.append({'task':task,'pregrasp':pregrasp})
    return ep_pregrasps
    



if __name__ == "__main__":
    

    # configure writer
    print(WEIGHTS_PATH)
    
    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(',')]
    
    if args.render:
        writer = imageio.get_writer('rollout_mouse_lift.mp4', fps=25)
    else:
        writer = None
    
    #evaluate_model(writer)
    dagger(writer,tasks)
    #gen_rollouts(writer)
    writer.close()
   
