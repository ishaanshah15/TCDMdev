from __future__ import print_function
import cv2 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import BCEWithLogitsLoss
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
#from pytorch_grad_cam import GradCAM
import pandas as pd
import os
import random
import shutil
import sys
sys.path.append('/home/ishaans/TCDM_dev/dagger_simple')
from action_dataset import get_rgb_dataloader
from policy_net import ResNetPolicy,ParamFC,CombinedNet
import utils
STATE = 0
COMBINED = 1
RGB = 2


def train(args, model, optimizer, train_loader, scheduler=None, model_name='model',save_path=None):
    writer = SummaryWriter()
    best = 0

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        total_loss = 0
        for batch_idx, batch_dict in enumerate(train_loader):
            data,target = batch_dict['images'],batch_dict['action']
            goal,state = batch_dict['goal'],batch_dict['state']

            
            data, target = data.to(args.device), target.to(args.device)
            goal,state  = goal.to(args.device), state.to(args.device)
            data = data.type(torch.float32)

           
            #print('target batch',np.mean((target == 1).numpy()))

            
            optimizer.zero_grad()

            if type(model) == CombinedNet:
                output = model(data,goal)
            elif type(model) == ResNetPolicy:
                output = model(data,None)
            else:
                output = model(None,state)

            loss = torch.nn.MSELoss()(output,target)
            loss.backward()

            total_loss += loss.item()
           
            
            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                

                

            optimizer.step()
            
            # Validation iteration
           
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

    # Validation iteration

        if epoch % 50 == 0:
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()}, save_path)
        

        print(f'avg loss {total_loss/len(train_loader)}')

    #model.load_state_dict(torch.load('best_model_2-5'))
    print('save_path',save_path)

    
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()}, save_path)
    

    
    
def train_policy_2d(dataset_path,weights_path=None,model_type=0):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)


    # create hyperparameter argument class
    # Use image size of 64x64 in Q1. We will use a default size of 224x224 for the rest of the questions.

    # TODO experiment a little and choose the correct hyperparameters
    # You should get a map of around 22 in 5 epochs
    args = utils.ARGS(
         epochs=50,
         inp_size=64,
         use_cuda=True,
         val_every=70,
         lr=2e-4,
         batch_size=64,
         step_size=100,
         gamma=0.75
    )

    print(args)
   
    #train_loader = get_dataloader(dataset_path)
    #train_loader = get_state_dataloader(dataset_path)
    train_loader = get_rgb_dataloader(dataset_path)
    if model_type == RGB:
        model = ResNetPolicy()
    elif model_type == STATE:
        model = ParamFC()
    else:
        model = CombinedNet()

    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print("Num Epochs", args.epochs)
    if weights_path != None:
        if 'model' in torch.load(weights_path):
            model.load_state_dict(torch.load(weights_path)['model'])
            optimizer.load_state_dict(torch.load(weights_path)['optimizer'])
        else:
            model.load_state_dict(torch.load(weights_path))
    

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map

        

    
    train(args, model, optimizer, train_loader,  scheduler, save_path=weights_path)





if __name__ == "__main__":
   
    #train_loader = get_dataloader(dataset_path)
    dataset_path = '/home/ishaans/TCDM_dev/distill_policy/buffers/exp_buffer_3d_dagger_fixed_policy_1.npy'

    train_policy_2d([dataset_path])
    
