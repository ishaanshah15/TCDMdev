import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torchvision
import random 
import sys

class CombinedNet(nn.Module):
    def __init__(self, num_actions=30) -> None:
        super().__init__()
        self.resnet_fc = ResNetPolicy(num_actions=128)
        self.param_fc = ParamFC(inp_size=21,out_size=128)
        self.fc_last = nn.Linear(256,num_actions)
    
    def forward(self,img,goal):
        resnet_out = self.resnet_fc(img,None)
        param_out = self.param_fc(None,goal)
        out = torch.cat((resnet_out,param_out),dim=1)
        out = self.fc_last(out)
        return nn.Tanh()(out)




class ResNetPolicy(nn.Module):
    def __init__(self, num_actions=30) -> None:
        super().__init__()
        
       
        self.resnet = torchvision.models.resnet18()
        
        
        # TODO define a FC layer here to process the features
        
        
        self.resnet.fc = nn.Linear(512,128)
        
        self.fc_last = nn.Linear(4*128,128)

        self.fc_last2 = nn.Linear(128,num_actions)
     
        

    def forward(self,x,state=None):
        # TODO return unnormalized log-probabilities here

        #Split batch across channel dimensions into evenly sized chunks of 3

        xtups = torch.split(x,3,dim=1)
        outs = []
        for i in range(len(xtups)):
            outs.append(self.resnet.forward(xtups[i]))
        out = torch.cat(outs,dim=1)
        out = nn.ReLU()(self.fc_last(out))
        out = self.fc_last2(out)
        return nn.Tanh()(out)
    

class ParamFC(nn.Module):

    def __init__(self,inp_size=322,out_size=30):
        super().__init__()
        fc_dims = 2048
        self.fc1 = nn.Linear(inp_size,fc_dims*4)
        self.relu1 = nn.ReLU()
        self.fc_mid = nn.Linear(fc_dims*4,fc_dims//2)
        self.relu_mid = nn.ReLU()
        self.fc_mid2 = nn.Linear(fc_dims//2,256)
        self.relu_mid2 = nn.ReLU()
        #self.fc_mid3 = nn.Linear(fc_dims*4,256)
        #self.relu_mid3 = nn.ReLU()
        self.fc2 = nn.Linear(256, out_size)
        self.relu2 = nn.ReLU()
        self.out_size = out_size
        
    
    def forward(self,img,x):
        x = self.relu1(self.fc1(x))
        x = self.relu_mid(self.fc_mid(x))
        x = self.relu_mid2(self.fc_mid2(x))
        #x = self.relu_mid3(self.fc_mid3(x))
        return nn.Tanh()(self.fc2(x))