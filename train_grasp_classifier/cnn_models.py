import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torchvision
import torch.nn as nn
import random 



class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        
        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        
        """
        for name,parameter in self.resnet.named_parameters():
            parameter.requires_grad = False
            #pass
        """
        
        # TODO define a FC layer here to process the features
        
        
        self.resnet.fc = nn.Linear(512,num_classes)
     
        

    def forward(self,x,params=None):
        # TODO return unnormalized log-probabilities here
        return self.resnet.forward(x)
    
        


def get_fc(inp_dim, out_dim, non_linear='relu'):
    """
    Mid-level API. It is useful to customize your own for large code repo.
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers


class SimpleCNN(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=10, inp_size=28, c_dim=1, out_size=256, multimodal=False):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(c_dim, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.nonlinear = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

        # TODO set the correct dim here
        self.flat_dim = 16384

        # Sequential is another way of chaining the layers.
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, out_size, 'none'))
        if not multimodal:
            self.fc2 = nn.Sequential(*get_fc(out_size, num_classes, 'none'))
        self.multimodal = multimodal
        self.out_size = out_size

    def forward(self, x, params):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification logits in shape of (N, Nc)
        """

        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        flat_x = x.view(N, self.flat_dim)
        out = self.fc1(flat_x)

        if self.multimodal:
            return out
        out = self.fc2(out)
        return out

class ParamFC(nn.Module):

    def __init__(self,inp_size=48,out_size=256):
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
        
    
    def forward(self,im,x):
        x = self.relu1(self.fc1(x))
        x = self.relu_mid(self.fc_mid(x))
        x = self.relu_mid2(self.fc_mid2(x))
        #x = self.relu_mid3(self.fc_mid3(x))
        return self.fc2(x)



class MultiModalCNN(nn.Module):

    def __init__(self,num_classes=1,inp_size=64):
        super().__init__()
        self.cnn_backbone = ResNet(num_classes=256)#SimpleCNN(num_classes=1, inp_size=inp_size, c_dim=3, out_size=256, multimodal=True)
        self.param_backbone = ParamFC(out_size=256)
        self.fc = nn.Linear(512,num_classes)

    
    def forward(self,im,params):
        out_im = self.cnn_backbone(im,params)
        out_param = self.param_backbone(im,params)
        fc_inp = torch.cat([out_im,out_param],dim=1)
        return self.fc(fc_inp)