import os
import torch
import numpy as np
import sklearn.metrics
from torch.utils.data import DataLoader
import torch.nn.functional as F


class ARGS(object):
    """
    Tracks hyper-parameters for trainer code 
        - Feel free to add your own hparams below (cannot have __ in name)
        - Constructor will automatically support overrding for non-default values
    
    Example::
        >>> args = ARGS(batch_size=23, use_cuda=True)
        >>> print(args)
        args.batch_size = 23
        args.device = cuda
        args.epochs = 14
        args.gamma = 0.7
        args.log_every = 100
        args.lr = 1.0
        args.save_model = False
        args.test_batch_size = 1000
        args.val_every = 100
    """
    # input batch size for training 
    batch_size = 64
    # input batch size for testing
    test_batch_size=1000
    # number of epochs to train for
    epochs = 14
    # learning rate
    lr = 1.0
    # Learning rate step gamma
    gamma = 0.7
    step_size = 1
    # how many batches to wait before logging training status
    log_every = 100
    # how many batches to wait before evaluating model
    val_every = 100
    # set flag to True if you wish to save the model after training
    save_at_end = False
    # set this to value >0 if you wish to save every x epochs
    save_freq=-1
    # set true if using GPU during training
    use_cuda = False
    # input size
    inp_size = 224

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert '__' not in k and hasattr(self, k), "invalid attribute!"
            assert k != 'device', "device property cannot be modified"
            setattr(self, k, v)
        
    def __repr__(self):
        repr_str = ''
        for attr in dir(self):
            if '__' not in attr and attr !='use_cuda':
                repr_str += 'args.{} = {}\n'.format(attr, getattr(self, attr))
        return repr_str
    
    @property
    def device(self):
        return torch.device("cuda" if self.use_cuda else "cpu")


def get_data_loader(dataset,train=True, batch_size=64,sampler=None):

    if sampler:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            sampler=sampler
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=4,
        )
    return loader