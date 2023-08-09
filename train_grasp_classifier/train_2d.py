import torch
import trainer_rgb
import utils
from cnn_models import SimpleCNN,ResNet,MultiModalCNN,ParamFC
import numpy as np
from prepare_data import get_dataset2d,get_dataset2d_balanced
import random
from torch.utils.data.sampler import WeightedRandomSampler

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # create hyperparameter argument class
    # Use image size of 64x64 in Q1. We will use a default size of 224x224 for the rest of the questions.

    # TODO experiment a little and choose the correct hyperparameters
    # You should get a map of around 22 in 5 epochs
    args = utils.ARGS(
         epochs=100,
         inp_size=64,
         use_cuda=True,
         val_every=70,
         lr=1e-3,
         batch_size=64,
         step_size=100,
         gamma=0.75
    )

    print(args)

    train_set,val_set,test_set = get_dataset2d_balanced(use_params=True)
    

    #train_sz = int(len(train_set)*0.2)
    #train_set,_ = torch.utils.data.random_split(train_set, [train_sz, len(train_set) - train_sz])


    

    y = train_set.labels 

    counts = np.bincount(y)
    labels_weights = 1. / counts
    weights = labels_weights[y]
    sampler = WeightedRandomSampler(weights, len(weights))

    #test_set.labels = list((np.array(test_set.labels) == 0).astype(np.int32))
    print(f'Train size resampled: {train_set.__len__()}, Test size resampled: {test_set.__len__()}')
    print('test ishaans')

    train_loader = utils.get_data_loader(train_set,train=True, batch_size=args.batch_size, sampler = sampler)
    val_loader = utils.get_data_loader(val_set,train=False, batch_size=args.batch_size)
    test_loader = utils.get_data_loader(test_set,train=False, batch_size=args.batch_size)


    #import ipdb
    #ipdb.set_trace()
    
    

    # initializes the model
    #model = SimpleCNN(num_classes=1, inp_size=64, c_dim=3, out_size=128)

    #model = ResNet(1)
    
    #model = MultiModalCNN(inp_size=64).to(args.device)
    #model = ParamFC(out_size=1)
    
    # initializes Adam optimizer and simple StepLR scheduler
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    trainer_rgb.train(args, model, optimizer, train_loader, val_loader, test_loader, scheduler)
    



