"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd

import datetime
import logging
import provider
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from prepare_data_3d import get_dataset3d,get_dataset_hal
from models import pointnet_cls,pointnet2_cls_ssg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
THRESHOLD = 0.7

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=2, type=int, choices=[2,10,40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=10, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=8000, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='runs', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def per_task_accuracy():
    
    fn_df = pd.read_csv('analysis/false_negatives/scores.csv')
    fp_df = pd.read_csv('analysis/false_positives/scores.csv')
    tp_df = pd.read_csv('analysis/true_positives/scores.csv')
    tn_df = pd.read_csv('analysis/true_negatives/scores.csv')

    
    tasks = set(fn_df.task) 
    tasks.update(set(fp_df.task))
    tasks.update(set(tp_df.task))
    tasks.update(set(tn_df.task))


    stats_dict = []


    for t in tasks:
        print(t)
        
        fn_count = float(len(fn_df[fn_df.task == t]))
        fp_count = float(len(fp_df[fp_df.task == t]))
        tp_count = float(len(tp_df[tp_df.task == t]))
        tn_count = float(len(tn_df[tn_df.task == t]))

        print(f'{t} total pos count: {fn_count + tp_count} total neg count: {fp_count + tn_count}')

        try:
            recall = tp_count/(fn_count + tp_count)
            precision = tp_count/(fp_count + tp_count)
        except:
            recall = -1
            precision = -1

        print(f'{t} False Neg:{fn_count} False Pos: {fp_count}')
        print(f'{t} True Neg:{tn_count} True Pos: {tp_count}')
        
        print(f'precision {t} {precision}')
        print(f'recall {t} {recall}')
        

        stats_dict.append({'task':t,'recall': recall, 'precision':precision,
                           'gt positives':fn_count + tp_count, 'gt negative': fp_count + tn_count,
                           'true positives':tp_count, 'true negatives':tn_count, 'false positives':fp_count, 
                           'false negatives': fn_count})
    

    for t in tasks:
        save_dir = os.path.join('analysis','tasks',t)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

       

        fn_datapath = list(fn_df[fn_df.task == t]['datapath'])
        fp_datapath = list(fp_df[fp_df.task == t]['datapath'])
        tp_datapath = list(tp_df[tp_df.task == t]['datapath'])
        tn_datapath = list(tn_df[tn_df.task == t]['datapath'])


        all_datapaths = fn_datapath + fp_datapath + tp_datapath + tn_datapath
        labels = [1]*len(fn_datapath) + [0]*len(fp_datapath) + [1]*len(tp_datapath) + [0]*len(tn_datapath)
        preds = [0]*len(fn_datapath) + [1]*len(fp_datapath) + [1]*len(tp_datapath) + [0]*len(tn_datapath)
        outputs = list(fn_df['outputs']) + list(fp_df['outputs']) + list(tp_df['outputs']) + list(tn_df['outputs'])

        labels = np.array(labels)
        preds = np.array(preds)

        for i in range(len(all_datapaths)):
            dp = all_datapaths[i]
            dp_label = labels[i]
            dp_pred = preds[i]
            out = outputs[i]
            root_path = '/'.join(dp.split('/')[:-1])
            copy_path = '/'.join(dp.split('/')[5:-1])
            import ipdb
            #ipdb.set_trace()

            if not os.path.exists(os.path.join(save_dir,copy_path)):
                os.makedirs(os.path.join(save_dir,copy_path))

            predir = os.path.join(save_dir,copy_path)
            
            if dp_label:
                with open(os.path.join(predir,'passed_test.txt'), 'w') as f:
                    f.write(dp)
            else:
                 with open(os.path.join(predir,'failed_test.txt'), 'w') as f:
                    f.write(dp)

            
            if dp_pred:
                with open(os.path.join(predir,'pred_true.txt'), 'w') as f:
                    f.write(str(out))
            else:
                with open(os.path.join(predir,'pred_false.txt'), 'w') as f:
                    f.write(str(out))


            pg_path = os.path.join(root_path,'pregrasp.png')

            try:
                shutil.copyfile(pg_path,os.path.join(save_dir,copy_path,'pregrasp.png'))
            except:
                print('failed copy')
        
    
    stats_df = pd.DataFrame.from_dict(stats_dict)
    stats_df.to_csv(os.path.join('analysis','task_stats.csv'))


def test_accuracy(loader,classifier,device,epoch,save=False):
    tasks = set()
    save_dir = 'analysis'
    folders = ['true_negatives','true_positives','false_negatives','false_positives']
    info_keys = ['outputs','task','datapath']
    false_negatives = {k:[] for k in info_keys}
    true_negatives = {k:[] for k in info_keys}
    false_positives = {k:[] for k in info_keys}
    true_positives = {k:[] for k in info_keys}
    flist = [true_negatives,true_positives,false_negatives,false_positives]

    total = 0
    for j, data_dict in tqdm(enumerate(loader), total=len(loader)):
        points = data_dict['points']
        target = data_dict['labels']
        task = data_dict['tasks']
        finger_params = data_dict['finger_params']
        datapath = data_dict['datapath']
        
        
        tasks.update(set(task))
       
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
            finger_params = finger_params.cuda()

        

        points = points.transpose(2, 1)
        pred, _ = classifier(points,finger_params)
        preds = pred.data.max(1)[1].cpu().numpy()
        preds = ((F.softmax(pred.data)[:,1] > THRESHOLD).type(torch.int)).cpu().numpy() #TODO: Use 0.7 as threshold

        gt = target.cpu().numpy()
        
        probs_sm = F.softmax(pred.data)
        outputs = probs_sm.max(1)[0].detach().cpu().numpy()
        total += np.sum(preds == gt)

        true_pos_indices = (gt == 1)*(preds == 1)
        true_neg_indices = (gt == 0)*(preds == 0)
        false_pos_indices = (gt == 0)*(preds == 1)
        false_neg_indices  = (gt == 1)*(preds == 0)

        info_data = [outputs,np.array(task),np.array(datapath)]

        
        for i in range(len(info_keys)):

            
            try:
                true_positives[info_keys[i]].append(info_data[i][true_pos_indices])
                true_negatives[info_keys[i]].append(info_data[i][true_neg_indices])
                false_negatives[info_keys[i]].append(info_data[i][false_neg_indices])
                false_positives[info_keys[i]].append(info_data[i][false_pos_indices])
            except:
                import ipdb
                ipdb.set_trace()
            
            


    #samples = data[:10].permute(0,2,3,1).cpu().numpy()
    for fli in flist:
        for k in info_keys:
            fli[k] = np.concatenate(fli[k])

    
    save_dir = 'analysis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for f in folders:
        if not os.path.exists(os.path.join(save_dir,f)):
            os.makedirs(os.path.join(save_dir,f))

    if save:
        print('Analyzing ...')
        for i in range(len(flist)):
            fname = folders[i]
            fdata= flist[i]
            
            #df = {'score':fdata['outputs'],'paths':fdata['paths']}
            
            df = pd.DataFrame.from_dict(fdata)

            
            df.to_csv(os.path.join(save_dir,fname,'scores.csv'))

    
    return total/len(loader.dataset)


def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()


    for j, data_dict in tqdm(enumerate(loader), total=len(loader)):
        points = data_dict['points']
        target = data_dict['labels']
        task = data_dict['tasks']
        finger_params = data_dict['finger_params']

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
            finger_params = finger_params.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points,finger_params)
        pred_choice = pred.data.max(1)[1]

        
        pred_choice = (F.softmax(pred.data)[:,1] > THRESHOLD).type(torch.int) #TODO: Use 0.7 as threshold

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    train_dataset,val_dataset, test_dataset = get_dataset_hal()
    train_sz = int(len(train_dataset))
    test_sz = int(len(test_dataset))
    val_sz = int(len(val_dataset))
    print(f'Train size resampled: {train_sz}, Val size resampled: {val_sz}, Test size resampled: {test_sz}')
    #train_dataset,_ = torch.utils.data.random_split(train_dataset, [train_sz, len(train_dataset) - train_sz])
    #test_dataset, _ = torch.utils.data.random_split(test_dataset, [test_sz, len(test_dataset) - test_sz])

    import ipdb
    #ipdb.set_trace()
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    valDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    classifier = model.MultiModal3D(num_class, normal_channel=args.use_normals)

    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        total_loss = 0

        
        for batch_id, data_dict in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points = data_dict['points']
            target = data_dict['labels']
            task = data_dict['tasks']
            finger_params = data_dict['finger_params']
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)

            
            
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
                finger_params = finger_params.cuda()

            pred, trans_feat = classifier(points,finger_params)

            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            pred_choice = (F.softmax(pred.data)[:,1] > THRESHOLD).type(torch.int) #TODO: Use 0.7 as threshold
            total_loss += loss

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            
            loss.backward()
            optimizer.step()
            global_step += 1
        
        
        
        print(f'Training Loss: {total_loss.item()/len(trainDataLoader)}')

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            test_instance_acc, test_class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
            val_instance_acc, val_class_acc = test(classifier.eval(), valDataLoader, num_class=num_class)
            train_instance_acc, train_class_acc = test(classifier.eval(), trainDataLoader, num_class=num_class)
            

            #test(classifier.eval(), trainDataLoader, num_class=num_class)

            if (val_instance_acc >= best_instance_acc):
                best_instance_acc = val_instance_acc
                best_epoch = epoch + 1

            if (val_class_acc >= best_class_acc):
                best_class_acc = val_class_acc
            #log_string('Train Instance Accuracy: %f, Class Accuracy: %f' % (tr_instance_acc, tr_class_acc))
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (test_instance_acc, test_class_acc))
            log_string('Val Instance Accuracy: %f, Class Accuracy: %f' % (val_instance_acc, val_class_acc))
            log_string('Train Instance Accuracy: %f, Class Accuracy: %f' % (train_instance_acc, train_class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))


            if (val_instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': best_instance_acc,
                    'class_acc': best_class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1


    savepath = str(checkpoints_dir) + '/best_model.pth'
    model = classifier
    model.load_state_dict(torch.load(savepath)['model_state_dict'])
    model.eval()


    instance_acc, class_acc = test(model, testDataLoader, num_class=num_class)
    log_string('Final Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

    test_accuracy(testDataLoader,model,'cuda',0,save=True)
    per_task_accuracy()

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
