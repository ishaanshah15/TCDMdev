from __future__ import print_function
import cv2 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import BCEWithLogitsLoss
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from pytorch_grad_cam import GradCAM
import pandas as pd
import os
import shutil


SAVE_PATH = 'best_model'


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False

def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)

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

        fn_task_df = fn_df[fn_df.task == t]
        fp_task_df = fp_df[fp_df.task == t]
        tp_task_df = tp_df[tp_df.task == t]
        tn_task_df = tn_df[tn_df.task == t]



        all_datapaths = fn_datapath + fp_datapath + tp_datapath + tn_datapath
        labels = [1]*len(fn_datapath) + [0]*len(fp_datapath) + [1]*len(tp_datapath) + [0]*len(tn_datapath)
        preds = [0]*len(fn_datapath) + [1]*len(fp_datapath) + [1]*len(tp_datapath) + [0]*len(tn_datapath)
        outputs = list(fn_task_df['outputs']) + list(fp_task_df['outputs']) + list(tp_task_df['outputs']) + list(tn_task_df['outputs'])

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



def test_accuracy(test_loader,model,device,epoch,save=False):
    tasks = set()
    save_dir = 'analysis'
    folders = ['true_negatives','true_positives','false_negatives','false_positives']
    info_keys = ['paths','outputs','test1','test2','task']
    false_negatives = {k:[] for k in info_keys}
    true_negatives = {k:[] for k in info_keys}
    false_positives = {k:[] for k in info_keys}
    true_positives = {k:[] for k in info_keys}
    flist = [true_negatives,true_positives,false_negatives,false_positives]

    with torch.no_grad():

        total = 0
        for batch_idx, batch_dict in enumerate(test_loader):
            
            data,params,target = batch_dict['images'],batch_dict['params'],batch_dict['labels']
            test1,test2,task = batch_dict['test1'].numpy(),batch_dict['test2'].numpy(),np.array(batch_dict['task'])
            paths = np.array(batch_dict['paths'])
            tasks.update(set(task))
            data = data.to(device)
            params = params.to(device)
            
            output = model(data,params)
            gt = target.cpu().numpy()
            preds = np.array(output.cpu() > -1.0)[:,0]
            outputs = output.detach().cpu().numpy()[:,0]
            total += np.sum(preds == gt)

            true_pos_indices = (gt == 1)*(preds == 1)
            true_neg_indices = (gt == 0)*(preds == 0)
            false_pos_indices = (gt == 0)*(preds == 1)
            false_neg_indices  = (gt == 1)*(preds == 0)

            info_data = [paths,outputs,test1,test2,task]

            

            for i in range(len(info_keys)):
                
                true_positives[info_keys[i]].append(info_data[i][true_pos_indices])
                true_negatives[info_keys[i]].append(info_data[i][true_neg_indices])
                false_negatives[info_keys[i]].append(info_data[i][false_neg_indices])
                false_positives[info_keys[i]].append(info_data[i][false_pos_indices])
                


    #samples = data[:10].permute(0,2,3,1).cpu().numpy()
    for fli in flist:
        for k in info_keys:
            fli[k] = np.concatenate(fli[k])

    
    save_dir = 'analysis'
    for f in folders:
        if not os.path.exists(os.path.join(save_dir,f)):
            os.makedirs(os.path.join(save_dir,f))

    if save:
        print('Analyzing ...')
        for i in range(len(flist)):
            fname = folders[i]
            fdata= flist[i]
            """
            im = (samples[i] - np.min(samples[i]))/(np.max(samples[i]) - np.min(samples[i]))
            im = 255*samples[i]
            im = im.astype(np.uint8)

            
            Image.fromarray(im).save(f'analysis/{i}.png')
            

            df = {'preds':preds[:10],'gt':gt[:10],'output':output[:10,0].detach().cpu().numpy()}

            df = pd.DataFrame.from_dict(df)
            df.to_csv('analysis/results.csv')
            """

            #df = {'score':fdata['outputs'],'paths':fdata['paths']}
            

            for i in range(len(fdata['paths'])):
                p = fdata['paths'][i]
                #shutil.copyfile(p,os.path.join(save_dir,fname,str(i) + '.png'))

            fdata['datapath'] = fdata['paths']
            del fdata['paths']
            df = pd.DataFrame.from_dict(fdata)

            
            df.to_csv(os.path.join(save_dir,fname,'scores.csv'))

    
    return total/len(test_loader.dataset)


def sigmoid_fn(x):
    return 1/(1 + torch.exp(-x))



def get_gradcam(batch_dict,model,args,index=0):

    data = batch_dict['images'].to(args.device)

    target_layers = [model.resnet.layer4[-1]]
    cam = GradCAM(model=model.resnet, target_layers=target_layers, use_cuda=True)
            
    targets = [ClassifierOutputTarget(0)]

            
    grayscale_cam = cam(input_tensor=data[index:index+1], targets=targets)

    grayscale_cam = grayscale_cam[0, :]

    fpath = batch_dict['paths'][index]
    img = Image.open(fpath)


    half_img = np.array(img)[:,256:]

    img = Image.fromarray(half_img)
    img = img.resize((64,64))

    img.save('base_img2.png')
    half_img = np.array(img)



    half_img = (half_img - np.min(half_img))/(np.max(half_img) - np.min(half_img))

    

    visualization = show_cam_on_image(half_img, grayscale_cam, use_rgb=True)

    Image.fromarray(visualization).save('grad_cam_img2.png')
    



def train(args, model, optimizer, train_loader, val_loader, test_loader, scheduler=None, model_name='model'):
    writer = SummaryWriter()
    best = 0

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    
    test_acc = test_accuracy(test_loader,model,args.device,0)
    train_acc = test_accuracy(train_loader,model,args.device,0)
    print(f'test accuracy: {test_acc}')
    print(f'train accuracy: {train_acc}')

    

    

    for epoch in range(args.epochs):
        for batch_idx, batch_dict in enumerate(train_loader):
            data,params,target = batch_dict['images'],batch_dict['params'],batch_dict['labels']
            data, target = data.to(args.device), target.to(args.device)
            params = params.to(args.device)

            #print('target batch',np.mean((target == 1).numpy()))

            
            optimizer.zero_grad()
            output = model(data,params)

            
            


            #get_gradcam(batch_dict,model,args)
            

            
            
            
            

            
            # TODO implement a suitable loss function for multi-label classification
            # This function should take in network `output`, ground-truth `target`, weights `wgt` and return a single floating point number
            # You are NOT allowed to use any pytorch built-in functions
            # Remember to take care of underflows / overflows when writing your function
            
            target = target.to(torch.float64)
            
            loss = BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(args.device))(output[:,0],target)
            loss.backward()
            
            """
            norm = torch.sum(target*wgt)
            logits = output - torch.max(output,axis=1,keepdims=True).values
            log_probs= logits - torch.log(torch.sum(torch.exp(logits),axis=1,keepdims=True))
            loss = -torch.sum(log_probs*target*wgt)/norm
            """
            
            
            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                

                # Log gradients
                """
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)
                """

            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                val_acc = test_accuracy(val_loader,model,args.device,epoch)
                test_acc = test_accuracy(test_loader,model,args.device,epoch)
                
                train_acc = test_accuracy(train_loader,model,args.device,epoch)
                print(f'test accuracy: {test_acc}')
                print(f'val accuracy: {val_acc}')
                print(f'train accuracy: {train_acc}')

                if val_acc > best:
                    print('Best so far')
                    torch.save(model.state_dict(), SAVE_PATH)
                    best = val_acc

                model.train()
            
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration

    #model.load_state_dict(torch.load('best_model_2-5'))
    model.load_state_dict(torch.load('best_model'))
    model.eval()

    val_acc = test_accuracy(val_loader,model,args.device,epoch)
    test_acc = test_accuracy(test_loader,model,args.device,epoch)      
    train_acc = test_accuracy(train_loader,model,args.device,epoch)
    print(f'test final accuracy: {test_acc}')
    print(f'val final accuracy: {val_acc}')
    print(f'train final accuracy: {train_acc}')

    test_acc = test_accuracy(test_loader,model,args.device,epoch,save=True)
    
    
    #get_gradcam(batch_dict,model,args)
    per_task_accuracy()
    import ipdb
    ipdb.set_trace()
    





