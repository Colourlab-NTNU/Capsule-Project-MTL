#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:24:31 2020

@author: user1
"""


from __future__ import print_function
import os, tqdm, copy, math, sys
import os.path
from time import time


import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
from torchvision.utils import make_grid
from torch.nn import ModuleList, Parameter
from sklearn.metrics import confusion_matrix

import logging 
import datetime

from visuals import visualize_feature_maps, add_embed, write_conv
from losses import UncertaintyLoss

def load_checkpoint(net, ckp_path):
    files = [f for f in os.listdir(ckp_path) if 'pth' in f]
    if len(files)>0:
        ckp = sorted(files)[-1]
        try:
            net.load_state_dict(torch.load(ckp_path+'/'+ckp))
        except: 
            dictstate = torch.load(ckp_path+'/'+ckp)
            # new_dict = copy.deepcopy(dictstate)
            # for key in dictstate.keys():
            #     if key[:7] == 'module.':
            #         new_key = key[7:]
            #         new_dict[new_key] = dictstate.pop(key)
            new_dict = load_unparallel(dictstate)
        
            net.load_state_dict(new_dict)
        iter_start = int(ckp.split(".")[-3].split("_")[-1])
        print('Starting from checkpoint \t: ',ckp)
    return net, iter_start

def load_unparallel(state_dict):
    # check if the keys are already compatible with data parallel, i.e, have prefix 'module'
    unparallel_dict = copy.deepcopy(state_dict)
    for key in state_dict.keys():
        if key[:7] == 'module.':
            new_key = key[7:]
            unparallel_dict[new_key] = unparallel_dict.pop(key)
        else:
            print('already un-parallel')
            break
                
    return unparallel_dict
        
def load_to_gpu(net, visualize):
    if not visualize:
        net = nn.DataParallel(net)
    else: pass
    return net.cuda()  

# def init_net():      
    
class Train():
    def __init__(self,  config, ckp_directory, gpu, inv_tfm, distortion_type):
        
        self.config = config
        self.set_experiment_dir(config['exp_dir'])
        self.distortion_type = distortion_type
        self.ckp_dir = ckp_directory
        self.gpu = gpu
        self.writer = SummaryWriter('./%s/%s'%(config['exp_dir'],config['tbX_run']))
        self.writer.add_text('args', " \n".join(['%s' % (config[key] for key in config.keys())]))
        self.inv_tfm = inv_tfm
        
        # net configurations : 
        # self.algo_type = config['algorithm_type'] 
        self.batch_size = config['data_train_opt']['batch_size']
        self.dataset_name = config['data_train_opt']['dataset_name']
        # self.train_step(net, train_loader, val_loader, start_iter)
        self.head1_cat = list(config['networks']['opt']['pathologies'].keys())
        # self.head2_cat = list(config['networks']['opt']['distortion'][distortion_type].keys())
        
        
    def set_experiment_dir(self,directory_path):
        
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir)

        self.vis_dir = os.path.join(directory_path,'visuals')
        if (not os.path.isdir(self.vis_dir)):
            os.makedirs(self.vis_dir)

        self.preds_dir = os.path.join(directory_path,'preds')
        if (not os.path.isdir(self.preds_dir)):
            os.makedirs(self.preds_dir)
            
            
    def set_log_file_handler(self):
        
        self.logger = logging.getLogger(__name__)
    
        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO)
    
        log_dir = os.path.join(self.config['exp_dir'], 'logs')
        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)
        now_str = datetime.datetime.now().__str__().replace(' ','_')
        self.log_file = os.path.join(log_dir, 'LOG_INFO_'+now_str+'.txt')
        self.log_fileHandler = logging.FileHandler(self.log_file)
        self.log_fileHandler.setFormatter(formatter)
        self.logger.addHandler(self.log_fileHandler)
        

    def init_criterion(self, ctype, copt, mtl_u=None):
    # self.logger.info('Initialize criterion[%s]: %s with options: %s' % (key, crit_type, crit_opt))
        if not mtl_u:
            return getattr(nn, ctype)(copt)
        else:
            print(' **************  The chosen Loss is - UNCERTAINITY LOSS  **************** ')
            criterion = UncertaintyLoss(ctype, copt)
            return criterion
            
        

    def init_optimizer(self, net, criterion):
        self.optim_opts = self.config['networks']['optim_params']
        optim_type = self.optim_opts['optim_type']
        learning_rate = self.optim_opts['lr']
        optimizer = None
        parameters = [{'params': filter(lambda p: p.requires_grad, net.parameters())},
                      {'params': criterion.parameters()}]
        # self.logger.info('Initialize optimizer: %s with params: %s for netwotk: %s'
            # % (optim_type, self.optim_opts, key))
        if optim_type == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
            
        elif optim_type == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=learning_rate,
                momentum=self.optim_opts['momentum'],
                nesterov=self.optim_opts['nesterov'] if ('nesterov' in self.optim_opts) else False,
                weight_decay=self.optim_opts['weight_decay'])
        else:
            raise ValueError('Not supported or recognized optim_type', optim_type)
        print(('Optimizer params : lr %f,  optim type  %s '%(learning_rate, optim_type)))
        return optimizer
    

    def adjust_learning_rate(self, optimizer, epoch, init_lr=0.1, step=100, decay=0.1): # step chnaged from 30 to 100
    
        if self.optim_opts['LUT_lr'] is None:
        # the lr has become v low, so restoring it temporarily after epoch 1000 by 
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = init_lr * (decay ** (epoch // step))
            print('Learning Rate %f'%lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            LUT = self.optim_opts['LUT_lr']
            lr = next((lr for (max_epoch, lr) in LUT if max_epoch>epoch), LUT[-1][1])
            # self.logger.info('==> Set to %s optimizer lr = %.10f' % (key, lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        return lr
    
    
    def get_loss(self, gt_pathology, gt_distortion, net_output, criterion, mtl_u): 
        if not mtl_u:
            pathology_loss = criterion(net_output['pathology'], gt_pathology)
            d1_loss = criterion(net_output['distortion_1'], gt_distortion)
            a, b = self.config['criterions']['loss']['a'],self.config['criterions']['loss']['b'] # TODO mtl
            # loss = a*pathology_loss + b*d1_loss 
            loss = a*pathology_loss    # TODO for single head only
            return loss, pathology_loss,  d1_loss
        else:
            loss, pathology_loss,  d1_loss, log_vars = criterion(net_output, gt_pathology, gt_distortion)  
            return loss, pathology_loss,  d1_loss, log_vars
    
    
    def compute_accuracy(self, output, target, topk=(1,),fscore=False):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        prob, pred = output.topk(maxk, 1, True, True)  
        pred = pred.t() 
        cm = confusion_matrix(pred.squeeze(0), target)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        if not fscore:
            return res, pred
        else:       
           
            p,r,f,s = precision_recall_fscore_support(pred.squeeze(0), target, average=None, labels=np.unique(pred.squeeze(0))) # return fscore also during val
            print('precision {} , recall {}, fscore {}'.format(p,r,f))
            conf_mat = np.array(cm)
            conf_df = pd.DataFrame(conf_mat)
            conf_df.columns = np.arange(0,3)
            print(conf_df)
            conf_df.to_csv('confusion_matrix_'+ '.csv')
            return res,p,r,f
    

 
    def evaluate(self, net, val_loader, criterion, steps, epoch, add_embedd):
        for i, val_dict in enumerate(val_loader(epoch)): #images = torch.Size([128, 9, 3, 67, 67]), labels =torch.Size([256]) the permustation)
            images, ground_truth = val_dict['imgs']['imgs'], val_dict['labels']
            images = Variable(images)
            gt_pathology = Variable(ground_truth['pathology'])
            gt_distortion = Variable(ground_truth['distortion'])
            if self.gpu:
                images = images.cuda() 
                gt_pathology = gt_pathology.cuda()
                gt_distortion = gt_distortion.cuda()
                
            vfeats, out_dict = net(images, out_feat_keys=self.config['networks']['opt']['out_feat_keys'])
            # patho_prec1, predp = self.compute_accuracy(out_dict['pathology'].cpu().data, ground_truth['pathology'].cpu().data, topk=(1,), fscore=False)
            patho_prec1, precision, recall, f_score = self.compute_accuracy(out_dict['pathology'].cpu().data, ground_truth['pathology'].cpu().data, topk=(1,), fscore=True)
            print(f_score)
            # distort_prec1 , _ = self.compute_accuracy(out_dict['distortion_1'].cpu().data, ground_truth['distortion'].cpu().data)
            patho_Vacc = patho_prec1[0].item()
            # distort_Vacc = distort_prec1[0].item()
            # loss, p_loss, d_loss = self.get_loss(gt_pathology,gt_distortion, out_dict, criterion)
            # val_path_label, val_dist_label = 'Pathology_'+str(self.distortion_type),  'Distortion'+str(self.distortion_type)
            # self.writer.add_scalars('Val Accuracy', {val_path_label: np.mean(patho_Vacc), val_dist_label: np.mean(distort_Vacc)}, steps)
            self.writer.add_scalars('Val F1 Score', {'F_score': np.mean(f_score)}, steps)
            print('############ Evaluating network on validation set: ############')
            print('[%2d/%2d] \t  '%(epoch+1, self.config['max_num_epochs']))
            print('Pathology :\t  Accuracy % 2.2f%%\n '%(np.mean(patho_Vacc)))
            print('precision[I, N, V] : {}, recall[I, N, V] : {}, f_score[I, N, V] :  {}'.format(precision, recall, f_score))
            # VISUALIZE Embedding :  
            if add_embedd:
                print( 'ADDING EMBEDDING FOR VAL DATA' )
                # print(predp.squeeze(0)[0])
                # print(ground_truth['pathology'])
                # meta = [[self.head1_cat[ground_truth['pathology'][i]],self.head2_cat[ground_truth['distortion'][i]]] for i in range(len(gt_pathology))] # Multi Head
                # meta = [[self.head1_cat[ground_truth['pathology'][i]],self.head1_cat[predp.squeeze(0)[i]]] for i in range(len(gt_pathology))] # Multi Head
                # meta = [[self.head1_cat[ground_truth['pathology'][i]]] for i in range(len(gt_pathology))]  # Single Head
                # embed images with sprite
                # add_embed(feats=vfeats, label_imgs=images.cpu().data, meta=meta, tfm=self.inv_tfm, step=steps, writer=self.writer, tag=self.distortion_type)
                # embed images without sprite
                # add_embed(feats=feats, meta=meta, tfm=self.inv_tfm, step=steps, writer=self.writer, tag=self.distortion_type)
                print('********************Embedding for VALIDATION DONE***********************')
                sys.exit()
    
    
    def train_step(self, net, train_loader, val_loader, start_iter, add_embedd):
        
        batch_time, net_time = [], []
        steps, valstep = start_iter, 0
        max_epochs = self.config['max_num_epochs']
        iter_per_epoch = len(train_loader.dataset)//self.batch_size
        if add_embedd:
            max_epochs = int(start_iter/iter_per_epoch) + 1
        # loss configurations : 
        criterion_opt = self.config['criterions']['loss']
        mtl =  criterion_opt['mtl']  
        # init criterion : 
        criterion = self.init_criterion(criterion_opt['ctype'], criterion_opt['opt'], mtl) 
        # init optimizer : 
        optimizer = self.init_optimizer(net, criterion) 
        # print(iter_per_epoch)
        for epoch in range(int(start_iter/iter_per_epoch), max_epochs):
            # self.logger.info('Training epoch [%3d / %3d]' % (self.curr_epoch+1, self.max_num_epochs))
            if (epoch%10==0 and epoch>0) or add_embedd == True:
                valstep += 1
                self.evaluate(net, val_loader, criterion, steps, epoch, add_embedd)
                
            lr = self.adjust_learning_rate(optimizer, epoch, init_lr=0.1, step=200, decay=0.1)
            end = time()
            for i, data_dict in enumerate(train_loader(epoch)): #images = torch.Size([128, 9, 3, 67, 67]), labels =torch.Size([256]) the permustation)
                images, ground_truth = data_dict['imgs']['imgs'], data_dict['labels']
                batch_time.append(time()-end)
                if len(batch_time)>100:
                    del batch_time[0]
    
                images = Variable(images)
                gt_pathology = Variable(ground_truth['pathology'])
                gt_distortion = Variable(ground_truth['distortion'])
                if self.gpu:
                    images = images.cuda() 
                    gt_pathology = gt_pathology.cuda()
                    gt_distortion = gt_distortion.cuda()
                    
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                t = time()
                feats, out_dict = net(images, out_feat_keys=self.config['networks']['opt']['out_feat_keys'])
                net_time.append(time()-t)
                if len(net_time)>100:
                    del net_time[0]
                patho_prec1, predp  = self.compute_accuracy(out_dict['pathology'].cpu().data, ground_truth['pathology'].cpu().data, topk=(1,))
                distort_prec1, _  = self.compute_accuracy(out_dict['distortion_1'].cpu().data, ground_truth['distortion'].cpu().data) #TODO
                patho_acc = patho_prec1[0].item()
                # distort_acc = distort_prec1[0].item()  #TODO mtl
                # loss, pathology_loss, dist1_loss, log_vars = self.get_loss(gt_pathology,gt_distortion, out_dict, criterion, mtl)    # Multi Head  #TODO
                loss, pathology_loss, _ = self.get_loss(gt_pathology,gt_distortion, out_dict, criterion, mtl)  # None
                loss.backward()
                optimizer.step()
                loss = float(loss.cpu().data.numpy())
                
                # # VISUALIZE Embedding :  
                # if add_embedd:
                #     # print(predp.squeeze(0)[0])
                #     # print(ground_truth['pathology'])
                #     # meta = [[self.head1_cat[ground_truth['pathology'][i]],self.head2_cat[ground_truth['distortion'][i]]] for i in range(len(gt_pathology))] # Multi Head
                #     # meta = [[self.head1_cat[ground_truth['pathology'][i]],self.head1_cat[predp.squeeze(0)[i]]] for i in range(len(gt_pathology))] # Multi Head
                #     meta = [[self.head1_cat[ground_truth['pathology'][i]]] for i in range(len(gt_pathology))]  # Single Head
                #     # embed images with sprite
                #     add_embed(feats=feats, label_imgs=images.cpu().data, meta=meta, tfm=self.inv_tfm, step=steps, writer=self.writer, tag=self.distortion_type)
                #     # embed images without sprite
                #     # add_embed(feats=feats, meta=meta, tfm=self.inv_tfm, step=steps, writer=self.writer, tag=self.distortion_type)
                #     # print('********************DONE***********************')
                ########################### LOGGING #########################

                if self.writer:
                    # self.writer.add_scalar('lr', lr, steps)
                    self.writer.add_scalar('Total Loss', loss, steps)
                    self.writer.add_scalar('Pathology Loss', pathology_loss, steps)
                    # self.writer.add_scalars('Loss', {'Pathology' : pathology_loss, 'Distortion' : dist1_loss}, steps)  #TODO
                    # self.writer.add_scalars('Loss', {'Pathology' : pathology_loss}, steps)
                    self.writer.add_scalars('Accuracy', {'Pathology Acc' : patho_acc}, steps)
                    # self.writer.add_scalars('Train Accuracy', {'Pathology Acc' : patho_acc, 'Distortion Acc' : distort_acc}, steps)  #TODO
                    self.writer.add_scalars('Train Accuracy', {'Pathology Acc' : patho_acc}, steps)
                    # std1, std2 = round(math.exp(log_vars[0])** 0.5, 3) , round(math.exp(log_vars[1])** 0.5, 3) # TODO
                    
                    # self.writer.add_scalars('loss weightage', {'sigma_p' : std1, 'sigma_d' : std2}, steps) # TODO
   
                if steps%100==0:
                    # print(('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Total Loss: % 1.3f\n'\
                    #         'Pathology Loss: % 1.3f\t Distortion Loss: % 1.3f\n' \
                    #         'Pathology Accuracy % 2.2f%%\t Distortion Accuracy % 2.2f%%\n' %(
                    #             epoch+1, max_epochs, steps,
                    #             np.mean(batch_time), np.mean(net_time),
                                # lr, loss ,pathology_loss, dist1_loss, patho_acc, distort_acc))) #TODO
                    print(('[%2d/%2d] %5d) [batch load % 2.3fsec, net %1.2fsec], LR %.5f, Total Loss: % 1.3f\n'\
                            'Pathology Loss: % 1.3f\t' \
                            'Pathology Accuracy % 2.2f%%\n' %(
                                epoch+1, max_epochs, steps,
                                np.mean(batch_time), np.mean(net_time),
                                lr, loss ,pathology_loss,patho_acc))) #TODO
                steps += 1
                end = time()
                if steps%500==0:
                    filename = '%s/%s_%03i_%06d.pth.tar'%(self.ckp_dir,self.dataset_name, max_epochs,steps)
                    # save compatible for nn.DataParallel , else juse use net.save()
                    try:
                        state_dict = net.state_dict()  # saved with module prefix
                    except AttributeError:
                        state_dict = net.module.state_dict()
                    torch.save(state_dict, filename)
                    print('Saved: '+filename)

                
                  # visualize_feature_maps(feats, epoch, steps,self.config['exp_dir'], self.writer)  
            #     if self.writer:
            #         write_conv(feats, net, epoch, steps, self.writer)
            
            if os.path.exists(self.ckp_dir+'/stop.txt'):
                # break without using CTRL+C
                break
            
            
            # del images, ground_truth
        # self.writer.close()
            
            
    
            




