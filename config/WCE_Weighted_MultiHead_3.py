
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:31:57 2020

@author: user1
"""

batch_size = 64
val_batch_size = 160
config = {}
# set the parameters related to the training and validating set
data_train_opt = {}
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = True
data_train_opt['epoch_size'] = None
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'wce'
data_train_opt['add_data'] = ''
data_train_opt['split'] = 'train'
data_train_opt['distortion_type'] = 'None'

data_val_opt = {}
data_val_opt['batch_size'] = val_batch_size
data_val_opt['unsupervised'] = True
data_val_opt['epoch_size'] = None
data_val_opt['random_sized_crop'] = False
data_val_opt['dataset_name'] = 'wce'
data_val_opt['add_data'] = ''
data_val_opt['split'] = 'val'
data_val_opt['distortion_type'] = data_train_opt['distortion_type']

config['data_train_opt'] = data_train_opt
config['data_val_opt']  = data_val_opt
config['max_num_epochs'] = 250


# number of classification heads , # number of categories per head, in order
net_opt = {}
net_opt['num_heads'] = 2
net_opt['num_classes'] = [3, 4]
# net_opt['num_classes'] = [3, 3]
net_opt['pathologies'] = {'I': 0, 'N': 1, 'V': 2}
# net_opt['distortion'] = {'Blur': 0, 'Contrast': 1, 'Saturate': 2, 'Brighten' : 3, 'All': 4}
net_opt['distortion'] = {'motion_blur_fuc': {'None':0, 'V':1, 'H':2,  'D' : 3}, 'contrast_fuc': {'None':0, 'L' : 1, 'k' :2, 'H':3}, 'Brighten_fuc': {'None':0, 'L' : 1, 'M':2, 'H':3}, 'Saturate': {'None':0, 'L' : 1, 'H':2, 'M':3}, 'None': None }  #TODO
net_opt['num_stages']  = 4
out_feat_keys = ['conv1', 'conv2','conv3', 'conv4', 'conv5', 'fc_block']
net_opt['out_feat_keys'] = out_feat_keys

networks = {}
# pretrained = './experiments/ImageNet_RotNet_AlexNet/model_net_epoch50'

net_optim_params = {'optim_type': 'adam', 'lr': 0.01, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(50, 0.01),(100, 0.001),(150, 0.001),(200, 0.0001)]}
networks['pretrained'] = None
networks['def_file'] = 'architectures/Weighted_MultiHeadAlexNet.py'
networks['opt'] = net_opt 
networks['optim_params'] = net_optim_params
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None, 'a':1, 'b':0, 'mtl': False}
config['criterions'] = criterions
# config['tbX_run'] = 'tblog_runs/hyperparam_%s/run_%s_'%('0.1',data_train_opt['distortion_type'])
config['tbX_run'] = 'tblog_runs/run_None_aug_%s_'%(data_train_opt['distortion_type'])
# out_feat_keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']




