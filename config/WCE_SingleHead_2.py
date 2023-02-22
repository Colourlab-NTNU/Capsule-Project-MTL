
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:31:57 2020

@author: user1
"""

batch_size = 64

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
data_val_opt['batch_size'] = batch_size
data_val_opt['unsupervised'] = True
data_val_opt['epoch_size'] = None
data_val_opt['random_sized_crop'] = False
data_val_opt['dataset_name'] = 'wce'
data_val_opt['add_data'] = ''
data_val_opt['split'] = 'val'
data_val_opt['distortion_type'] = 'None'

config['data_train_opt'] = data_train_opt
config['data_val_opt']  = data_val_opt
config['max_num_epochs'] = 800


# number of classification heads , # number of categories per head, in order
net_opt = {}
net_opt['num_heads'] = 2
net_opt['num_classes'] = [3, 4]
net_opt['pathologies'] = {'inflammatory': 0, 'normal': 1, 'vascularlesion': 2}
net_opt['distortion'] = {'None': 0, 'H ': 1, 'V ': 2, 'D' : 3}
net_opt['num_stages']  = 4
out_feat_keys = ['conv1', 'conv2','conv3', 'conv4', 'conv5', 'fc_block']
net_opt['out_feat_keys'] = out_feat_keys

networks = {}
# pretrained = './experiments/ImageNet_RotNet_AlexNet/model_net_epoch50'

net_optim_params = {'optim_type': 'sgd', 'lr': 0.001, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(15, 0.01),(30, 0.001),(45, 0.0001),(50, 0.00001)]}
networks['pretrained'] = None
networks['def_file'] = 'architectures/OneHeadAlexNet.py'
networks['opt'] = net_opt 
networks['optim_params'] = net_optim_params
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None, 'weighted':False, 'mtl': False}
config['criterions'] = criterions
config['algorithm_type'] = 'ClassificationModel'
config['tbX_run'] = 'run_review_aug_'
# out_feat_keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']




