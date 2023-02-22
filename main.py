#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:11:45 2020
@author: user1
    """

from __future__ import print_function
import argparse
import imp

from train_utils_mtl import *
from extras.cnn_layer_visualization import CNNLayerVisualization
from MultiHeadAlexNet import MHAlexNet


from wce_dataset_simplified import WCE_DataLoader, WCE_Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--exp',         type=str,      default='WCE_Weighted_MultiHead_3',  help='config file with parameters of the experiment')
parser.add_argument('--evaluate',    default=True, action='store_true')
parser.add_argument('--checkpoint',  type=bool,      default=True,     help=' True : ckp of latest epochID , False : start new training')
parser.add_argument('--num_workers', type=int,      default=16,     help='number of data loading workers')
parser.add_argument('--gpu'  ,      type=str,     default='0,1',  help='enables cuda')
parser.add_argument('--data_parallel', default = True, help='use DataPrallelism for multiple GPUs')
parser.add_argument('--disp_step',   type=int,      default=50,    help='display step during training')
parser.add_argument('--add_embed',   type=bool,     default=True,    help='add embedding after training for 400 points')
parser.add_argument('--visualize',   type=bool,     default=False,    help='Visualize The CNN layers')
args_opt = parser.parse_args()

exp_config_file = os.path.join('.','config',args_opt.exp+'.py')
exp_directory = os.path.join('.','experiments',args_opt.exp)



    ############### SET CUDA ENV #################
print(('Using GPU :'+args_opt.gpu))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= args_opt.gpu

# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config
config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
arch_config = config['networks']['opt']
merge_seq =True  # for non pretrained moedels
print("Loading experiment %s from file: %s" % (args_opt.exp, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

# Set train and val datasets and the corresponding data loaders
data_train_opt = config['data_train_opt']
data_val_opt = config['data_val_opt']
config['disp_step'] = args_opt.disp_step
num_imgs_per_cat = data_train_opt['num_imgs_per_cat'] if ('num_imgs_per_cat' in data_train_opt) else None
ckp_directory = os.path.join('.','experiments',args_opt.exp, data_train_opt['distortion_type'], 'checkpoints')


dataset_train = WCE_Dataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'],
    random_sized_crop=data_train_opt['random_sized_crop'],
    add_data = data_train_opt['add_data'],
    num_imgs_per_cat=num_imgs_per_cat)
dataset_val = WCE_Dataset(
    dataset_name=data_val_opt['dataset_name'],
    split=data_val_opt['split'],
    random_sized_crop=data_val_opt['random_sized_crop'],
    add_data = data_val_opt['add_data'])

print(' The training dataset size :\t',len(dataset_train))
print('The validation dataset size :\t' ,len(dataset_val))

dloader_train = WCE_DataLoader(
    dataset=dataset_train,
    distort_type= data_train_opt['distortion_type'],
    batch_size=data_train_opt['batch_size'],
    epoch_size=data_train_opt['epoch_size'],
    num_workers=args_opt.num_workers,
    shuffle=True)

dloader_val = WCE_DataLoader(
    dataset=dataset_val,
    distort_type= data_train_opt['distortion_type'],
    batch_size=data_val_opt['batch_size'],
    epoch_size=data_val_opt['epoch_size'],
    num_workers=args_opt.num_workers,
    shuffle=False)
inv_tfm = dloader_train.inv_transform

dataset_test = WCE_Dataset(
    dataset_name=data_val_opt['dataset_name'],
    split=data_val_opt['split'],
    random_sized_crop=data_val_opt['random_sized_crop'],
    add_data = data_val_opt['add_data'])

dloader_test = WCE_DataLoader(
    dataset=dataset_val,
    distort_type= data_train_opt['distortion_type'],
    batch_size=data_val_opt['batch_size'],
    epoch_size=data_val_opt['epoch_size'],
    num_workers=args_opt.num_workers,
    shuffle=False)




net = MHAlexNet(arch_config)

if args_opt.gpu is not None: # Load model to cuda
    net = load_to_gpu(net, args_opt.visualize)
    set_GPU = True
else:
    print('CPU mode')
# load checkpoint
if args_opt.checkpoint > 0 : 
    net, starting_iter = load_checkpoint(net, ckp_directory)  
    if args_opt.visualize:
        # Unparallelize model
        if merge_seq: 
            collection =[]
            top = list(net.children())[0]
            for j in range(0, 8):
                collection.extend(list(top[j].children()))
            model = nn.Sequential(*collection)
        else: pass
        layer_vis = CNNLayerVisualization(model, 6, 84)
        layer_vis.visualise_layer_without_hooks()
        
else: starting_iter = 0
train = Train(config,  ckp_directory, set_GPU, inv_tfm, data_train_opt['distortion_type'])
train.train_step(net, dloader_train, dloader_test, starting_iter,args_opt.add_embed)


