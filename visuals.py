#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:16:32 2020

@author: user1
"""


from __future__ import print_function
import os, tqdm, copy

import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision.utils import make_grid


def visualize_feature_maps(feats, epoch, steps, direc, writer):
    path = '%s/visuals'%direc
    
    for lyr in range(len(feats)-1):
        feat_map = feats[lyr][0, 0:64, :, :]
        fm = feat_map.view(feat_map.shape[0], -1, feat_map.shape[1], feat_map.shape[2])
        grid = make_grid(fm, nrow=8, normalize=True, scale_each= True)
        writer.add_image('Conv Block {} '.format(lyr), grid, epoch)
        plt.imshow(np.transpose(np.array(grid.cpu().data), (1, 2, 0)),cmap='viridis')
        plt.title('Feature map - CONV_{}_EPOCH_{}'.format(lyr, epoch))
        plt.savefig(path+'/CONV_{}_EPOCH_{}'.format(lyr, epoch))
    return


def gradcam():
    pass
        
    
def add_embed(feats, label_imgs, meta, tfm, step, writer, tag):
    tensor = np.zeros((label_imgs.shape[0], label_imgs.shape[2], label_imgs.shape[3], label_imgs.shape[1]))
    for i in range(label_imgs.shape[0]):
        tensor[i] = np.transpose(label_imgs[i], (1,2,0)) * np.array([[0.008, 0.005, 0.003]]) + np.array([0.364, 0.247, 0.138])
    tensor = np.transpose(tensor, (0,3,2,1))
    # log embeddings Once training is finished in the fc_block features :
    fc_feat = feats[-1].cpu().data
    # get the class labels for each image
    writer.add_embedding(fc_feat,
                           metadata= meta,
                           label_img=tensor,
                          global_step=step,
                          tag=tag
                          )
    return 
def write_conv(features, net, epoch, steps, writer):
    conv1_ = make_grid(list(nn.ModuleList(list(net.children())[0].children())[0].parameters())[0][:, 0:3, :, :],
               nrow=8, normalize=True, scale_each=True)
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.transpose(conv1_.detach().cpu().numpy(), (1,2,0)))   
    writer.add_figure('conv1', fig, epoch)
    # writer.add_image('conv1', conv1_, epoch)
    conv5_ = make_grid(list(nn.ModuleList(list(net.children())[0].children())[0][6][0].parameters())[0][:, 0:1, :, :],
               nrow=16, normalize=True, scale_each=True)
    writer.add_image('conv5', conv5_, epoch)
    
    return