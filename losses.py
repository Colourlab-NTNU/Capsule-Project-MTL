#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:16:55 2020

@author: user1
"""
import sys
import torch
import torch.nn as nn
class UncertaintyLoss(nn.Module):
    def __init__(self, ctype, copt):
        super(UncertaintyLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros((2), requires_grad=True, dtype=torch.float32).cuda())
        self.std_vars = [torch.exp(log_var) ** 0.5 for log_var in self.log_vars]
        self.crit = getattr(nn, ctype)(copt)

        
    def forward(self, net_output, gt_pathology, gt_distortion):

        precision1, precision2 = torch.exp(-self.log_vars[0]), torch.exp(-self.log_vars[1])

        loss_p = self.crit(net_output['pathology'], gt_pathology) 
        loss_d = self.crit(net_output['distortion_1'], gt_distortion)
        pathology_loss = precision1 * loss_p  + self.log_vars[0]
        d1_loss = precision2 * loss_d + self.log_vars[1]
        loss = torch.mean(pathology_loss + d1_loss)
        return loss, pathology_loss, d1_loss, self.log_vars.data.tolist()
    
