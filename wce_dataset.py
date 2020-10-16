#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 11:29:20 2020

@author: user1
"""


from __future__ import print_function
import torch
import torch.utils.data as data
import torchnet as tnt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random, cv2, sys
from torch.utils.data.dataloader import default_collate
from PIL import Image

from pdb import set_trace as breakpoint

# Set the paths of the datasets here.
_WCE_DATASET_DIR = '/home/user1/PhD_CAPSULEAI/CAPSULEAI_DATA/IQMED/wcetraining/wcetraining_baseline_clf'
_KID_DATASET_DIR = '/home/user1/PhD_CAPSULEAI/CAPSULEAI_DATA/IQMED/wcetraining/KID'
# {'inflammatory': 0, 'normal': 1, 'vascularlesion': 2}

class WCE_Dataset(data.Dataset):
    def __init__(self, dataset_name, split, random_sized_crop=False,
                 add_data='', num_imgs_per_cat=None):
        self.split = split.lower()
        self.dataset_name =  dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split
        self.random_sized_crop = random_sized_crop
        self.additional_data = add_data

        self.num_imgs_per_cat = num_imgs_per_cat
        
        self.mean_pix = [0.364, 0.247, 0.138]
        self.std_pix = [0.008, 0.005, 0.003]
        
        if self.random_sized_crop:
            raise ValueError('The random size crop option is not supported for the WCE dataset')
                
        transform = []
        if (split != 'test'):  
            transform.append(transforms.Resize((576,576)))                  
            transform.append(transforms.CenterCrop(400))
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(lambda x: np.asarray(x))
        self.transform = transforms.Compose(transform)
        split_data_dir = _WCE_DATASET_DIR + '/' + self.split
        self.data = datasets.ImageFolder(split_data_dir, self.transform)
        if self.additional_data  == 'KID':
            add_transform = []
            if (split != 'test'):
                add_transform.append(transforms.Resize((360,360)))
                add_transform.append(transforms.CenterCrop(240))
                add_transform.append(transforms.Resize((400,400)))    # TODO : Check where to resize like this. 
                add_transform.append(transforms.RandomHorizontalFlip())
            add_transform.append(lambda x: np.asarray(x))
            self.add_transform = transforms.Compose(add_transform)
            add_split_data_dir = _KID_DATASET_DIR + '/' + self.split
            self.add_data = datasets.ImageFolder(add_split_data_dir, self.add_transform)
            self.data = torch.utils.data.ConcatDataset((self.data, self.add_data))
        
    
        if num_imgs_per_cat is not None:
            raise ValueError('The num_imgs_per_cat is currently not supported  for the WCE dataset as it is fully supervised')
    

    def __getitem__(self, index):
        
        img, label = self.data[index]
        # plt.imshow(img)
        # plt.show()
        return img, int(label)

    def __len__(self):
        return len(self.data)

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
def one_hot(a, num_classes=4):
  return torch.tensor(np.squeeze(np.eye(num_classes)[a.reshape(-1)])).type(torch.LongTensor)

def motion_blur(img, angle, size=None) :
    size = 10 if size == None else size
    k = np.zeros((size, size), dtype=np.float32)
    k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    k = k * ( 1.0 / np.sum(k) )        
    # print(k)
    return cv2.filter2D(img, -1, k)

def gaussian_blur(img, lev):
    pass

class WCE_DataLoader(object):
    def __init__(self,
                 dataset,
                 batch_size=1,
                 epoch_size=None,
                 num_workers=0,
                 shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.epoch_size = epoch_size if epoch_size is not None else len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers

        mean_pix  = self.dataset.mean_pix
        std_pix   = self.dataset.std_pix
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        self.inv_transform = transforms.Compose([
            Denormalize(mean_pix, std_pix),
            lambda x: x.numpy() * 255.0,
            lambda x: x.transpose(1,2,0).astype(np.uint8),
        ])

    def get_iterator(self, epoch=0):
        rand_seed = epoch * self.epoch_size
        random.seed(rand_seed)

        def _load_function(idx):
            idx = idx % len(self.dataset)
            img0, pathology_label = self.dataset[idx] # dropped the class label
            blur_setting = random.randint(1,4)
            # for deactivating Blur head set blur setting to 1
            # blur_setting = 1
            if blur_setting == 1:
                motion_blurred_img = self.transform(img0)                           # No blur
                motion_blur_label = 0                        
            elif blur_setting == 2:
                angle = random.choice([0, 180])
                motion_blurred_img = self.transform(motion_blur(img0, angle).copy())     # Horizontal blur
                motion_blur_label = 1  
            elif blur_setting == 3:
                angle = random.choice([90, 270])
                motion_blurred_img = self.transform(motion_blur(img0, angle).copy()) 
                motion_blur_label = 2                                               # Vertical blur
            elif blur_setting == 4:
                angle = random.choice([45, 135])
                motion_blurred_img = self.transform(motion_blur(img0, angle).copy())
                motion_blur_label = 3                                               # Diagonal blur
            # plt.imshow(self.inv_transform(motion_blurred_img))
            # plt.show()
            motion_blur_label = torch.tensor(motion_blur_label) # {0 : No blur , 1 : Horizontal, 2 : Vertical, 3: diagnonal}
            patho_label = torch.tensor(pathology_label)
            dict_data = {
            'imgs': motion_blurred_img,
            'labels': {
                'pathology': patho_label,
                'mblur': motion_blur_label
                }}
            return dict_data
        _collate_fun = default_collate

        tnt_dataset = tnt.dataset.ListDataset(elem_list=range(self.epoch_size),
            load=_load_function)
        data_loader = tnt_dataset.parallel(batch_size=self.batch_size,
            collate_fn=_collate_fun, num_workers=self.num_workers,
            shuffle=self.shuffle)
    
        return data_loader
    
    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size / self.batch_size

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = WCE_Dataset('wce','train', random_sized_crop=False, add_data='KID')
    dataloader = WCE_DataLoader(dataset, batch_size=8)
    patho_list = ['inflammatory', 'normal', 'vascularlesion']
    for b in dataloader(0):
        data_dict = b
        data = data_dict['imgs']
        label_p, label_d =  data_dict['labels']['pathology'], data_dict['labels']['mblur']
        break

    inv_transform = dataloader.inv_transform
    for i in range(data.size(0)):
        plt.subplot(data.size(0)/4,4,i+1)
        fig=plt.imshow(inv_transform(data[i]))
        plt.title(patho_list[label_p[i]])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.show()
