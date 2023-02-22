#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageEnhance 
# from .DefocusBlur import  DefocusBlur_random
from .GaussianBlur import GaussianBlur_random
from .LinearMotionBlur import LinearMotionBlur_random
from .PsfBlur import PsfBlur_random
# from GaussianBlur import GaussianBlur_random
# from LinearMotionBlur import LinearMotionBlur_random
# from PsfBlur import PsfBlur_random

blurFunctions     = {0: None, 1: GaussianBlur_random, 2: LinearMotionBlur_random, 3: PsfBlur_random}
ContrastFunctions = {0: None, 1: 0.8, 2: 1.2, 3: 1.3 }
SaturateFunctions = {0: None, 1: 1.2, 2: 1.7, 3: 2}
BrightenFunctions = {0: None, 1: 0.8, 2: 1.2, 3: 1.4}
def blur(img, blur_type):
    if blur_type == 0:
        return img
    else :
        blurToApply = blurFunctions[blur_type]
        return blurToApply(img)

def contrast(img, level):
    contrastToApply = ContrastFunctions[level]
    # print(contrastToApply)
    if contrastToApply == None:
        return img
    else:
        LC = ImageEnhance.Contrast(img)
        # LC.enhance(level).show() 
        return LC.enhance(contrastToApply) 

    
def saturate(img, level):
    saturateToApply = SaturateFunctions[level]
    if saturateToApply == None:
        return img
    else:
        sat = ImageEnhance.Color(img)
        return sat.enhance(saturateToApply)
        # sat.enhance(level).show() 
    
#[0 - less, 1 - original, >1 - saturated]  # saturate a channel randomly
    
def brighten(img, level):
    brightToApply = BrightenFunctions[level]
    # print(brightToApply)
    if brightToApply == None:
        return img
    else:
        bright = ImageEnhance.Brightness(img)
        return bright.enhance(brightToApply)
   # br.enhance(level).show()
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img_path = '/home/user1/PhD_CAPSULEAI/CAPSULEAI_DATA/IQMED/wcetraining/wcetraining_baseline_clf/train/inflammatory/U501.jpg'
    img = Image.open(img_path)
    # blurred_img = blur(img, 0)
    # plt.imshow(blurred_img)
    # plt.show()
    # blurred_img.show() 
    # img = saturate(img, 0)
    img1 = brighten(img, 0)
    img2 = brighten(img, 1)
    img3 = brighten(img, 2)
    img4 = brighten(img, 3)
    print(np.unique(np.asarray(img1) - np.asarray(img2)))
    # Brightness(img, 1.5)
    plt.imshow(img1)
    plt.show()
    plt.imshow(img2)
    plt.show()
    plt.imshow(img3)
    plt.show()
    plt.imshow(img4)
    plt.show()
