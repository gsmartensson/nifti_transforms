#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Example script of how to use the transformations of .nii(.gz) images.
    
    The transformations matrices (data augmentations) are multiplied before 
    being applied to the image, resulting in only having to do the time consuming
    part once.
    
    Input is the path to the .nii or .nii.gz image.

@author: gustav
"""
import numpy as np 
import matplotlib.pyplot as plt
import os
import transforms as tfs
np.random.seed(0)

# Filename to augment. Uses MNI brain template in FSL dir if installed.
fsl_path=os.environ['FSLDIR']
fname = os.path.join(fsl_path,'data','standard','MNI152_T1_1mm.nii.gz' )

#%% Four different augmentations
new_dim = [160,160,160] # output dimension of transformed images
new_res = [1,1,1]# output resolution (1mm x 1mm x 1mm)


# no random augmentation in transform
title_dict = {0:'No augmentation'}
t0=tfs.ComposeMRI([
                tfs.LoadNifti(), # image information stored as dict
                tfs.TranslateToCom(), # translate to image's center of mass
                tfs.SetResolution(new_dim= new_dim, new_res=new_res),
                tfs.CropShift(np.array([0,-1,-30])), # to "manually" shift image to center
                tfs.ApplyAffine(so=2), # apply all transforms
                tfs.ReturnImageData(), # dict -> numpy array
                tfs.ToTensor(), # from numpy to torch.tensor
                tfs.SwapAxes(0,1), # swap primary axis
                tfs.UnitInterval(), # normalize image to be in [-1,1]
                ])

# adding random shift
title_dict[1]='With random shift'
t1=tfs.ComposeMRI([
                tfs.LoadNifti(),
                tfs.TranslateToCom(),
                tfs.SetResolution(new_dim= new_dim, new_res=new_res),
                tfs.CropShift(np.array([0,-1,-30])),
                tfs.RandomShift([30,0,0]), # random shift
                tfs.ApplyAffine(so=2),
                tfs.ReturnImageData(),
                tfs.SwapAxes(0,1),
                tfs.ToTensor(),
                tfs.UnitInterval(),
                ])

# adding rotation
title_dict[2]='With random shift + rotation'
t2=tfs.ComposeMRI([
                tfs.LoadNifti(),
                tfs.TranslateToCom(),
                tfs.SetResolution(new_dim= new_dim, new_res=new_res),
                tfs.CropShift(np.array([0,-1,-30])),
                tfs.RandomShift([10,0,0]),  
                tfs.RandomRotation(angle_interval=[-10,10],rotation_axis=[0,1,0]),# random rotation
                tfs.ApplyAffine(so=2),
                tfs.ReturnImageData(),
                tfs.SwapAxes(0,1),
                tfs.ToTensor(),
                tfs.UnitInterval(),
                ])

# adding gamma transform + gaussian noise
title_dict[3]='Rand shift, rotation, gamma, noise'
t3=tfs.ComposeMRI([
                tfs.LoadNifti(),
                tfs.TranslateToCom(),
                tfs.SetResolution(new_dim= new_dim, new_res=new_res),
                tfs.CropShift(np.array([0,-1,-30])),
                tfs.RandomShift([10,0,0]),   
                tfs.RandomRotation(angle_interval=[10,10],rotation_axis=[0,1,0]),
                tfs.ApplyAffine(so=2),
                tfs.ReturnImageData(),
                tfs.SwapAxes(0,1),
                tfs.ToTensor(),
                tfs.UnitInterval(),
                tfs.Gamma(gamma_range=[.8,1.5],chance=1),
                tfs.RandomNoise(noise_var=.05),
                ])

#%% Plot example images
def plot_img(img,title_str=''):
    fig=plt.figure()
    plt.imshow(np.rot90(img[new_dim[0]//2,:,:]),cmap='gray',vmin=-1,vmax=.6); 
    
    plt.axis('off')#;plt.colorbar()
    plt.title(title_str)
    plt.show()
    fig.savefig(title_str.replace(' ','_').replace('+','_').replace(',','_').replace('__','_').replace('__','_').lower()+'.png')
    
for i,transforms in enumerate([t0,t1,t2,t3]):
    img = transforms(fname)
    plot_img(img,title_dict[i])
