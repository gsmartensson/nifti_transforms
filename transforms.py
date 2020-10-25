#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Jun 18 14:39:07 2020

@author: gustav
'''
from __future__ import print_function, division
import nibabel
from scipy import ndimage
import numpy as np
import torch
import matplotlib.pyplot as plt
import math


class LoadNifti(object):    
    def __call__(self,file):
    # Load mri feils with same orientation and numpy format        
        a = nibabel.load(file) # load file
        a= nibabel.as_closest_canonical(a) # transform into RAS orientation
        pixdim = a.header.get('pixdim')[1:4]
        a = np.array(a.dataobj) # fetch data as float 32bit
        a=np.float32(a)
        return {'data':a,'pixdim':pixdim,'affine':[],'new_dim':a.shape}

class CropShift(object):
    '''
        Shift center voxel of croping and adds to affine transformation matrix
    '''
    def __init__(self,shift):
        self.shift =np.array(shift)
    def __call__(self,image):
        ndims = len(image['data'].shape)
        T = np.identity(ndims+1)
        T[0:ndims,-1]=self.shift

        image['crop_shift']=T
        image['affine'].append('crop_shift')
        return image



class RandomShift(object):
    '''
        Randomly shift center voxel of croping between \pm max_shift.
        Adds transformation matrix to list of affine transformations.
    '''
    def __init__(self, max_shift=[0,0,0]):
        self.max_shift =np.array(max_shift)
    def __call__(self, image):
        ndims = len(image['data'].shape)
        
        shift = 2*(np.random.rand(ndims)-.5)*self.max_shift
        
        T = np.identity(ndims+1)
        
        T[0:ndims,-1]=shift

        image['random_shift']=T
        image['affine'].append('random_shift')
        return image

class RandomScaling(object):
    '''
        Adds an isotropic random scaling of the 3D image with a 
        factor s \in scale_range
    '''
    def __init__(self,scale_range=[.95,1.05]):
        self.scale_range=scale_range
        
    def __call__(self,image):
        old_res = np.array(image['pixdim'])
        
        scale_factor = np.random.rand()*np.diff(self.scale_range)+self.scale_range[0]
        scale_factor = np.ones(len(old_res))*scale_factor
        
        S = np.ones(old_res.size+1)
        S[0:len(scale_factor)] = scale_factor
        S = np.diag(S)
        
        image['random_scale']=S
        image['affine'].append('random_scale')
        return image


class SetResolution(object):
    '''
        Specify resolution and size of output array.
        args:
            new_dim: output dimensions, e.g. [128,128,64]
            new_res: output resolution, e.g. [1,1,2] (1mmx1mmx2mm), and scales 
                image appropriately based on information of original resolution
                in nifti header.
    '''
    def __init__(self,new_dim,new_res=None):
        self.new_res=new_res
        self.new_dim = np.array(new_dim)
    def __call__(self,image):
        old_res = np.array(image['pixdim'])
        if self.new_res==None: # don't change resolution, only matrix size
            new_res_tmp=image['new_dim']
        else:
            new_res_tmp=self.new_res
            image['new_dim'] = self.new_dim
        new_res_tmp=np.array(new_res_tmp)
        #old_size = np.array(image['data'].shape)
        scale_factor = (old_res/new_res_tmp)
        #scale_factor *= (self.new_dim/old_size)
        S = np.ones(old_res.size+1)
        S[0:len(scale_factor)] = scale_factor
        S = np.diag(S)
        #print(S)
        image['scale']=S
        image['affine'].append('scale')
        return image

class TranslateToCom(object):
    '''
        Translate image to center of mass ("Com"). Can be useful as a 
        quick-and-dirty method to e.g. "center" the brain in an MRI image.
        args:
            scale_f: downscaling factor used when calculating the center of mass.
                scale_f=1 yields precise COM coordinates but more computationally 
                expensive than e.g. scale_f=4.
    '''
    def __init__(self,scale_f=4):
        self.f = scale_f
    def __call__(self,image):
        img_tmp=image['data'][::self.f,::self.f,::self.f]
       # prc5 = np.percentile(img_tmp.ravel()[::4],5)
       # img_tmp=img_tmp>prc5
#        #plt.hist(img_tmp.ravel())
#        #plt.show()
        com = self.f*np.array(ndimage.center_of_mass(img_tmp))
#        com = self.f*np.array(ndimage.center_of_mass(image['data'][::self.f,::self.f,::self.f]))
        
        mid = np.array(image['data'].shape)/2
        #print(com)
        T = np.identity(len(mid)+1)
        T[0:len(mid),-1] = mid-com
        image['com'] = T
        image['affine'].append('com')
        return image
    
class RandomRotation(object):
    '''
        Adds a random rotation along a random or specifed axis.
        args:
            angle_interval: upper and lower bound of interval of angle (in degrees)
                of random rotation.
            rotation_axis: axis of which rotation occurs. If None, 
                then axis is random. If e.g. [1,0,0], the rotation is done in 
                coronal plane for brain images.
            
        Code adapted from https://stackoverflow.com/questions/47623582/efficiently-calculate-list-of-3d-rotation-matrices-in-numpy-or-scipy
    '''
    def __init__(self,angle_interval=[-5,5],rotation_axis=None):
        self.a_l,self.a_u =angle_interval
        self.rotation_axis =rotation_axis
    def unit_vector(self,data, axis=None, out=None):
        '''
            Return ndarray normalized by length, i.e. Euclidean norm, along axis.
        '''
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data*data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data
    def rotation_matrix(self,angle, direction, point=None):
        '''
            Return matrix to rotate about axis defined by point and direction.

        '''
        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = self.unit_vector(direction[:3])
        # rotation matrix around unit vector
        R = np.diag([cosa, cosa, cosa])
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array([[ 0.0,         -direction[2],  direction[1]],
                          [ direction[2], 0.0,          -direction[0]],
                          [-direction[1], direction[0],  0.0]])
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M
    def __call__(self,image):
        theta = np.random.uniform(self.a_l,self.a_u)

        angle =theta/180*np.pi
        if self.rotation_axis is None:
            u=np.random.rand(3)-.5
        else:
            u=self.rotation_axis
        u = u/(np.dot(u,u))            
        R = self.rotation_matrix(-angle,u)
        
        image['rotation']=R
        image['affine'].append('rotation')
        return image

class ApplyAffine(object):
    '''
        Multiply all previously added affine transformations matrices and
        apply them. Returns transformed image.
        args:
            new_dim:
            so: order of interpolation. Trade-off between speed and accuracy.
    '''
    def __init__(self,new_dim=None, so = 3,chance=1):
        self.chance=chance # if random number is below self.chance then apply transformation
        if not new_dim==None:
            self.new_dim = np.array(new_dim)
        else:
            self.new_dim = np.array([new_dim])
        self.so=so
    def __call__(self,image):
        if image['affine']==[] or np.random.rand()>self.chance:
            #print('no transform')
            return image
        else:
            # forward mapping
            if self.new_dim[0] is not None:
                if np.any(image['new_dim']!=image['data'].shape) and np.any(image['new_dim']!=self.new_dim):
                      raise Exception('Error - two different new_dim were given. Probably also in SetResolution()?')
#                    new_dim = image['new_dim']
                else:
#                    print('setting new dim (or same as before)')
                    image['new_dim'] = self.new_dim
                    new_dim = self.new_dim
            else:
                new_dim = image['new_dim']                
#                print('using previous new_dim')

            ndims = len(image['pixdim'])
            T=np.identity(ndims+1)
            for a in image['affine']:
                T = np.dot(image[a],T)
            T_inv = np.linalg.inv(T)
            
            # compute offset for centering translation
            c_in = np.array(image['data'].shape)*.5
            c_out=np.array(new_dim)*.5
            s=c_in-np.dot(T_inv[:ndims,:ndims],c_out)
            #tx,ty = -s
            translation =np.identity(ndims +1)
            translation[0:ndims,-1]=-s
            T_inv = np.dot(np.linalg.inv(translation),T_inv)

            image['data'] = ndimage.affine_transform(
                    image['data'],T_inv,output_shape=new_dim,order=self.so)
            return image
        
class ReturnImageData(object):
    '''
        Return image from dict after ApplyAffine() has been called
    '''
    def __call__(self,image):
        return image['data']

class Gamma(object):
    ''' 
        Apply gamma correction V_out = V_in^(gamma) at probability chance with 
        a random gamma value \in gamma_range.
        
        This tranformation should be applied after ReturnImageData() has been called.
    '''
    def __init__(self, gamma_range = [.8,1.2],chance=1):
        self.chance=chance # a value between [0,1]
        self.gamma_range=gamma_range 
    def __call__(self, image):
        if np.random.rand()<self.chance: # only do gamma if random number < chance
            img_min= image.min()
            img_max = image.max()
            image-=img_min
            image/=(np.abs(img_max - img_min) + 1e-9) # add small offset
            # generate random gamma value in gamma_range
            gamma=(np.random.rand())*(self.gamma_range[1] - self.gamma_range[0]) + self.gamma_range[0]
            image=np.power(image,gamma)
            image = image*(img_max - img_min) + img_min

        return image


class ToTensor(object):
    '''
        Convert np arrays in sample to Tensors.
    '''

    def __call__(self,image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)

        return image


class SwapAxes(object):
    '''Switch axes so that convolution is applied in different plane

    Args:
        axis1: dim1 to be swapped with axis2
    '''

    def __init__(self,axis1,axis2):
       self.axis1 =axis1
       self.axis2 =axis2
    def __call__(self, image):
        
        return np.swapaxes(image,self.axis1,self.axis2)


class Return4D(object):
    '''
    Returns 3D volume as a 4D object with 1 channel (i.e. dim: 1x h x w x d) to 
    work with PyTorch 3D functions

    Args:
        MRI volume
        image is a tensor
    '''
    def __call__(self, img):
        
        img =img.unsqueeze(0)

        return img
    

class Threshold(object):
    '''
    Threshold numpy values in image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    '''

    def __init__(self, lower_limit, upper_limit):
        
        self.ll = lower_limit
        self.ul = upper_limit
    def __call__(self, image):
        image[image<=self.ll]=self.ll
        image[image>=self.ul]=self.ul
        return image

class RandomNoise(object):
    '''Add random normally distributed noise to image tensor.

    Args:
        noise_var (float): maximum variance of added noise 
        p (float): probability of adding noise
    '''

    def __init__(self, noise_var=.1, p=1):
        
        self.noise_var = noise_var
        self.p = p
        
    def __call__(self, image):
        if torch.rand(1)[0]<self.p:
            var = torch.rand(1)[0]*self.noise_var
            image += torch.randn(image.shape)*var
            
        return image


class RandomMirrorLR(object):
    '''
        Randomly mirror an image in specifed plane
    '''
    def __init__(self, axis):
        self.axis=axis
    def __call__(self, image):
        if np.random.randn()>0: # 50/50 if to rotate        
            image = np.flip(image,self.axis).copy()
        return image    
        
class PerImageNormalization(object):
    ''' 
        Transforms all pixel values to to have mean= 0 and std = 1
    '''
    def __call__(self, image):
        image -=image.mean()
        image /=image.std()

        return image
    
class Window(object):
    ''' 
        Cap image to be between [low,high]
    '''
    def __init__(self, low,high):
        self.low=low
        self.high=high
    def __call__(self, image):
        # transform data to 
        
        image[image<self.low] =self.low
        image[image>self.high] =self.high

        return image
    
class PrcCap(object):
    ''' 
        Cap all pixel values between two percentiles
        args:
            low: lower percentile value to cap high pixel values to.
            high: upper percentile value to cap high pixel values to.
    '''
    def __init__(self, low=5,high=99):
        self.low = low
        self.high= high
    def __call__(self, image):
        # transform data to 
        
        l= np.percentile(image,self.low)
        h= np.percentile(image,self.high)
        image[image<l] = l; image[image>h] = h

        return image
    
class UnitInterval(object):
    ''' 
        Transforms all pixel values to be in [-1,1]
    '''

    def __call__(self, image):
        # transform data to 
        
        image -=image.min()
        image /=image.max()
        image = (image-.5)*2
        return image

class ComposeMRI(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    #TODO
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            #print(t)
            input= t(input)   
        return input