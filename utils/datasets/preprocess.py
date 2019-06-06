import torch
import random
from torchvision import transforms
import numpy as np
from PIL import Image, ImageChops

def get_match_transform_ops(resize=480, normalize=True, crop='center', crop_size=448):
    ops = []
    if resize:
        ops.append(transforms.Resize(size=(resize, resize)))
    if crop == 'center':
        ops.append(transforms.CenterCrop(size=crop_size))
    elif crop == 'random':
        ops.append(transforms.RandomCrop(size=crop_size))
    if normalize: 
        ops.append(ToTensorScaled())
        ops.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        ops.append(ToTensorUnscaled())
    return transforms.Compose(ops)

def get_transform_ops(resize=256, crop='center', crop_size=224):
    ops = []
    if resize:
        ops.append(transforms.Resize(resize, Image.BICUBIC))
    if crop == 'center':
        crop = CenterCropNumpy(crop_size)
        ops.append(crop)       
    elif crop == 'random':
        crop = RandomCropNumpy(crop_size)
        ops.append(crop)
    ops.append(ToTensorUnscaled())
    return transforms.Compose(ops)

def get_pair_transform_ops(resize=256, crop='center', crop_size=224,
                           scale=False, normalize=False):
    ops = []
    if resize:
        ops.append(PairResize(resize))
    if crop == 'center':
        crop = PairCenterCropNumpy(crop_size)
        ops.append(crop)
    elif crop == 'random':
        crop = PairRandomCropNumpy(crop_size)
        ops.append(crop)
    if normalize: 
        ops.append(PairToTensorScaled())
        ops.append(PairNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        ops.append(PairToTensorUnscaled())
    return PairCompose(ops)

class PairToTensor(object):
    """Convert a pair of RGB PIL Image to a CHW ordered Tensor,
    default pytorch ToTensor scales the image value to [0, 1]"""
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        
    def __call__(self, im1, im2):
        return self.to_tensor(im1), self.to_tensor(im2)

    def __repr__(self):
        return 'PairToTensor() [Default in pytorch]'

class ToTensorScaled(object):
    '''Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]'''
    def __call__(self, im):
        im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
        im /= 255.0 
        return torch.from_numpy(im)

    def __repr__(self):
        return 'ToTensorScaled(./255)'
    
class PairToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()
        
    def __call__(self, im1, im2):
        return self.to_tensor(im1), self.to_tensor(im2)

    def __repr__(self):
        return 'PairToTensorScaled(./255)'

class ToTensorUnscaled(object):
    '''Convert a RGB PIL Image to a CHW ordered Tensor'''
    def __call__(self, im):    
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return 'ToTensorUnscaled()'

class PairToTensorUnscaled(object):
    '''Convert a RGB PIL Image to a CHW ordered Tensor'''
    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im1, im2):
        return self.to_tensor(im1), self.to_tensor(im2)

    def __repr__(self):
        return 'PairToTensorUnscaled()'

class CenterCropNumpy(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, im):
        im = np.array(im)
        size = self.size    
        h, w, _ = im.shape
        if w == size and h == size:
            return im
        x = int(round((w - size) / 2.))
        y = int(round((h - size) / 2.))
        return im[y:y+size, x:x+size, :]
    
    def __repr__(self):
        return 'CenterCropNumpy(size={})'.format(self.size)    

class PairCenterCropNumpy(object):
    def __init__(self, size):
        self.center_crop = CenterCropNumpy(size)
        self.size = size

    def __call__(self, im1, im2):
        return self.center_crop(im1), self.center_crop(im2)

    def __repr__(self):
        return 'PairCenterCropNumpy(size={})'.format(self.size)

class RandomCropNumpy(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        im = np.array(im)
        size = self.size
        h, w, _ = im.shape
        if w == size and h == size:
            return im
        x = np.random.randint(0, w - size)
        y = np.random.randint(0, h - size)
        return im[y:y+size, x:x+size, :]

    def __repr__(self):
        return 'RandomCropNumpy(size={})'.format(self.size) 

class PairRandomCropNumpy(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im1, im2):
        im1 = np.array(im1)
        im2 = np.array(im2)
        size = self.size
        assert im1.shape == im2.shape
        h, w, _ = im1.shape
        if w == size and h == size:
            return im1, im2
        x = np.random.randint(0, w - size)
        y = np.random.randint(0, h - size)
        return im1[y:y+size, x:x+size, :], im2[y:y+size, x:x+size, :]

    def __repr__(self):
        return 'PairRandomCropNumpy(size={})'.format(self.size)

class PairResize(object):
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(size, Image.BICUBIC)

    def __call__(self, im1, im2):
        return self.resize(im1), self.resize(im2)

    def __repr__(self):
        return 'PairResize(size={})'.format(self.size)
    
class PairNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im1, im2):
        return self.normalize(im1), self.normalize(im2)

    def __repr__(self):
        return 'PairNormalize(mean={}, std={})'.format(self.mean, self.std)

class PairCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im1, im2):
        for t in self.transforms:
            im1, im2 = t(im1, im2)
        return im1, im2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string