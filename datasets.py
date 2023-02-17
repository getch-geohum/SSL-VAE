import numpy as np
import random
import torch
from torch import nn
from torchvision.models.resnet import resnet18, resnet50
from torch.utils.data import Dataset, DataLoader
from skimage.transform import rescale
from torchvision import transforms
import copy
from glob import glob
import os
import matplotlib.pyplot as plt
from skimage.io import imread
import cv2


def make_normal(fil):
    return (fil - np.amin(fil)) / (np.amax(fil) - np.amin(fil))

def NDVI(image, channel='first', normalize=True, func=True):
        if normalize:
            if func:
                image = make_normal(image)
            else:
                image = image/1000

        if channel=='first':
            r = image[-2, :, :]
            n = image[-1, :, :]
        else:
            r = image[:, :, -2]
            n = image[:, :, -1]
        const = 0.0000001
        ndvi = ((n-r)+const)/((r+n)+const)
        return ndvi

def grayIntensity(x, equalize=False, treshold=120, b_treshold=0, c_treshold=0, nb_channels=4):# 120
    print(f'b_treshold: {b_treshold}')
    print(f'c_treshold: {c_treshold}')

    rgb = np.dstack((x[:,:,2], x[:,:,1], x[:,:, 0]))
    gray = cv2.cvtColor((rgb*255).astype(np.uint8),cv2.COLOR_BGR2GRAY)
    contrast = round(gray.std(),2)
    brightness = round(np.sqrt(gray.mean()),2)
    if equalize:
        gray = cv2.equalizeHist(gray)

    contrast = round(gray.std(),2)
    brightness = round(np.sqrt(gray.mean()),2)

    if (b_treshold > 0) and (brightness < b_treshold):
        print(f'got b_treshold: {b_treshold}')
        reta, mask = cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif (c_treshold > 0) and (contrast < c_treshold):
        print(f'got c_treshold: {c_treshold}')
        reta, mask = cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        mask = np.where(gray<=treshold,0,1).astype(np.uint8)
        print('Mask will be generated without acounting low ontrast objects')
    mask_clean = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3), int), iterations = 1) # cler ver smaller detached objects
    big_mask = np.dstack([mask_clean]*nb_channels) # multi-channel mask based on number of channels specifid
    if np.sum(mask_clean)>=16:
        mask_image = np.where(big_mask==1, x, big_mask)
        return (mask_image, big_mask)
    else:
        return None
        
        
def channelIntensity(self,x, channel_tresholds=[0.75, 0.65, 0.65], channel='last', approach='max', nb_channels=4):
    if channel == 'last':
        nc = x.shape[-1]
        a, b, c = x[:,:,0], x[:,:,1], x[:,:,2]
    else:
        nc = x.shape[0]
        a, b, c = x[0,:,:], x[1,:,:], x[2, :,:]

    assert len(channel_tresholds) == nc, 'number of provided channel tresholds and number of channels is not the same'
    aa = np.where(a>channel_tresholds[0],1,0)
    bb = np.where(b>=channel_tresholds[1],1,0)
    cc = np.where(c>=channel_tresholds[2],1,0)
    dd = np.dstack((aa, bb, cc))

    if approach == 'mode':
        m_tensor = torch.from_numpy(dd)
        m_tensor = m_tensor.mode(axis=-1, keepdim=True)
        mask = m_tensor.values.squeeze().numpy().astype(np.uint8)
    elif approach == 'max':
        mask = np.max(dd, axis=-1)
    elif approach == 'intersection':
        mask = aa*bb*cc
    mask_clean = cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3), int), iterations = 1)
    big_mask = np.dstack([mask_clean]*nb_channels)

    if np.sum(mask_clean)>=16:
        mask_image = np.where(big_mask==1, x, big_mask)
        return (mask_image, big_mask)
    else:
        print(f'Empty image obtained... ')
        None
        
        
def ndviMasker(x, treshold=0.2, nb_channels=None):
    ndvi = NDVI(image=x, channel='last', normalize=False, func=True)
    mask = np.where(ndvi<=treshold,1,0).astype(np.uint8)
    mask_clean = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3), int), iterations = 1) 
    big_mask = np.dstack([mask_clean]*nb_channels) 
    if np.sum(mask_clean)>=16:
        mask_image = np.where(big_mask==1, x, big_mask)
        return (mask_image, big_mask)
    else:
        return None    

def im_blurr(img, factor=16):
    img_down = rescale(img, 1/factor, multichannel=True, anti_aliasing=True)
    blur = rescale(img_down, factor, multichannel=True, anti_aliasing=True)
    assert img.shape == blur.shape
    return blur

def toFloat(x):
    return x.float()

class TrainDataset(Dataset):
    def __init__(
        self,
        root, 
        func,
        equalize=True,
        ndvi_treshold=0.2,
        intensity_treshold=120,
        nb_channels=4,
        fake_dataset_size=None,
        c_treshold=0,
        b_treshold=0,
        with_prob=True
    ):
        self.root = root
        self.func = func
        self.equalize=equalize
        self.ndvi_treshold = ndvi_treshold
        self.intensity_treshold = intensity_treshold
        self.nb_channels = nb_channels
        self.fake_dataset_size = fake_dataset_size
        self.c_treshold = c_treshold
        self.b_treshold = b_treshold
        self.with_prob = with_prob
        
        self.train_path = f'{self.root}/train'
        self.test_path = f'{self.root}/test'
        
        self.img_dir = sorted(glob(f'{self.train_path}/images/*.tif'))
        self.lbl_dir = sorted(glob(f'{self.test_path}/images/*.tif'))
        print("Number of train images", len(self.img_dir), 
                "Number of test images", len(self.lbl_dir),
                "Fake dataset size", self.fake_dataset_size,
                "brightness treshold", self.b_treshold,
                "contrast treshold", self.c_treshold,
                "Tresholding function", self.func)

        if ((self.fake_dataset_size is not None)
            and (self.fake_dataset_size < len(self.img_dir))):
            #inds = list(range(min(len(self.lbl_dir), len(self.img_dir))))
            inds = list(range(len(self.img_dir)))
            print(f'Length of indexes for sampling: {len(inds)}')
            sample = random.sample(inds, self.fake_dataset_size)
            self.img_dir = [self.img_dir[ind] for ind in sample]
            #self.lbl_dir = [self.lbl_dir[ind] for ind in sample]

            print("Number of train images after restriction", len(self.img_dir), 
                "Number of test images after restriction", len(self.lbl_dir))

        self.image_array = self.image2Array(self.img_dir)   # read image arra
        self.mask_array = self.computeMask(self.lbl_dir,
                                           func=self.func,
                                           ndvi_treshold=self.ndvi_treshold,
                                           intensity_treshold=self.intensity_treshold,
                                           equalize=self.equalize,
                                           c_treshold=self.c_treshold,
                                           b_treshold=self.b_treshold,
                                           nb_channels=self.nb_channels)   # read image arra and compute array
        
        N = len(self.image_array)
        n = len(self.mask_array)
        dif = N-n

        if dif>0:
            added = [random.choice(self.mask_array) for i in range(dif)]
            print(f'Sampled mask: {len(added)}')
            self.mask_array = self.mask_array + added
            print(f'image lenegth: {len(self.image_array)}, mask length: {len(self.mask_array)}')
            print(f'length of masks: {len(self.mask_array[0])}')
        elif dif<0:
            added_img = [random.choice(self.image_array) for i in range(abs(dif))]
            self.image_array = self.image_array + added_img
        else:
            print('Images and maks have equal length')
        
        masked_a = [data[0] for data in self.mask_array]
        masked_l = [data[1] for data in self.mask_array]
        mod_a = [np.where(masked_a[i]==0,self.image_array[i], masked_a[i]) for i in range(len(masked_a))]
        
        self.normal = copy.deepcopy(self.image_array)
        self.modified = copy.deepcopy(mod_a)
        self.mask = copy.deepcopy(masked_l)
        self.mask_inv = [np.where(image==0,1,0) for image in self.mask]
        
        print(len(self.normal), len(self.modified), len(self.mask))
        
        del self.image_array 
        del self.mask_array
        del masked_a
        del masked_l
        del mod_a
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(toFloat)
            #transforms.Lambda(lambda img: img.float())
        ])

    
    def image2Array(self, images):
        return [make_normal(imread(image)) for image in images]
        
    def computeMask(self, files, func='NDVI', ndvi_treshold=None, intensity_treshold=None, equalize=None, c_treshold=None, b_treshold=None, nb_channels=None):
        arrays = self.image2Array(files)
        if func == 'NDVI': 
            masks = [ndviMasker(img, treshold=ndvi_treshold, nb_channels=nb_channels) for img in arrays] 
        else: 
            if func == 'grayIntensity':
                masks = [grayIntensity(img, equalize=equalize, treshold=intensity_treshold, c_treshold=c_treshold, b_treshold=b_treshold, nb_channels=nb_channels) for img in arrays]
            else:
                print("========================================")
                masks = [channelIntensity(img, nb_channels=nb_channels) for img in arrays]    
        masks = [mask for mask in masks if mask is not None]
        return masks
        
    def __len__(self):
        return len(self.normal)

    def __getitem__(self, index):
        nrm = self.normal[index]
        mdf = self.modified[index]  # modified image
        msk = self.mask[index]      # mask for modified image region
        msk_iv = self.mask_inv[index] # mask for unmodified region
        if self.with_prob:
            p = np.where(msk[:,:,0] == 1, np.sum(msk[:,:,0])/(256*256), 1-(np.sum(msk[:,:,0])/(256*256)))     # this P which is teh probability of a pixel being 0 or 1
            p = np.dstack([p]*self.nb_channels)  # duplicate channels for element wise multiplication
            #print(f"shape of the probability raster is: {p.shape}")
            return self.transform(nrm), self.transform(mdf), self.transform(msk), self.transform(msk_iv), self.transform(p) # nrm
        else:
            return self.transform(nrm), self.transform(mdf), self.transform(msk), self.transform(msk_iv)
    
    
class TestDataset(Dataset):
    def __init__(self,
                 root,
                 func=None,
                 equalize=True,
                 ndvi_treshold=0.2,
                 intensity_treshold=120,
                 nb_channels=4,
                 fake_dataset_size=None,
                 c_treshold=0,
                 b_treshold=0,
                 with_mask=True):
        self.root = root
        self.func=func
        self.equalize=equalize
        self.ndvi_treshold=ndvi_treshold
        self.intensity_treshold=intensity_treshold
        self.nb_channels=nb_channels
        self.fake_dataset_size=fake_dataset_size
        self.test_path = f'{root}/test'
        self.with_mask = with_mask
        self.nb_channels = nb_channels
        self.c_treshold = c_treshold
        self.b_treshold = b_treshold
        
        self.img_dir = sorted(glob(f'{self.test_path}/images/*.tif'))
        self.lbl_dir = sorted(glob(f'{self.test_path}/labels/*.tif'))
            
        if self.fake_dataset_size is not None:
            inds = list(range(len(self.img_dir)))
            sample = random.sample(inds, self.fake_dataset_size)
            self.img_dir = [self.img_dir[ind] for ind in sample]
            self.lbl_dir = [self.lbl_dir[ind] for ind in sample]

        self.image_array = self.image2Array(self.img_dir, normalize=True)   # read image arra
        self.lbl_array = self.image2Array(self.lbl_dir, normalize=False)   # read image arra and compute array
        if self.with_mask:
            self.mask_array = self.computeMask(self.img_dir,
                                               func=self.func,
                                               ndvi_treshold=self.ndvi_treshold,
                                               intensity_treshold=self.intensity_treshold,
                                               equalize=self.equalize,
                                               c_treshold=self.c_treshold,
                                               b_treshold=self.b_treshold,
                                               nb_channels=self.nb_channels) # self.img_dir is the directory that contained 
        
        
        assert len(self.image_array) == len(self.mask_array), f'image {len(self.image_array)} and label {len(self.mask_array)} numbers are not the same'
        
        print(f'Obtained {len(self.image_array)} images and labels for testing')
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(toFloat)
            #transforms.Lambda(lambda img: img.float())
        ])

    
    def image2Array(self, images, normalize=False):
        if not normalize:
            return [imread(image).astype(float) for image in images]
        else:
            return [make_normal(imread(image).astype(float)) for image in images]
        
    def computeMask(self, files, func='NDVI', ndvi_treshold=None, intensity_treshold=None, equalize=None, c_treshold=None, b_treshold=None, nb_channels=None):
        arrays = self.image2Array(files)
        if func == 'NDVI': 
            masks = [ndviMasker(img, treshold=ndvi_treshold, nb_channels=nb_channels) for img in arrays] 
        else: 
            if func == 'grayIntensity':
                masks = [grayIntensity(img, equalize=equalize, treshold=intensity_treshold, c_treshold=c_treshold, b_treshold=b_treshold, nb_channels=nb_channels) for img in arrays]
            else:
                print("========================================")
                masks = [channelIntensity(img, nb_channels=nb_channels) for img in arrays]    
        masks = [mask if mask is not None else np.zeros((256,256,nb_channels), dtype=np.uint8) for mask in masks]
        return masks
        
    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, index):
        img = self.image_array[index]
        lbl = self.lbl_array[index]
        if not self.with_mask:
            return self.transform(img), self.transform(lbl)
        else:
            msk = self.mask_array[index][1]
            # print(f"type of mask: {type(msk)}")
            # print(f"type of label: {type(lbl)}")
            # print(f"type of image: {type(img)}")
            return self.transform(img), self.transform(lbl) #, self.transform(msk)



