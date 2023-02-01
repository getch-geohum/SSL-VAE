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
        b_treshold=0
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
        
        self.train_path = f'{self.root}/train'
        self.test_path = f'{self.root}/test'
        
        self.img_dir = sorted(glob(f'{self.train_path}/images/*.tif'))
        self.lbl_dir = sorted(glob(f'{self.test_path}/images/*.tif'))
        print("Number of train images", len(self.img_dir), 
                "Number of test images", len(self.lbl_dir),
                "Fake dataset size", self.fake_dataset_size,
                "brightness treshold", self.b_treshold,
                "contrast treshold", self.c_treshold)

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
                                           b_treshold=self.b_treshold)   # read image arra and compute array
        
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
        #del self.added
        del masked_a
        del masked_l
        del mod_a
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.float())
        ])

    def im_blurr(self, img, factor=16): # not used for this AE experiemnt
        img_down = rescale(img, 1/factor, multichannel=True, anti_aliasing=True)
        blur = rescale(img_down, factor, multichannel=True, anti_aliasing=True)
        assert img.shape == blur.shape
        return blur
    
    def make_normal(self, fil):
        return (fil - np.amin(fil)) / (np.amax(fil) - np.amin(fil))
    
    def NDVI(self, image, channel='first', normalize=True, func=True):
        if normalize:
            if func:
                image = self.make_normal(image)
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
    
    def image2Array(self, images):
        return [self.make_normal(imread(image)) for image in images]
    
    def intensityMasker(self, x, equalize=False, treshold=120, b_treshold=0, c_treshold=0):# 120
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
        big_mask = np.dstack([mask_clean]*self.nb_channels) # multi-channel mask based on number of channels specifid
        if np.sum(mask_clean)>=16:
            mask_image = np.where(big_mask==1, x, big_mask)
            return (mask_image, big_mask)
        else:
            return None
    
    def ndviMasker(self, x, treshold=0.2):
        ndvi = self.NDVI(image=x, channel='last', normalize=False, func=True)
        mask = np.where(ndvi<=treshold,1,0).astype(np.uint8)
        mask_clean = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3), int), iterations = 1) # cler ver smaller detached objects
        big_mask = np.dstack([mask_clean]*self.nb_channels) # multi-channel mask based on number of channels specified 
        if np.sum(mask_clean)>=16:
            mask_image = np.where(big_mask==1, x, big_mask)
            return (mask_image, big_mask)
        else:
            return None
        
    def computeMask(self, files, func='NDVI', ndvi_treshold=None, intensity_treshold=None, equalize=None, c_treshold=None, b_treshold=None):
        arrays = self.image2Array(files)
        if func == 'NDVI': # ndvi based tresholding
            masks = [self.ndviMasker(img, treshold=ndvi_treshold) for img in arrays] 
        else: # intensity based tresholding
            masks = [self.intensityMasker(img, equalize=equalize, treshold=intensity_treshold, c_treshold=c_treshold, b_treshold=b_treshold) for img in arrays]
            
        masks = [mask for mask in masks if mask is not None]
        return masks
        
    def __len__(self):
        return len(self.normal)

    def __getitem__(self, index):
        nrm = self.normal[index]
        mdf = self.modified[index]
        msk = self.mask[index]
        msk_iv = self.mask_inv[index]
        return self.transform(nrm), self.transform(mdf), self.transform(msk), self.transform(msk_iv)
    
class TestDataset(Dataset):
    def __init__(self, root, fake_dataset_size=None):
        self.root = root
        self.fake_dataset_size=fake_dataset_size
        self.test_path = f'{root}/test'
        
        self.img_dir = sorted(glob(f'{self.test_path}/images/*.tif'))
        self.lbl_dir = sorted(glob(f'{self.test_path}/labels/*.tif'))
            
        if self.fake_dataset_size is not None:
            inds = list(range(len(self.img_dir)))
            sample = random.sample(inds, self.fake_dataset_size)
            self.img_dir = [self.img_dir[ind] for ind in sample]
            self.lbl_dir = [self.lbl_dir[ind] for ind in sample]

        self.image_array = self.image2Array(self.img_dir, normalize=True)   # read image arra
        self.mask_array = self.image2Array(self.lbl_dir, normalize=False)   # read image arra and compute array
        
        
        assert len(self.image_array) == len(self.mask_array), f'image {len(self.image_array)} and label {len(self.mask_array)} numbers are not the same'
        
        print(f'Obtained {len(self.image_array)} images and labels for testing')
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.float())])

    def im_blurr(self, img, factor=16):
        img_down = rescale(img, 1/factor, multichannel=True, anti_aliasing=True)
        blur = rescale(img_down, factor, multichannel=True, anti_aliasing=True)
        assert img.shape == blur.shape
        return blur
    
    def make_normal(self, fil):
        return (fil - np.amin(fil)) / (np.amax(fil) - np.amin(fil))
    
    def image2Array(self, images, normalize=False):
        if not normalize:
            return [imread(image).astype(float) for image in images]
        else:
            return [self.make_normal(imread(image).astype(float)) for image in images]
        
    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, index):
        img = self.image_array[index]
        msk = self.mask_array[index]
        return self.transform(img), self.transform(msk)
