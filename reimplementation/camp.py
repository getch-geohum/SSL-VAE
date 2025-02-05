import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from skimage.io import imread
import numpy as np
from shapely.geometry import Polygon
from skimage import measure
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import KDTree

def make_normal(img):
    arr = np.clip(((img - np.amin(img)) + 0.00001) / ((np.amax(img) - np.amin(img)) + 0.00001),0,1)
    return arr

def toFloat(x):
    return x.float()

def gaussian_filter_density(img_shape=(256,256), points=None):
    '''Adapted from https://github.com/CommissarMa/MCNN-pytorch'''
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 10  # default
    beta = 0.08 # see https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf
   
    tree = KDTree(points.copy(), leafsize=leafsize) # to generate nearesk k distaces

    distances, locations = tree.query(points, k=5)

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        pt2d[pt[0], pt[1]] = 1.
        if gt_count > 1:
            sigma = ((distances[i][1]+distances[i][2]+distances[i][3]+distances[i][4])/4)*beta
        else:
            sigma = np.average(np.array(gt.shape))/2./2. # incase: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density

def getDesnity(img, sigma): # assuming the less dense dwellings
    mask = np.zeros(img.shape, img.dtype)
    contours = measure.find_contours(img, 0.5)
    for i, cont in enumerate(contours):
        try:
            poly = Polygon(cont) # .simplify(1.0, preserve_topology=False)
            cents = poly.centroid
            mask[int(cents.x),int(cents.y)] = 1
        except:
            pass
    density = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return density

def getDesnityAdaptive(img):
    '''adaptive gaussian kernel filtering for dense dwellings'''
    pts = []                 # a series of [x, y] points
    contours = measure.find_contours(img, 0.5)
    for i, cont in enumerate(contours):
        try:
            poly = Polygon(cont)      # .simplify(1.0, preserve_topology=False)
            cents = poly.centroid
            pts.append([int(cents.x),int(cents.y)])
        except:
            pass
    density = gaussian_filter_density(img_shape=img.shape, points=pts)
    return density

class CampDataset(Dataset): # for SPADE and PADIM
    def __init__(self, root_path='../data', fold='Minawao_june_2016', phase='train'):
        self.root_path = root_path
        self.fold = fold
        self.phase = phase
        
        print('Processing mask to density!...')
        self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_x = T.Compose([
                                    T.ToTensor(),
                                    T.Lambda(toFloat)
                                    ])
        self.transform_mask = T.Compose([
                                        T.ToTensor(),
                                        T.Lambda(toFloat)
                                        ])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = imread(x)[:,:,:3]
        x = make_normal(x)
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, x.shape[1], x.shape[2]])
        else:
            mask = imread(mask)
            mask = self.transform_mask(mask.astype(float))
        return x, y, mask
                          
    def __len__(self):
        return len(self.x)
                          
    def load_dataset_folder(self):
            
        img_dir = os.path.join(self.root_path, self.fold, self.phase,'images')
        if self.phase == 'test':
            gt_dir = os.path.join(self.root_path, self.fold, self.phase, 'labels')
        else:
            gt_dir = None
            
            
        if self.phase == 'train':
            img_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if f.endswith('.tif')])[:16]
            y = [0] * len(img_list)
            mask = [None] * len(img_list)
            assert len(img_list) == len(y), f'number of x {len(img_list)} and y {len(y)} should be same for {self.phase}'
            return img_list, y, mask
        else:
            img_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if f.endswith('.tif')])[:16]
            y = [1] * len(img_list)
            mask = sorted([os.path.join(gt_dir, f)
                                     for f in os.listdir(gt_dir)
                                     if f.endswith('.tif')])[:16]
            assert len(img_list) == len(y), f'number of x {len(img_list)} and y {len(y)} should be same for {self.phase}'
            
            return img_list, y, mask
        

class CampDatasetUnet(Dataset): # for supervised density based counting
    def __init__(self, imgs, lbls, sigma, adaptive_kernle=False):
        self.images = imgs
        self.labels = lbls
        self.sigma = sigma
        self.adaptive_kenel = adaptive_kernle
        
        self.x, self.y, self.m = self.mask2density()
        

        self.transform_x = T.Compose([
                                    T.ToTensor(),
                                    T.Lambda(toFloat)
                                    ])
        self.transform_mask = T.Compose([
                                        T.ToTensor(),
                                        T.Lambda(toFloat)
                                        ])

    def __getitem__(self, idx):
        x = self.x[idx]
        x = self.transform_x(x)

        y = self.y[idx]
        y = self.transform_mask(y)
        
        m = self.m[idx]
        m = self.transform_mask(m.astype(float))
        return x, y, m

    def __len__(self):
        return len(self.x)
                          
    def mask2density(self):
        images = [make_normal(imread(im)[:,:,:3]) for im in self.images] # offcourse not memory efficient way forlarge datasets
        masks = [imread(im) for im in self.labels]
        if self.adaptive_kenel:
            densities = [getDesnityAdaptive(imread(im)) for im in self.labels]
        else: 
            densities = [getDesnity(imread(im), sigma=self.sigma) for im in self.labels]
        assert len(masks) == len(images) == len(densities), f'Number of masks {len(masks)}, mages {len(images)} and generated densities {len(densities)} are not equal'
        return images, densities, masks
    
    
        
        

