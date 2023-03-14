import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from skimage.io import imread
import elasticdeform  # look at https://pypi.org/project/elasticdeform/ for pytorch version too
from skimage.transform import resize
import os
from glob import glob
from random import randrange
import random
from shapely.geometry import Polygon
from rasterio import features   # sometime hard to install
from torch.utils.data import Dataset
from torchvision import transforms
import copy
import matplotlib.pyplot as plt
import cv2


def drawLine(img1, img2, pt1, pt2, color, thickness):
    x1, y1, x2, y2 = *pt1, *pt2
    theta = np.pi - np.arctan2(y1 - y2, x1 - x2)
    dx = int(np.sin(theta) * thickness / 2)
    dy = int(np.cos(theta) * thickness / 2)
    pts = [[x1 + dx, y1 + dy],
           [x1 - dx, y1 - dy],
            [x2 - dx, y2 - dy],
            [x2 + dx, y2 + dy]]
    cv2.fillPoly(img1, [np.array(pts)], color)
    cv2.fillPoly(img2, [np.array(pts)], color)


def addLineNoise(img, color=(255, 255, 255), thickness=10, n_lines=2, alpha=0.3):
    open_img = np.full((512, 512, 3), 0, np.uint8)
    rst_img = img.copy()
    for i in range(n_lines):
        pt1 = np.random.randint(0, 512, 2)
        pt2 = np.random.randint(0, 512, 2)
        drawLine(img1=rst_img,img2=open_img, pt1=pt1, pt2=pt2, color=color, thickness=thickness)
    mask = np.where(open_img==255, 255, 255)
    if alpha is not None:
        image_new = cv2.addWeighted(img, alpha, rst_img, 1 - alpha, 0)
        return image_new, mask
    else:
        return rst_img, mask


def drawCeircle(image, color, alpha=0.1, xc=50, yc=50, r=70):
    overlay = image.copy()
    im2 = cv2.circle(overlay, (xc, yc), r, (0, 0, 0), -1)
    mask = cv2.circle(np.zeros(overlay.shape), (xc, yc,), r, (1, 1, 1), -1)
    image_new = cv2.addWeighted(image, alpha, im2, 1 - alpha, 0)
    return (image_new, mask)


def addCeircleNoise(image, color=(0, 0, 0)):
    r = np.random.randint(30, 70) # radius of a ceircle
    xc = np.random.randint(r+1, image.shape[0]-r-1)
    yc = np.random.randint(r+1, image.shape[1]-r-1)
    alpha = np.random.choice([0.3, 0.4, 0.5, 0.6])
    im, mask = drawCeircle(image, color=color, alpha=alpha, xc=xc, yc=yc, r=r)
    return im, mask


# some functions are taken from https://itecnote.com/tecnote/python-create-random-shape-contour-using-matplotlib/

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)

def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a

def computemask(size=200, points=10, rad = 0.5, edgy = 0.5, plot=False):
            xs, ys = [randrange(0, size) for i in range(points)], [randrange(0,size) for i in range(points)]
            points = list(zip(xs, ys))
            points = np.array(points)
            x,y, _ = get_bezier_curve(points,rad=rad, edgy=edgy)
            polyn = Polygon([(x[i], y[i]) for i in range(len(y))])
            img_msk = features.rasterize([polyn], out_shape=(size, size))
            if plot:
                plt.plot(x, y)
                plt.show()
                plt.imshow(img_msk)
                plt.show()
            patch_mask = np.dstack([img_msk]*3)
            return patch_mask
        
def manipulateImage(image, mask, size=200):
    sigma = randrange(9, 16)
    xi = randrange(0, image.shape[0]-size)
    yi = randrange(0, image.shape[1]-size)
    
    new_mask = np.zeros(image.shape)
    new_mask[xi:xi+size, yi:yi+size,:] = mask
    
    sample = image[xi:xi+size, yi:yi+size,:]
    X_deformed = elasticdeform.deform_random_grid(sample, axis=(0, 1), sigma=sigma, points=5, order=1)
    outs = mask*X_deformed
    final = np.where(outs == 0,sample, outs)
    
    final_img = image.copy()
    final_img[xi:xi+size, yi:yi+size,:] = final
    return final_img, new_mask

def toFloat(x):
    return x.float()

class MvtechTrainDataset(Dataset):
    def __init__(self, root, texture='carpet', with_prob=True):
        self.root = root
        self.with_prob = with_prob
        if texture == 'all':
            self.img_dir = []
            for fold in ['carpet', 'grid', 'leather', 'tile', 'wood']:
                files = sorted(glob(f'{self.root}/{fold}/train/good/*.png'))
                self.img_dir+=files
        else:
            self.train_path = f'{root}/{texture}/train/good'
            self.img_dir = sorted(glob(f'{self.train_path}/*.png'))
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
             transforms.Lambda(toFloat)
            #transforms.Lambda(lambda img: img.float())
        ])
        
        print('Length of dataset: ', len(self.img_dir))

    def make_normal(self, fil):
        return (fil - np.amin(fil)) / (np.amax(fil) - np.amin(fil))
    
    def image2Array(self, file):
        image = imread(file)
        image = resize(image, (512, 512, 3))
        normal_image = self.make_normal(image)
        mask = computemask(plot=False)
        changed_image, changed_mask = manipulateImage(normal_image, mask=mask)
        normal_mask = np.where(changed_mask==0,1,0)
        assert normal_mask.shape == changed_mask.shape, f'Mask shapes {normal_mask.shape} and {changed_mask.shape} are not equal'
        return normal_image, changed_image, changed_mask, normal_mask
        
    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        file = self.img_dir[index]
        nrm, mdf, mskm, mskn = self.image2Array(file=file)
        if self.with_prob:
            p = np.where(mskm[:,:,0] == 1, np.sum(mskm[:,:,0])/(512*512), 1-(np.sum(mskm[:,:,0])/(512*512))) 
            p = np.dstack([p]*3) 
            return self.transform(nrm), self.transform(mdf), self.transform(mskm), self.transform(mskn), self.transform(p)
        else:
            return self.transform(nrm), self.transform(mdf), self.transform(mskm), self.transform(mskn)

class MvtechTestDataset(Dataset):
    def __init__(self, root, texture='carpet', fake_dataset_size=None,validate=False):
        self.root = root
        self.validate = validate
        self.test_path = root + '/{}/test'
        self.img_dir = []
        if texture == 'all':
            for fold in ['carpet', 'grid', 'leather', 'tile', 'wood']:
                test_path = self.test_path.format(fold)
                if self.validate:
                    sub_files = sorted(glob(f'{test_path}/good/*.png'))
                    self.img_dir += sub_files
                else:
                    for s_fold in os.listdir(test_path):
                        if s_fold != 'good':
                            sub_files = sorted(glob(f'{test_path}/{s_fold}/*.png'))
                            self.img_dir += sub_files
        else:
            test_path = self.test_path.format(texture)
            print(test_path)
            if self.validate:
                sub_files = sorted(glob(f'{test_path}/good/*.png'))
                self.img_dir += sub_files
            else:
                for s_fold in os.listdir(test_path):
                    if s_fold != 'good':
                        sub_files = sorted(glob(f'{test_path}/{s_fold}/*.png'))
                        self.img_dir += sub_files
        

        self.lbl_dir =  [img.replace('test', 'ground_truth').replace('.png', '_mask.png') for img in self.img_dir]
        
        if fake_dataset_size is not None:
            inds = list(range(len(self.img_dir)))
            sample = random.sample(inds, fake_dataset_size)
            self.img_dir = [self.img_dir[ind] for ind in sample]
            self.lbl_dir = [self.lbl_dir[ind] for ind in sample]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
             transforms.Lambda(toFloat)
            #transforms.Lambda(lambda img: img.float())
            ])

    def make_normal(self, fil):
        return (fil - np.amin(fil)) / (np.amax(fil) - np.amin(fil))
    
    def image2Array(self, image, normalize=False):
        image = imread(image)
        if not normalize:
            return resize(image, (512, 512)).astype(float)
        else:
            image = resize(image, (512, 512, 3))
            return self.make_normal(image).astype(float)
        
    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        img = self.image2Array(self.img_dir[index], normalize=True)
        if self.validate:
            msk = np.ones(img.shape[:2], float)
        else:
            msk = self.image2Array(self.lbl_dir[index], normalize=False)
        return self.transform(img), self.transform(msk)
    
    
