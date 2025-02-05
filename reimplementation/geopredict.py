'''Run spatialprediction and provides a set of points vvv'''
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from shapely.geometry import Polygon, Point
from skimage import measure
from skimage.io import imread
import rasterio as rio
import torch
import pandas as pd
import geopandas as gpd
import os
import argparse
from tqdm import tqdm
from unet import UNet


def mask2Point(xx): # assuming the less dense dwellings
    contours = measure.find_contours(xx, 0.5)
    xs = []
    ys = []
    for i, cont in enumerate(contours):
        try:
            poly = Polygon(cont)
            cents = poly.centroid
            xs.append(int(cents.x))
            ys.append(int(cents.y))
        except:
            pass
    return xs, ys

def make_normal(img):
    arr = np.clip(((img - np.amin(img)) / ((np.amax(img) - np.amin(img)) + 0.000001)),0,1)
    return arr

def predictSingle(model, arr, p, trsfm):
    density = model(arr.float())
    density = density.detach().cpu().numpy()
    trs = np.percentile(density, p)
    mask = np.squeeze(density>=trs)
    row, col = mask2Point(mask)
    lons, lats = rio.transform.xy(trsfm, row, col) # transform
    if len(lons)>=1:
        points = [Point(lons[i], lats[i]) for i in range(len(lons))]
        s = gpd.GeoSeries(points)
        return s
    else:
        return None


def loadModel(weight):
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(weight))
    print(f'Model weight loaded from {weight}')
    return model

def geoPredict(model, files, save_root, p, device, name):
    file_name = f'{save_root}/{name}.shp'
    epsg = None
    alls = []
    for i, file in tqdm(enumerate(files)):
        with rio.open(file) as oss:
            prf = oss.profile
            rst = oss.read()[:3,:,:]
            if rst.sum()>0:
                xof = oss.profile['transform'][2]
                yof =  oss.profile['transform'][5]
                if i == 0:
                    epsg = int(prf['crs'].to_dict()['init'].split(':')[1])
                rst = make_normal(rst)
                rst = torch.from_numpy(rst).unsqueeze(dim=0).to(device)
                series = predictSingle(model, rst, p=p,trsfm=prf['transform'])
                if series is not None:
                #series = series.translate(xof,yof)
                    alls.append(series)
    alls = pd.concat(alls)
    vals = {'ID':[f'{i}' for i in range(len(alls.geometry))], 'geometry':alls.geometry}
    dfs = gpd.GeoDataFrame(vals).set_crs(epsg=epsg)
    dfs.to_file(file_name)

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("--save_root", type=str, default="/ Minawao_feb_2017") # 
    parser.add_argument('--root', type=str, default="/Minawao_feb_2017/images") # 
    parser.add_argument('--p', type=int, default=96)
    parser.add_argument('--weight', type=str, default='/model/Minawao_feb_2017/weights.pt')
    parser.add_argument('--name',type=str, default='Minawao_feb_2017')
    return parser.parse_args()


if __name__ == '__main__':
    use_cuda=True
    args = parse_args()
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)
    files = [f'{args.root}/{file}' for file in os.listdir(args.root) if '.tif' in file]
    print(f'Total images for prediction: --> {len(files)}')
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = loadModel(args.weight).to(device)
    model.eval()
    geoPredict(model=model, files=files, save_root=args.save_root, p=args.p, device=device,name=args.name)
