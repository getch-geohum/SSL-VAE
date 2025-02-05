from glob import glob    
import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unet import UNet, RMSELoss # implemented model
import camp as camp  # dataset


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def systematic_split(root_dir,split_ratio= [0.7, 0.2], val_test_split=True):
    imgs = sorted(glob(root_dir + '/test/images' + '/*.tif'))
    N = len(imgs)
    
    tr = round(N*split_ratio[0])
    tr_ts = round(N*split_ratio[1])
    vest_ind = list(range(0, N, round(N/tr_ts)))

    val_ts_im = [imgs[i] for i in vest_ind]
    val_ts_lb = [am.replace('images', 'labels') for am in val_ts_im] # labels
    
    tr_im = list(set(imgs).symmetric_difference(set(val_ts_im)))
    tr_lb = [am.replace('images', 'labels') for am in tr_im] # labels
    
    if val_test_split:
        val = [val_ts_im[i] for i in range(0, len(val_ts_im), 2)]
        ts = [val_ts_im[i] for i in range(1, len(val_ts_im), 2)]
        
        val_lb = [am.replace('images', 'labels') for am in val] # labels
        ts_lb = [am.replace('images', 'labels') for am in ts] # labels
        
        return (tr_im, tr_lb), (val, val_lb), (ts, ts_lb)
    else:
        return (tr_im, tr_lb), (val_ts_im, val_ts_lb)


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("--save_path", type=str, default="UNet_out")
    parser.add_argument('--root', type=str, default="/VAE")
    parser.add_argument('--sigma', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    return parser.parse_args()


def main():

    args = parse_args()
    
    model = UNet(n_channels=3, n_classes=1) 
    model.to(device)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)  # change this
    criterion = RMSELoss()
    
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    

    for fold in list(os.listdir(args.root)):
        os.makedirs(os.path.join(args.save_path, fold), exist_ok=True)
        path = f'{args.root}/{fold}'
        test_files, train_files = systematic_split(root_dir=path,
                                       split_ratio= [0.7, 0.3],
                                       val_test_split=False)
        
        train_dataset = camp.CampDatasetUnet(imgs=train_files[0][:32], lbls=train_files[1][:32], sigma=args.sigma)
        train_dataloader = DataLoader(train_dataset, batch_size=4, pin_memory=True)
        test_dataset = camp.CampDatasetUnet(imgs=test_files[0][:32], lbls=test_files[1][:32], sigma=args.sigma)
        test_dataloader = DataLoader(test_dataset, batch_size=4, pin_memory=True)
        
        for j in range(1, args.epochs+1):
            for (x,y) in train_dataloader:
                optimizer.zero_grad()
                out = model(x.to(device))
                loss = criterion(out, y.to(device))
                loss.backward()
                optimizer.step()
                print(f'|{fold}|epoch {j}|step-loss {loss.item()}|')
        os.makedirs(os.path.join(args.save_path, fold, 'model'), exist_ok=True)
        PATH = os.path.join(args.save_path, fold, 'model/weights.pt')
        torch.save(model.state_dict(), PATH)
    
        results = []
        referes = []
        images = []
        model.eval()
        
        with torch.no_grad():
            for (x,y) in tqdm(test_dataloader, '| feature extraction | train | %s |' % fold):
                out = model(x.to(device))
                results.append(out.squeeze(1))
                referes.append(y.squeeze(1).to(device))
                images.append(x)
        
        results = torch.cat(results, 0)
        referes = torch.cat(referes, 0)
        images = torch.cat(images, 0)
        
        print(f'Resl --> {results.shape} Refs --> {referes.shape} images --> {images.shape}')
        
        ref_count = torch.sum(referes, dim=(1,2))
        pred_count = torch.sum(results, dim=(1,2))
        
        count = torch.mean(torch.abs(pred_count-ref_count)) # Mean Absolute Error
        print(f'count MAE: --> {count}')
        
        ref_cl = ref_count.tolist()
        pred_cl = pred_count.tolist()
        dicts = {'ref':ref_cl, 'pred':pred_cl}
        df = pd.DataFrame.from_dict(dicts)
        df.to_csv(f"{args.save_path}/{fold}/results/count_rep.csv")
        
        
        os.makedirs(os.path.join(args.save_path, fold, 'results'), exist_ok=True)
        with open(f"{args.save_path}/{fold}/results/REPORT.txt", "a") as ff:
            ff.write(f"{fold} {count}\n")
        
        for i in range(results.shape[0]):
            fig, ax = plt.subplots(1, 3, figsize=(30, 10), sharex = True, sharey=True)
            img = denormalization(images[i].numpy())
            ref = referes[i].cpu().numpy()
            pred = results[i].cpu().numpy()
            ax[0].imshow(img)
            ax[1].imshow(ref)
            ax[2].imshow(pred)
            ax[0].set_title('input')
            ax[1].set_title(f'ref:{ref.sum()}')
            ax[2].set_title(f'pred:{pred.sum()}')
            for i in range(3):
                ax[i].set_xticks([])
            plt.savefig(f'{args.save_path}/{fold}/results/image_{i}.png', dpi=100, bbox_inches='tight')
            plt.close('all')
            
def denormalization(x):
    mean = 0 # np.array([0.485, 0.456, 0.406]) # changed based on the input image implementation
    std = 1 # np.array([0.229, 0.224, 0.225])  # changed based on the input image implementation
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)    
    return x



if __name__ == '__main__':
    main()
