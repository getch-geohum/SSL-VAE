''' final '''
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
import camp as camp             # dataset




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
    parser = {}
    parser['save_path'] = "/Unet_output"
    parser['root'] = "/data"
    parser['sigma'] = 5
    parser['lr']=0.0001
    parser['epochs']=50
    parser['batch_size']=4

    return parser
def main():

    args = parse_args()
    load_weight = True

    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    folds = list(os.listdir(args.get('root'))) # files to leave on out

    for fold in folds:
        print(f'Building model for --> {fold}')
        model = UNet(n_channels=3, n_classes=1)
        model.to(device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=args.get('lr'), momentum=0.9)  # change this
        criterion = RMSELoss()


        print(f'Loading training data to test {fold}')
        TRAIN = [[],[]] # containes all training files from all folders
        TEST = [[],[]]

        for fold_ in list(os.listdir(args.get('root'))): # loading data by leaving one
            if fold_ == fold:
                path = f'{args.get("root")}/{fold}'
                test_files, train_files = systematic_split(root_dir=path,
                                       split_ratio= [0.7, 0.3],
                                       val_test_split=False)
                TEST[0].extend(test_files[0])
                TEST[0].extend(train_files[0])
                TEST[1].extend(test_files[1])
                TEST[1].extend(train_files[1])

            else:
                path = f'{args.get("root")}/{fold_}'
                test_files, train_files = systematic_split(root_dir=path,
                                       split_ratio= [0.7, 0.3],
                                       val_test_split=False)
                TRAIN[0].extend(train_files[0])
                TRAIN[0].extend(test_files[0])
                TRAIN[1].extend(train_files[1])
                TRAIN[1].extend(test_files[1])

        print(f'Length of training: {len(TRAIN[0])}')
        print(f'Length of testing: {len(TEST[0])}')

        # train_dataset = camp.CampDatasetUnet(imgs=TRAIN[0], lbls=TRAIN[1], sigma=args.get('sigma'))
        # train_dataloader = DataLoader(train_dataset, batch_size=args.get('batch_size'), pin_memory=True, shuffle=True)


        # for j in range(1, args.epochs+1):
        #     for (x,y,_) in train_dataloader:
        #         optimizer.zero_grad()
        #         out = model(x.to(device))
        #         loss = criterion(out, y.to(device))
        #         loss.backward()
        #         optimizer.step()
        #         print(f'|{fold}|epoch {j}|step-loss {loss.item()}|')
        # os.makedirs(os.path.join(args.save_path, f'model/{fold}'), exist_ok=True)
        PATH = os.path.join(args.get('save_path'), f'model/{fold}/weights.pt')
        # torch.save(model.state_dict(), PATH)
        
        if load_weight:
            model.load_state_dict(torch.load(PATH))
            print('Model weight loaded!')
        model.eval()
        
        fig, fig_pixel_rocauc = plt.subplots(1,1, figsize=(10, 10)) # this is to plot the curve
        total_pixel_roc_auc = []

        # for fold in list(TEST.keys()):
        print('started testing!.....')
        os.makedirs(args.get('save_path') + f'/report/{fold}', exist_ok=True)
        os.makedirs(args.get('save_path') + f'/preds/{fold}', exist_ok=True)

        test_dataset = camp.CampDatasetUnet(imgs=TEST[0], lbls=TEST[1], sigma=args.get('sigma'))
        test_dataloader = DataLoader(test_dataset, batch_size=args.get('batch_size'), pin_memory=True, shuffle=True)

        results = []
        referes = []
        images = []
        gt_mask_list = []

        with torch.no_grad():
            for (x,y,yy) in tqdm(test_dataloader, f'|{fold}|feature extraction | test |'):
                out = model(x.to(device))
                results.append(out.squeeze(1))
                referes.append(y.squeeze(1).to(device))
                images.append(x)
                gt_mask_list.append(yy)

        results = torch.cat(results, 0)
        referes = torch.cat(referes, 0)
        images = torch.cat(images, 0)
        gt_mask = torch.cat(gt_mask_list,0)

        print(f'Resl --> {results.shape} Refs --> {referes.shape} images --> {images.shape}')

        ref_count = torch.sum(referes.squeeze(), dim=(1,2))
        pred_count = torch.sum(results.squeeze(), dim=(1,2))
        
        
        count = torch.mean(torch.abs(pred_count-ref_count)) # Mean Absolute Error
        print(f'count MAE: --> {count}')


        # == For localization
        max_score = results.max()           # Normalization
        min_score = results.min()
        scores = (results - min_score) / (max_score - min_score)

        scores = scores.cpu().numpy()
        gt_mask = np.squeeze(gt_mask.numpy())    # get optimal threshold
        print(f'Mask shapes: {scores.shape}, {gt_mask.shape}')
        assert scores.shape == gt_mask.shape, f'scores {scores.shape} and gtmaks {gt_mask.shape} shapes are not the same'
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]


        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())  # calculate per-pixel level ROCAUC
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))
        
        with open(f'{args.get("save_path")}/report/{fold}/rocauc_count.txt', 'a') as txt:
            txt.write(f'roc_auc: {per_pixel_rocauc} \n')
            txt.write(f'MAE: {count}\n')

        #fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (fold, per_pixel_rocauc))
        #save_dir = args.save_path + f'/report/{fold}'

        ref_cl = ref_count.tolist()
        pred_cl = pred_count.tolist()
        dicts = {'ref':ref_cl, 'pred':pred_cl}
        df = pd.DataFrame.from_dict(dicts)
        df.to_csv(f"{args.get('save_path')}/report/{fold}/count_rep.csv")

        del model

        print('writting results!!!')
        for idx in range(10):
            fig, ax = plt.subplots(1, 3, figsize=(30, 10), sharex = True, sharey=True)
            img = denormalization(images[idx].numpy())
            # ref = gt_mask[idx] #.cpu().numpy()
            ref = referes[idx].squeeze().cpu().numpy() # raw predictions
            pred = results[idx].squeeze().cpu().numpy() # this is normalized
            ax[0].imshow(img)
            ax[1].imshow(ref)
            ax[2].imshow(pred)
            ax[0].set_title('input')
            ax[1].set_title(f'ref:{ref.sum()}')
            ax[2].set_title(f'pred:{pred.sum()}')
            for i in range(3):
                ax[i].set_xticks([])
            plt.savefig(f'{args.get("save_path")}/preds/{fold}/image_{idx}.png', dpi=100, bbox_inches='tight')
            plt.close('all')
            
def denormalization(x):
    mean = 0 # np.array([0.485, 0.456, 0.406]) # changed based on the input image implementation
    std = 1 # np.array([0.229, 0.224, 0.225])  # changed based on the input image implementation
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

main()
