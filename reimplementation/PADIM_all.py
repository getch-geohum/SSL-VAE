'''
This code is forked from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/tree/main
and amended for custom dataset for high resolution satellite imagery
'''
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
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import camp as camp


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("--save_path", type=str, default="/PADIM_out")
    parser.add_argument('--root', type=str, default="/VAE")
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


def main():

    args = parse_args()
    reload = False
    save_result = False
    
    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(weights='IMAGENET1K_V1', progress=True) # Wide_ResNet50_2_Weights.IMAGENET1K_V1
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    outputs = [] # hook model outputs

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []
    

    all_dataset = []   # normal image datasets obtained from all camps 
    for fold in list(os.listdir(args.root)):
        train_dataset = camp.CampDataset(root_path=args.root, fold=fold, phase='train')
        all_dataset.append(train_dataset)
    all_dataset = torch.utils.data.ConcatDataset(all_datasets)
    train_dataloader = DataLoader(all_dataset, batch_size=4, pin_memory=True, shuffle=True)
    
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    

    os.makedirs(os.path.join(args.save_path, 'model'), exist_ok=True)
    train_feature_filepath = os.path.join(args.save_path, 'model', 'train_s.pkl')

    for (x, _, _) in tqdm(train_dataloader, '| feature extraction | train | all dataset |'):
        with torch.no_grad():
            _ = model(x.to(device))
        for k, v in zip(train_outputs.keys(), outputs):
            train_outputs[k].append(v.cpu().detach())
        outputs = []
    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)


    embedding_vectors = train_outputs['layer1'] # Embedding concat
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

    print(f'embedding_vectors: --> {embedding_vectors.shape}')
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    B, C, H, W = embedding_vectors.size()
    print(f'B, C, H, W: --> {(B, C, H, W)}')
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)
    for i in range(H * W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I

    print(f'mean, cov : --> {(mean.shape, cov.shape)}')
    train_outputs = [mean, cov]   # save learned parametric distributions
    if save_result:
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)
            print('Saved learned feature distribution to: %s' % train_feature_filepath)
            
    if reload:
        del train_outputs # for memory efficiency 
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)
            print('learned feature distribution loaded from: %s' % train_feature_filepath)
    
    for fold in os.listdir(args.save_dir): # test for each dataset
        test_dataset = camp.CampDataset(root_path=args.root, fold=fold, phase='test')
        test_dataloader = DataLoader(test_dataset, batch_size=4, pin_memory=True)
        
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % fold):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            
            with torch.no_grad():
                _ = model(x.to(device))
                
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
                
            outputs = []
            
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        
        embedding_vectors = test_outputs['layer1']  # Embedding concat
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])
            
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
        
        B, C, H, W = embedding_vectors.size()      # calculate distance matrix
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
#             if i in [0]:
#                 print(f'Train outputs mean: {train_outputs[0].shape}')
            mean_ = train_outputs[0][:, i]
            conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean_, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        dist_list = torch.tensor(dist_list)  # upsample
        score_map = F.interpolate(dist_list.unsqueeze(1), size=256, mode='bilinear', align_corners=False).squeeze().numpy()
        
        
        for i in range(score_map.shape[0]):  # apply gaussian smoothing on the score map
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
        max_score = score_map.max()           # Normalization
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        
        
        gt_mask = np.asarray(gt_mask_list)    # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        
        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())  # calculate per-pixel level ROCAUC
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (fold, per_pixel_rocauc))
        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, fold)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()

def denormalization(x):
    mean = 0    # np.array([0.485, 0.456, 0.406]) # changed based on the input image implementation
    std = 1     # np.array([0.229, 0.224, 0.225])  # changed based on the input image implementation
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)    
    return x

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()
