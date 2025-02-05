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
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

import camp as camp
'''
Notes:
    save the AUC score to file
    save score maps to file, if not normalized, try to normalize it
    look for mechanisms to implement on all datasets, change the train loader and the main file for example the train loader could take all the test samples
'''

def parse_args():
    parser = argparse.ArgumentParser('SPADE')
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="/SPADE_out")
    parser.add_argument('--root', type=str, default="/VAE")
    return parser.parse_args()


def main():
#     force_train = True
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1, progress=True)
    model.to(device)
    model.eval()

    
    outputs = []
    def hook(module, input, output): # hook to take last layers
        outputs.append(output)
    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.avgpool.register_forward_hook(hook)


    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for fold in list(os.listdir(args.root)):
        os.makedirs(os.path.join(args.save_path, fold), exist_ok=True)
        train_dataset = camp.CampDataset(root_path=args.root, fold=fold, phase='train')
        train_dataloader = DataLoader(train_dataset, batch_size=4, pin_memory=True)
        test_dataset = camp.CampDataset(root_path=args.root, fold=fold, phase='test')
        test_dataloader = DataLoader(test_dataset, batch_size=4, pin_memory=True)

        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])

        train_feature_filepath = os.path.join(args.save_path, fold, 'train_s.pkl')
#         if not force_train: # os.path.exists(train_feature_filepath):
        for (x, y, mask) in train_dataloader: # tqdm(train_dataloader, '| feature extraction | train | %s |' % fold):
            with torch.no_grad():
                pred = model(x.to(device))
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v)
                print(f'All: {k} -->: {v.shape}')
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)
            print(f'summary: {k} -->: {[vv.shape for vv in v]}')
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)
#         else:
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)

        gt_list = []
        gt_mask_list = []
        test_imgs = []

        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % fold):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())

            with torch.no_grad():
                pred = model(x.to(device))
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v)
    
            outputs = []
        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)
        
        print(f"train: {train_outputs['avgpool'].shape} test: {test_outputs['avgpool'].shape}")
        dist_matrix = calc_dist_matrix(torch.flatten(test_outputs['avgpool'], 1),
                                       torch.flatten(train_outputs['avgpool'], 1))
        
        print(f'distmatrix shape: {dist_matrix.shape}')

        # select K nearest neighbor and take average
        topk_values, topk_indexes = torch.topk(dist_matrix, k=args.top_k, dim=1, largest=False)
        print(f'Topk : {topk_values.shape}, inds: {topk_indexes}')
        scores = torch.mean(topk_values, 1).cpu().detach().numpy()
        
        print(f'score shape: {scores.shape}, scores: {scores}')

        # calculate image-level ROC AUC score this is not applicable as we do not have images without anomaly
#         fpr, tpr, _ = roc_curve(gt_list, scores)
#         roc_auc = roc_auc_score(gt_list, scores)
#         total_roc_auc.append(roc_auc)
#         print('%s ROCAUC: %.3f' % (class_name, roc_auc))
#         fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (fold, roc_auc))

        score_map_list = []
        for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]), '| localization | test | %s |' % fold):
            score_maps = []
            for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer

                # construct a gallery of features at all pixel locations of the K nearest neighbors
                topk_feat_map = train_outputs[layer_name][topk_indexes[t_idx]]
                test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]
                feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)
                print(f'topk_feat_map: {topk_feat_map.shape}, test_feat_map: {test_feat_map.shape}, feat_gallery: {feat_gallery.shape}')

                dist_matrix_list = []
                for d_idx in range(feat_gallery.shape[0] // 100):
                    dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                    dist_matrix_list.append(dist_matrix)
                dist_matrix = torch.cat(dist_matrix_list, 0)
                print(f'dist_matrix: {dist_matrix.shape}')
                

                score_map = torch.min(dist_matrix, dim=0)[0]
                print(f'score_map bint {layer_name}: {score_map.shape}')
                score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=256,
                                          mode='bilinear', align_corners=False)
                print(f'score_map aint {layer_name}: {score_map.shape}')
                score_maps.append(score_map)

            score_map = torch.mean(torch.cat(score_maps, 0), dim=0)
            print(f'IM score_map mean: {score_map.shape}')

            score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
            print(f'gauss score_map: {score_map.shape}')
            score_map_list.append(score_map)

        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        print(f'save this shape: {np.concatenate(score_map_list).shape}')  # save this for later count 
        flatten_score_map_list = np.concatenate(score_map_list).ravel()


        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('%s pixel ROCAUC: %.3f' % (fold, per_pixel_rocauc))   # open text file and let it write on the file
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (fold, per_pixel_rocauc))

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # visualize localization result
        visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, fold, vis_num=5)


    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def calc_dist_matrix(x, y):
    """Calculate Euclidean distance matrix with torch.tensor"""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold,
                         save_path, class_name, vis_num=5):

    for t_idx in range(vis_num):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        test_pred = score_map_list[t_idx]
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(test_pred, cmap='gray')
        ax_img[2].title.set_text('Predicted mask')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def denormalization(x):
    mean = 0 # np.array([0.485, 0.456, 0.406])
    std = 1 # np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()
