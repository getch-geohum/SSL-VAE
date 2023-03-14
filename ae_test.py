import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from utils import (get_test_dataloader,
                   load_model_parameters,
                   load_ssae,
                   parse_args
                   )

def make_normal(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))


def ssim(a, b, win_size):
    "Structural di-SIMilarity: SSIM"
    #print(f'shape a: {a.shape} \n shape b: {b.shape}')
    a = a.detach().cpu().permute(1, 2, 0).numpy()
    b = b.detach().cpu().permute(1, 2, 0).numpy()

    #b = gaussian_filter(b, sigma=2)

    try:
        score, full = structural_similarity(a, b, #multichannel=True,
                            channel_axis=2, full=True, win_size=win_size)
    except ValueError: # different version of scikit img
        score, full = structural_similarity(a, b, multichannel=True,
            channel_axis=2, full=True, win_size=win_size)
    #return 1 - score, np.median(1 - full, axis=2)  # Return disim = (1 - sim)
    return 1 - score, np.product((1 - full), axis=2)

def get_error_pixel_wise(model, x, loss="rec_loss"):
    x_rec = model(x)
    
    return x_rec

def test(args):
    ''' testing pipeline '''
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )
    print("Pytorch device:", device)

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    checkpoints_dir = f'{args.dst_dir}/torch_checkpoints'  
    feature_dir = f'{args.dst_dir}/torch_features' # to save latent space features
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir, exist_ok=True)
    args.batch_size_test = 1
    auc_file = f"{args.dst_dir}/torch_logs/{args.params_id}_localize_metric_ssmsummary.txt"
    if args.validate:  # mainly to save mvtec normal images without ground truth data
        predictions_dir =f"{args.dst_dir}/predictions_good/"
    else:
        predictions_dir =f"{args.dst_dir}/predictions/"   # need change
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir, exist_ok=True)

    test_dataloader = get_test_dataloader(args,fake_dataset_size=None) 

    # Load model
    model = load_ssae(args)   # loade vae args
    model.to(device)

    try:
        file_name = f"{args.exp}_{args.params_id}.pth"
        model = load_model_parameters(model, file_name, checkpoints_dir, device)
    except FileNotFoundError:
        raise RuntimeError("The model checkpoint does not exist !")

    dissimilarity_func = ssim

    classes = {}

    model.eval()

    aucs_ssim = []
    aucs_mads = []
    aucs_madsc = []
    aucs_comb = []

    pbar = tqdm(test_dataloader)
    for ii,(imgs, gt) in enumerate(pbar): # changed
        imgs = imgs.to(device)
        gt_np = gt[0].cpu().numpy().astype(float)
        #print('Ground truth shape', gt_np.shape)
        #print('Ground truth dtype', gt_np.dtype)
        if args.model in ['ssae','mv_ae']:
            x_rec = model(imgs)
        else:
            x_rec,_ = model(imgs)
            deep_embed = _[0][0].detach().cpu().numpy()
            np.save(feature_dir + f'/{ii}_deep_feat.npy', deep_embed)

        #print(f'Reconstruction shape: {x_rec.shape}')
        #print(f'shape of rec: {x_rec.shape}; shape of imgs: {imgs.shape}')
        score, ssim_map = dissimilarity_func(x_rec[0],imgs[0], 11)  

        ssim_map = ((ssim_map - np.amin(ssim_map)) / (np.amax(ssim_map) - np.amin(ssim_map))) # normalized structural similarity index

        mad = torch.mean(torch.abs(x_rec[0]-imgs[0]), dim=0) # mean absolute deviation mean(|Xrec-X|)  
        mad = mad.detach().cpu().numpy()
        mad = ((mad - np.amin(mad)) / (np.amax(mad) - np.amin(mad)))

        mad_a = torch.mean(torch.norm((x_rec[0]-imgs[0]), 2,dim=0, keepdim=True), dim=0) # mean(||Xrec-X||2
        mad_a = mad_a.detach().cpu().numpy()
        mad_a = ((mad_a - np.amin(mad_a)) / (np.amax(mad_a) - np.amin(mad_a)))


        amaps = ssim_map
        amaps_comb = mad * ssim_map   # combined with structural simmilarity and mean absolute deviation
        amaps = ((amaps - np.amin(amaps)) / (np.amax(amaps) - np.amin(amaps)))
        amaps_comb = ((amaps_comb - np.amin(amaps_comb)) / (np.amax(amaps_comb) - np.amin(amaps_comb)))
        print(f'== Shape of amaps: {amaps.shape}==')

        
        preds_a = amaps.copy()
        preds_b = mad_a.copy() #
        preds_c = mad.copy()
        preds_d = amaps_comb.copy()
        mask = np.zeros(gt_np.shape)

        try:
            auc_a = roc_auc_score(gt_np.astype(np.uint8).flatten(), preds_a.flatten()) # AUC score per image chip
            auc_b = roc_auc_score(gt_np.astype(np.uint8).flatten(), preds_b.flatten())
            auc_c = roc_auc_score(gt_np.astype(np.uint8).flatten(), preds_c.flatten())
            auc_d = roc_auc_score(gt_np.astype(np.uint8).flatten(), preds_d.flatten())
            
            #aucs.append(auc)

            aucs_ssim.append(auc_a)
            aucs_mads.append(auc_c)
            aucs_madsc.append(auc_b)
            aucs_comb.append(auc_d)

        except ValueError:
            pass
                # ROCAUC will not be defined when one class only in y_true

        #m_aucs = np.mean(aucs)   aucs_mads

        m_aucs_ssim_s = np.mean(aucs_ssim)   # m_aucs_ssim
        m_aucs_mads_s = np.mean(aucs_mads)
        m_aucs_madsc_s = np.mean(aucs_madsc)
        m_aucs_comb_s = np.mean(aucs_comb)


        pbar.set_description(f"mean ROCAUC: {m_aucs_ssim_s:.3f}")

        if args.save_preds:
            ori = imgs[0].permute(1, 2, 0).cpu().numpy()
        
            ori = ori[..., :3] # NOTE 4 bands panoptics
        
            rec = x_rec[0].detach().permute(1, 2, 0).cpu().numpy()
            rec = rec[..., :3] # NOTE 4 bands panoptics
            rec = np.dstack((rec[:,:,2], rec[:,:,1],rec[:,:,0]))  # to have clear RGB image
            path_to_save = predictions_dir # f'{args.dst_dir}/predictions/'  # needs reshafling

            img_to_save = Image.fromarray((ori * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '{}_ori.png'.format(str(ii))) 
            img_to_save = Image.fromarray(gt_np[0,:,:].astype(np.uint8))    
            img_to_save.save(path_to_save + '{}_gt.png'.format(str(ii))) 
            img_to_save = Image.fromarray((rec * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '{}_rec.png'.format(str(ii)))    
            #np.save(path_to_save + f'{ii}_final_amap.npy', amaps)  

            cm = plt.get_cmap('jet')
            amaps = cm(amaps)
            img_to_save = Image.fromarray((amaps[..., :3] * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '{}_final_amap.png'.format(str(ii))) 
        
            mads = cm(mad)
            img_to_save = Image.fromarray((mads[..., :3] * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '{}_final_mads.png'.format(str(ii))) 

            mads_a = cm(mad_a)
            img_to_save = Image.fromarray((mads_a[..., :3] * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '{}_final_mads_copy.png'.format(str(ii))) 

            combs = cm(amaps_comb)
            img_to_save = Image.fromarray((combs[..., :3] * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '{}_final_combs.png'.format(str(ii))) 

    #m_auc = np.mean(aucs)


    m_aucs_ssim = np.mean(aucs_ssim)  # m_aucs_ssim
    m_aucs_mads = np.mean(aucs_mads)
    m_aucs_madsc = np.mean(aucs_madsc)
    m_aucs_comb = np.mean(aucs_comb)

    if not args.validate:
        with open(auc_file, 'a+') as txt:
            txt.write('+===================================+\n')
            txt.write(f'SSIM mean AUC: {m_aucs_ssim}\n')
            txt.write(f'MAD mean AUC: {m_aucs_mads}\n')
            txt.write(f'MAD_baur mean AUC: {m_aucs_madsc}\n')
            txt.write(f'SSIM_MAD mean AUC: {m_aucs_comb}\n')
            txt.write('+===================================+\n')

            print("Mean auc on for dataset based on SSIM: ", m_aucs_ssim)
            print("Mean auc on for dataset based on MAD: ", m_aucs_mads)
            print("Mean auc on for dataset based on MAD_c: ", m_aucs_madsc)
            print("Mean auc on for dataset based on MAD_SSIM: ", m_aucs_comb)
    else:
        pass

if __name__ == "__main__":
    args = parse_args()
    test(args)
