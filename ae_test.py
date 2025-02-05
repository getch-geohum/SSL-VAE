import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity
from scipy.ndimage import gaussian_filter # (input, sigma
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
matplotlib.use('Agg')

from liu_processing import liu_anomaly_map

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

def test(args,return_loss=False):
    smooth = False # for Bauer et al implementation, smoothing of anomaly scores
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
    feature_dir = f'{args.dst_dir}/torch_features/{args.data[0]}' # to save latent space features
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir, exist_ok=True)
    args.batch_size_test = 1
    auc_file = f"{args.dst_dir}/torch_logs/{args.params_id}_localize_metric_ssmsummary__b.txt"
    if args.validate:  # mainly to save mvtec normal images without ground truth data
        predictions_dir = f"{args.dst_dir}/predictions_good/"
    else:
        predictions_dir =f"{args.dst_dir}/predictions/{args.data[0]}"   # need change
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir, exist_ok=True)

    test_dataloader = get_test_dataloader(args,fake_dataset_size=None) 
    print('length of test data loader-->',len(test_dataloader))
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
    
    # kldfile = f"{args.dst_dir}/torch_logs/{args.params_id}_{args.data[0]}_kld_loss.txt"
    # recfile = f"{args.dst_dir}/torch_logs/{args.params_id}_{args.data[0]}_rec_loss.txt"
    recfile = f"{args.dst_dir}/torch_logs/{args.params_id}_{args.data[0]}_rec_ts_noav_dv_loss.txt"
    
    # loss_file_kld  = open(kldfile, 'a+')
    # loss_file_kld.write('+===================================+\n')
    
    loss_file_rec  = open(recfile, 'a+')
    loss_file_rec.write('+===================================+\n')
    

    pbar = tqdm(test_dataloader)
    for ii,(imgs, gt,_) in enumerate(pbar): # changed
        #print(f'SAPES OF IMAGES: {imgs.shape}, SHAPE of ground truth: {gt.shape}')
        imgs = imgs.to(device)
        gt = gt.to(device)
        gt_bd = torch.where(gt == 0, 1, 0)
        
        gt_np = gt[0].cpu().numpy().astype(float)
        #print('Ground truth shape', gt_np.shape)
        #print('Ground truth dtype', gt_np.dtype)
        if args.model in ['ssae','mv_ae','vae','ssvae']:
            x_rec,_ = model(imgs)
            if return_loss:
                # kld = torch.mean(model.kld())
                # print(f'kld: {kld}')
                # loss_file_kld.write(f'{kld.item()}\n')
                print(f'Rec shape: {x_rec.shape}, {x_rec.dtype} --> Ori shape: {imgs.shape}, {imgs.dtype}')
                
                rec_loss_a = torch.mean(model.xent_continuous_ber(x_rec*gt, imgs*gt))
                rec_loss_b = torch.mean(model.xent_continuous_ber(x_rec*gt_bd, imgs*gt_bd))
                    # print(f'rec_loss_p: {rec_loss_a.item()}, rec_loss_n: {rec_loss_b.item()}')
                loss_file_rec.write(f'{rec_loss_a.item()},{rec_loss_b.item()}\n')
                
                # rec_loss = torch.mean(model.xent_continuous_ber(x_rec, imgs))
                # print(f'rec: {rec_loss}')
                # loss_file_rec.write(f'{rec_loss.item()}\n')
        elif args.model in ['liu_vae']:
            anomaly_maps = []
            for c in args.conv_layers:
                M, x_rec = liu_anomaly_map(c, model, imgs, device)
                anomaly_maps.append(M)
            amaps = np.squeeze(anomaly_maps[0].cpu().numpy())
            mask_ = (amaps > 0.1).astype(np.int8)
            amaps = ((amaps - np.amin(amaps)) / (np.amax(amaps) - np.amin(amaps))) # for localization 
        else:
            #x_rec_ = model.encoder(imgs)
            #print(x_rec_[0].shape)
            if args.model in ['dis_ssvae']:
                # if return_loss:
                    # x_rec, _ = model(imgs)
                    # kld = torch.mean(model.kld())
                    # loss_file_kld.write(f'{kld.item()}\n')
                    
#                     Mm = gt   # assuming the ground truth value are forground masks telling anomaly values during SSL-VAE
#                     Mn = torch.where(gt==0.0, 1.0, 0.0)
#                     base = 256 * 256
#                     lambda_ = 0.9 # start with lambda = 1, maybe modify it later
#                     w_n = (torch.sum(Mn, dim=(1, 2)) / base) 
#                     w_m = (torch.sum(Mm, dim=(1, 2)) / base) 

#                     rec_normal = torch.mean(
#                         model.xent_continuous_ber(x_rec, imgs, gamma=Mn) + torch.log(w_n * torch.sum(Mn, dim=(1,2)))
#                      )
#                     rec_modified = torch.mean(
#                         model.xent_continuous_ber(x_rec, imgs, gamma=Mm) + torch.log(w_m* torch.sum(Mm,dim=(1,2)))
#                      )
#                     rec_loss = (lambda_ * rec_normal + (1 - lambda_) * rec_modified)
                    
                    # rec_loss = torch.mean(model.xent_continuous_ber_vae(x_rec, imgs))
                    # print(f'rec:_loss {rec_loss}')
                    # loss_file_rec.write(f'{rec_loss.item()}\n')
                mu_train_modified = model.encoder(imgs)[0]
                b, c, h, w = mu_train_modified.shape
                    # print('-->Mode shape: ', mu_train_modified.shape)
                #print(c.shape, c.shape,h.shape, w.shape)   # the note for explainin the shape of the output space

                #mu_train_modified[:] = torch.mean(mu_train_modified[:], axis=1, keepdims=True)
                #mu_train_modified = mu_train_modified.reshape((b, -1, h, w))
                print('before averaging: ', mu_train_modified.shape)
                mu_train_modified[:],_ = torch.median(mu_train_modified[:], axis=1, keepdims=True)
                
                mu_train_modified = mu_train_modified.reshape((b, -1, h, w))

                print("final shape: ", mu_train_modified.shape)
                x_rec = model.mean_from_lambda(model.decoder(mu_train_modified))
                    
                    # rec_loss_a = torch.mean(model.xent_continuous_ber_vae(x_rec*gt, imgs*gt))
                    # rec_loss_b = torch.mean(model.xent_continuous_ber_vae(x_rec*gt_bd, imgs*gt_bd))
                    # print(f'rec_loss_p: {rec_loss_a.item()}, rec_loss_n: {rec_loss_b.item()}')
                    # loss_file_rec.write(f'{rec_loss_a.item()},{rec_loss_b.item()}\n')
                    
#             else:
#                 x_rec,_ = model(imgs)
#                 deep_embed = _[0][0].detach().cpu().numpy()
#                 np.save(feature_dir + f'/{ii}_deep_feat.npy', deep_embed)

#         #print(f'Reconstruction shape: {x_rec.shape}')
#         #print(f'shape of rec: {x_rec.shape}; shape of imgs: {imgs.shape}')
        if args.model not in ['liu_vae']:
            score, ssim_map = dissimilarity_func(x_rec[0],imgs[0][:4],11)

            ssim_map = ((ssim_map - np.amin(ssim_map)) / (np.amax(ssim_map) - np.amin(ssim_map))) # normalized 

            mad = torch.mean(torch.abs(x_rec[0]-imgs[0][:4]), dim=0) # mean absolute deviation mean(|Xrec-X|)  
            mad = mad.detach().cpu().numpy()
            mad = ((mad - np.amin(mad)) / (np.amax(mad) - np.amin(mad)))

            mad_a = torch.mean(torch.norm((x_rec[0]-imgs[0][:4]), 2,dim=0, keepdim=True), dim=0) # mean(||Xrec-X||2
            mad_a = mad_a.detach().cpu().numpy()
            
            if args.model == 'ssae' and smooth:
                mad_a = gaussian_filter(mad_a, sigma=4)

            mad_a = ((mad_a - np.amin(mad_a)) / (np.amax(mad_a) - np.amin(mad_a)))

            amaps = ssim_map
            amaps_comb = mad * ssim_map   # combined with structural simmilarity and mean absolute deviation
            amaps = ((amaps - np.amin(amaps)) / (np.amax(amaps) - np.amin(amaps)))
            amaps_comb = ((amaps_comb - np.amin(amaps_comb)) / (np.amax(amaps_comb) - np.amin(amaps_comb)))
#         #print(f'== Shape of amaps: {amaps.shape}==')

        

        preds_a = amaps.copy()
        mask = np.zeros(gt_np.shape)
        if args.model not in ['liu_vae']:
            preds_b = mad_a.copy() #
            preds_c = mad.copy()
            preds_d = amaps_comb.copy()

        try:
            auc_a = roc_auc_score(gt_np.astype(np.uint8).flatten(), preds_a.flatten()) # AUC score per image chip
            aucs_ssim.append(auc_a)
            if args.model not in ['liu_vae']:
                auc_b = roc_auc_score(gt_np.astype(np.uint8).flatten(), preds_b.flatten())
                auc_c = roc_auc_score(gt_np.astype(np.uint8).flatten(), preds_c.flatten())
                auc_d = roc_auc_score(gt_np.astype(np.uint8).flatten(), preds_d.flatten())
            
            #aucs.append(auc)

    
                aucs_mads.append(auc_c)
                aucs_madsc.append(auc_b)
                aucs_comb.append(auc_d)

        except ValueError:
            pass
#                 # ROCAUC will not be defined when one class only in y_true

        #m_aucs = np.mean(aucs)   aucs_mads

        m_aucs_ssim_s = np.mean(aucs_ssim)   # m_aucs_ssim
        if args.model not in ['liu_vae']:
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
            path_to_save = predictions_dir # f'{args.dst_dir}/predictions/{args.data[0]}/'  # needs reshafling

            img_to_save = Image.fromarray((ori * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '/{}_ori.png'.format(str(ii))) 
            img_to_save = Image.fromarray(gt_np[0,:,:].astype(np.uint8))    
            img_to_save.save(path_to_save + '/{}_gt.png'.format(str(ii))) 
            img_to_save = Image.fromarray((rec * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '/{}_rec.png'.format(str(ii)))    
            #np.save(path_to_save + f'{ii}_final_amap.npy', amaps)  

            cm = plt.get_cmap('jet')
            amaps = cm(amaps)
            img_to_save = Image.fromarray((amaps[..., :3] * 255).astype(np.uint8))
            img_to_save.save(path_to_save + '/{}_final_amap.png'.format(str(ii)))

            if args.model not in ['liu_vae']:
        
                mads = cm(mad)
                img_to_save = Image.fromarray((mads[..., :3] * 255).astype(np.uint8))
                img_to_save.save(path_to_save + '/{}_final_mads.png'.format(str(ii))) 

                mads_a = cm(mad_a)
                img_to_save = Image.fromarray((mads_a[..., :3] * 255).astype(np.uint8))
                img_to_save.save(path_to_save + '/{}_final_mads_copy.png'.format(str(ii))) 

                combs = cm(amaps_comb)
                img_to_save = Image.fromarray((combs[..., :3] * 255).astype(np.uint8))
                img_to_save.save(path_to_save + '/{}_final_combs.png'.format(str(ii)))
            else:
                np.save(path_to_save + '/{}_final_combs.npy'.format(str(ii)), mask_)

    #m_auc = np.mean(aucs)


    m_aucs_ssim = np.mean(aucs_ssim)  # m_aucs_ssim
    if args.model not in ['liu_vae']:
        m_aucs_mads = np.mean(aucs_mads)
        m_aucs_madsc = np.mean(aucs_madsc)
        m_aucs_comb = np.mean(aucs_comb)

    if not args.validate:
        with open(auc_file, 'a+') as txt:
            txt.write('+===================================+\n')
            txt.write(f'SSIM mean AUC: {m_aucs_ssim}\n')
            if args.model not in ['liu_vae']:
                txt.write(f'MAD mean AUC: {m_aucs_mads}\n')
                txt.write(f'MAD_baur mean AUC: {m_aucs_madsc}\n')
                txt.write(f'SSIM_MAD mean AUC: {m_aucs_comb}\n')
                txt.write('+===================================+\n')

            print("Mean auc on for dataset based on SSIM: ", m_aucs_ssim)
            if args.model not in ['liu_vae']:
                print("Mean auc on for dataset based on MAD: ", m_aucs_mads)
                print("Mean auc on for dataset based on MAD_c: ", m_aucs_madsc)
                print("Mean auc on for dataset based on MAD_SSIM: ", m_aucs_comb)
    else:
        pass

if __name__ == "__main__":
    args = parse_args()
    test(args)
