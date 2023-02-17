import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import DataLoader
from datasets import *
from AE import SSAE
from ssvae import SSVAE
from c_ssvae import SS_CVAE
import time
import argparse
import matplotlib
matplotlib.use('Agg')  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help='ssae or ssvae', type=str, default="ssae", required=True)
    parser.add_argument("--params_id", default=100)
    parser.add_argument("--img_size", help="The saptial size of the input image", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--batch_size_test", default=8, type=int)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--latent_img_size", help="Spatial size of encoded image", default=8, type=int)
    parser.add_argument("--fake_dataset_size", help="samples for runnung test", default=30, type=int)
    parser.add_argument("--z_dim", help="Number of final encoder channels", default=512, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lamda", help="To balance loss from modifed image part", default=0.5, type=float)
    parser.add_argument("--exp", help="The expession text for the file saving", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--nb_channels", default=4, type=int)
    parser.add_argument("--force_train", dest='force_train', action='store_true')
    parser.set_defaults(force_train=False)
    parser.add_argument("--force_cpu", dest='force_cpu', action='store_true')
    parser.add_argument("--dst_dir", help='full path to output directory to save logs and prediction results', type=str, default=os.getcwd())
    parser.add_argument("--data_dir", help='path to dataset dir that contain train and test folders', type=str)
    parser.add_argument("--func", help='strateg to create a wek mask, either NDVI or intensity', type=str, default="NDVI", required=False)
    parser.add_argument("--ndvi_treshold", help='NDVI treshold value to create binay mask', type=float,default=0.2,required=False)
    parser.add_argument("--intensity_treshold", help='intenity treshold to create binary mask', type=int, default=120,required=False)
    parser.add_argument("--contrast_treshold", help='Gray scale imge contrast treshold to screen image ', type=float, default=0,required=False)
    parser.add_argument("--brightness_treshold", help='Gray scale image brightness treshold to sreen image', type=float, default=0,required=False)
    parser.add_argument("--equalize", help='if func is intensity, whether to run histogram equaliation', dest='equalize', action='store_true')
    parser.set_defaults(equalize=False) # c_treshold
    
    parser.add_argument("--with_prob", help='To return the probability aray for a mask fr trainset', dest='with_prob', action='store_true')
    parser.set_defaults(with_prob=False) # c_treshold
    
    parser.add_argument("--with_mask", help='Whether to return the mask during testing phase', dest='with_mask', action='store_true')
    parser.set_defaults(with_mask=False) # c_treshold

    return parser.parse_args()

def load_ssae(args):
    if args.model == "ssae":
        print(f'with specified model param { args.model}: self-supervised autoencoder will be loaded')
        model = SSAE(latent_img_size=args.latent_img_size,
            z_dim=args.z_dim,
            img_size=args.img_size,
            nb_channels=args.nb_channels,
            lamda=args.lamda,
        )
    if args.model == "ssvae":
        print(f'with specified model param { args.model}: self-supervised variational autoencoder will be loaded')
        model = SSVAE(latent_img_size=args.latent_img_size,
            z_dim=args.z_dim,
            img_size=args.img_size,
            nb_channels=args.nb_channels,
        )
    if args.model == "ss_cvae":
        print(f'with specified model param { args.model}: self-supervised conditional variational autoencoder will be loaded')
        model = SS_CVAE(latent_img_size=args.latent_img_size,
                      z_dim=args.z_dim,
                      img_size=args.img_size,
                      nb_channels=args.nb_channels,
                      mask_nb_channel=args.nb_channels
        )

    return model

def load_model_parameters(model, file_name, dir1, device):
    print(f"Trying to load: {file_name}")
    state_dict = torch.load(
        os.path.join(dir1, file_name),
        map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print(f"{file_name} loaded !")

    return model

def get_train_dataloader(args):
    print(f'The histogram equalizaion is set to: {args.equalize}')
    print(f'The ndvi treshold is set to: {args.ndvi_treshold}')
    if os.path.exists(args.data_dir):
        train_dataset = TrainDataset(
            root=args.data_dir,
            func=args.func,
            equalize=args.equalize,
            nb_channels=args.nb_channels,
            ndvi_treshold=args.ndvi_treshold,
            intensity_treshold=args.intensity_treshold,
            fake_dataset_size=1024,
            c_treshold=args.contrast_treshold,
            b_treshold=args.brightness_treshold,
            with_prob=args.with_prob
        )
    else:
        raise RuntimeError("No / Wrong file folder provided")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    return train_dataloader

def get_test_dataloader(args, fake_dataset_size=None): # categ=None is added
    if os.path.exists(args.data_dir):
        test_dataset = TestDataset(args.data_dir,
                                   fake_dataset_size=fake_dataset_size,
                                   func=args.func,
                                   equalize=args.equalize,
                                   ndvi_treshold=args.ndvi_treshold,
                                   intensity_treshold=args.intensity_treshold,
                                   nb_channels=args.nb_channels,
                                   c_treshold=args.contrast_treshold,
                                   b_treshold=args.brightness_treshold,
                                   with_mask=args.with_mask)
        print(f'Test datset size: {len(test_dataset)}')
    else:
        raise RuntimeError("No / Wrong file folder provided")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        num_workers=2,
        drop_last=True
    )  # 

    return test_dataloader

def tensor_img_to_01(t, share_B=False):
    ''' t is a BxCxHxW tensor, put its values in [0, 1] for each batch element
    if share_B is False otherwise normalization include all batch elements
    '''
    t = torch.nan_to_num(t)
    if share_B:
        t = ((t - torch.amin(t, dim=(0, 1, 2, 3), keepdim=True)) /
            (torch.amax(t, dim=(0, 1, 2, 3), keepdim=True) - torch.amin(t,
            dim=(0, 1, 2, 3),
            keepdim=True)))
    if not share_B:
        t = ((t - torch.amin(t, dim=(1, 2, 3), keepdim=True)) /
            (torch.amax(t, dim=(1, 2, 3), keepdim=True) - torch.amin(t, dim=(1, 2,3),
            keepdim=True)))
    return t

def update_loss_dict(ld_old, ld_new):
    for k, v in ld_new.items():
        if k in ld_old:
            ld_old[k] += v
        else:
            ld_old[k] = v
    return ld_old

def print_loss_logs(f_name, out_dir, loss_dict, epoch, exp_name):
    if epoch == 0:
        with open(f_name, "w") as f:
            print("epoch,", end="", file=f)
            for k, v in loss_dict.items():
                print(f"{k},", end="", file=f)
            print("\n", end="", file=f)
    # then, at every epoch
    with open(f_name, "a") as f:
        print(f"{epoch + 1},", end="", file=f)
        for k, v in loss_dict.items():
            print(f"{v},", end="", file=f)
        print("\n", end="", file=f)
    if (epoch + 1) % 50 == 0 or epoch in [4, 9, 24]:
        # with this delimiter one spare column will be detected
        arr = np.genfromtxt(f_name, names=True, delimiter=",")
        fig, axis = plt.subplots(1)
        for i, col in enumerate(arr.dtype.names[1:-1]):
            axis.plot(arr[arr.dtype.names[0]], arr[col], label=col)
        axis.legend()
        fig.savefig(os.path.join(out_dir,
            f"{exp_name}_loss_{epoch + 1}.png"))
        plt.close(fig) 
