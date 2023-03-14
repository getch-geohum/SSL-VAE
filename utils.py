import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import DataLoader
from datasets import *
from mvtec_dataset import MvtechTrainDataset, MvtechTestDataset
from AE import SSAE
from ssvae import SSVAE
from c_ssvae import SS_CVAE
from mvtec_models import SS_AEmvtec, SS_CVAEmvtec
import time
import json
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
    parser.add_argument("--with_prob", help='To return the probability aray for a mask fr trainset', dest='with_prob', action='store_true')
    parser.set_defaults(with_prob=False) # c_treshold
    parser.add_argument("--with_mask", help='Whether to return the mask during testing phase', dest='with_mask', action='store_true')
    parser.set_defaults(with_mask=False) # c_treshold

    parser.add_argument("--max_beta", help="Beta term for kld annealing, if anneal if true, it will progress from min_beta", default=0.5, type=float)
    parser.add_argument("--min_beta", help="Base beta value for kld annealing", default=0.0, type=float)
    parser.add_argument("--anneal_beta", help='Whether to anneal beta or not', dest='anneal_beta',action='store_true')
    parser.set_defaults(anneal_beta=False)  # save_preds

    parser.add_argument("--anneal_cyclic", help='Whether to follow cclic annealing schedule', dest='anneal_cyclic',action='store_true')
    parser.set_defaults(anneal_cyclic=False)

    parser.add_argument("--save_preds", help='Whether to save predicted images during tesing phase', dest='save_preds',action='store_true')
    parser.set_defaults(save_preds=False)

    parser.add_argument("--dataset", help="Train test dataset type, either of 'camp' or 'mvtec' ", type=str, default='camp')
    parser.add_argument("--texture", help="One of the texture classes from MVtech dataset or 'all' ", type=str, default='carpet')

    parser.add_argument("--validate", help='Whether to run the test on validation samples provided in test "good" folder', dest='validate', action='store_true')
    parser.set_defaults(validate=False) # validation 

    parser.add_argument("--cycle", help='The number of ccles to oscilate the beta value within a total training steps', default=4, type=int)
    parser.add_argument("--ratio", help='ratio of training steps within a cycle where beta is suppose to anneal and the rest is kept to B==1', default=0.5, type=float)
    parser.add_argument("--data", help='The data folder to use for training and testing, either one folder or "all" ', default='all', type=str)
    return parser.parse_args()

# parser.add_argument("--func", help='strateg to create a wek mask, either NDVI or intensity', type=str, default="NDVI", required=False)
# parser.add_argument("--ndvi_treshold", help='NDVI treshold value to create binay mask', type=float,default=0.2,required=False)
# parser.add_argument("--intensity_treshold", help='intenity treshold to create binary mask', type=int, default=120,required=False)
# parser.add_argument("--contrast_treshold", help='Gray scale imge contrast treshold to screen image ', type=float, default=0,required=False)
# parser.add_argument("--brightness_treshold", help='Gray scale image brightness treshold to sreen image', type=float, default=0,required=False)
# parser.add_argument("--equalize", help='if func is intensity, whether to run histogram equaliation', dest='equalize', action='store_true')
# parser.set_defaults(equalize=False) # c_treshold

def computeLinearBeta(num_epochs=100, steps_per_epoch=20, cycle=4, ratio=0.5, plot=False):
    '''
    The function used to create beta values based on cyclic annealing with linear increase strategy
    num_epochs: nubber of training epochs where the model go through full datasets
    steps_per_epoch:number of data setps within a single epoch
    cycle: Number of cycles to oscilate the beta value
    ratio: The ratio of increase steps from total steps within a single annealing cycle
    plot: Boolean value whether to plot the computed beta values
    '''
    period = int(num_epochs/cycle)
    period_num_steps = period*steps_per_epoch
    increase_steps = period_num_steps*ratio
    scale = 1/increase_steps
    
    data = []
    
    for i in range(cycle):
        data+=[np.round(scale*i, 4) for i in range(int(increase_steps))]
        data+=[1]*int(period_num_steps-increase_steps)
    if plot:
        plt.plot(data)
        plt.show()
    return data

def load_ssae(args):
    if args.model == 'mv_ae':
        model = SS_AEmvtec(zdim=args.z_dim)
    if args.model == 'mv_cvae':
        model = SS_CVAEmvtec(zdim=args.z_dim)
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
        print(f'with specified model param { args.model} without kld annealing: self-supervised conditional variational autoencoder will be loaded')
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
    # print(f'The histogram equalizaion is set to: {args.equalize}')
    # print(f'The ndvi treshold is set to: {args.ndvi_treshold}')
    
    if os.path.exists(args.data_dir):
        if args.dataset == 'camp':
            print(f'All CAMP dataset will be loaded for training from {args.data_dir}')
            # assert os.path.exists(args.params_file), f'The preprocessing params file {args.params_file} is not existing, please check'
            with open('params_file.txt', "r") as params_log:
                params = json.load(params_log)
            if args.data == 'all':
                folds = list(os.listdir(args.data_dir)) # list of datasaet folders containing training and testing data
                inds = [folds.index(fold) for fold in folds] # index o each folder in the list for later use
                paths = [args.data_dir + f'/{folder}' for folder in folds] # dataset paths taken directly from the list
                train_dataset = [TrainDataset(root=paths[ind],
                                              func=params[folds[ind]]['func'],
                                              equalize=params[folds[ind]]['equalize'],
                                              nb_channels=args.nb_channels,
                                              ndvi_treshold=params[folds[ind]]['ndvi_treshold'],
                                              intensity_treshold=params[folds[ind]]['intensity_treshold'],
                                              fake_dataset_size=args.fake_dataset_size,
                                              c_treshold=params[folds[ind]]['contrast_treshold'],
                                              b_treshold=params[folds[ind]]['brightness_treshold'],
                                              with_prob=args.with_prob) for ind in inds
                                ]
                #train_dataset = torch.utils.data.ConcatDataset(train_dataset)
                print(f'Concatenated camp test datset size: {len(train_dataset)}')
            else:
                print(f'{args.data} CAMP dataset will be loaded for training from {args.data_dir}')
                path = args.data_dir + f'/{args.data}'
                train_dataset = TrainDataset(root=path,
                                             func=params[args.data]['func'],
                                             equalize=params[args.data]['equalize'],
                                             nb_channels=args.nb_channels,
                                             ndvi_treshold=params[args.data]['ndvi_treshold'],
                                             intensity_treshold=params[args.data]['intensity_treshold'],
                                             fake_dataset_size=args.fake_dataset_size,
                                             c_treshold=params[args.data]['contrast_treshold'],
                                             b_treshold=params[args.data]['brightness_treshold'],
                                             with_prob=args.with_prob
                                            )
        else:
            print('MVtECH dataset will be loaded for training')
            train_dataset = MvtechTrainDataset(root=args.data_dir, texture=args.texture, with_prob=args.with_prob)
    else:
        raise RuntimeError("No / Wrong file folder provided")

    print(f'Final train dataset length: {len(train_dataset)}')
    if args.data == 'all':
        train_dataloader = [DataLoader(train_dataset[i],
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=12,
                                  drop_last=True
                                 ) for i in range(len(train_dataset))]

    else:
        train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=12,
                                  drop_last=True
                                 )

    return train_dataloader

def get_test_dataloader(args, fake_dataset_size=None): # categ=None is added
    if os.path.exists(args.data_dir):
        if args.dataset == 'camp':
            print(f'All CAMP dataset will be loaded for training from {args.data_dir}')
            # assert os.path.exists(args.params_file), f'The preprocessing params file {args.params_file} is not existing, please check'
            with open("params_file.txt", "r") as params_log:
                params = json.load(params_log)
            if args.data == 'all':
                print(f'All CAMP dataset will be loaded for testing from {args.data_dir}')
                folds = list(os.listdir(args.data_dir)) # list of datasaet folders containing training and testing data
                inds = [folds.index(fold) for fold in folds] # index o each folder in the list for later use
                paths = [args.data_dir + f'/{folder}' for folder in folds] # dataset paths taken directly from the list
                test_dataset = [TestDataset(paths[ind],
                                   fake_dataset_size=fake_dataset_size,
                                   func=params[folds[ind]]['func'],
                                   equalize=params[folds[ind]]['equalize'],
                                   ndvi_treshold=params[folds[ind]]['ndvi_treshold'],
                                   intensity_treshold=params[folds[ind]]['intensity_treshold'],
                                   nb_channels=args.nb_channels,
                                   c_treshold=params[folds[ind]]['contrast_treshold'],
                                   b_treshold=params[folds[ind]]['brightness_treshold'],
                                   with_mask=args.with_mask) for ind in inds
                               ]
                #test_dataset =  torch.utils.data.ConcatDataset(test_dataset)
                print(f'Concatenated camp test datset size: {len(test_dataset)}')
            else:
                print(f'{args.data} CAMP dataset will be loaded for training from director {args.data_dir}')
                path = args.data_dir + f'/{args.data}'
                test_dataset = TestDataset(path,
                                           fake_dataset_size=fake_dataset_size,
                                           func=params[args.data]['func'],
                                           equalize=params[args.data]['equalize'],
                                           ndvi_treshold=params[args.data]['ndvi_treshold'],
                                           intensity_treshold=params[args.data]['intensity_treshold'],
                                           nb_channels=args.nb_channels,
                                           c_treshold=params[args.data]['contrast_treshold'],
                                           b_treshold=params[args.data]['brightness_treshold'],
                                           with_mask=args.with_mask
                                          )
            print(f'Camp test datset size: {len(test_dataset)}')
        else:
            test_dataset = MvtechTestDataset(root=args.data_dir, texture=args.texture, fake_dataset_size=fake_dataset_size, validate=args.validate)
            print(f'Mvtech test datset size: {len(test_dataset)}')
    else:
        raise RuntimeError("No / Wrong file folder provided")
    print(f'Final test dataset lengt: {len(test_dataset)}')
    if args.data == 'all':
        test_dataloader = [DataLoader(test_dataset[i],
        batch_size=args.batch_size_test,
        num_workers=12,
        drop_last=True) for i in range(len(test_dataset))]
    else:
        test_dataloader = DataLoader(test_dataset,
        batch_size=args.batch_size_test,
        num_workers=12,
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
